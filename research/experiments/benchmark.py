"""Benchmark decoders on public datasets and track runs in MLflow.

    uv sync --extra research [--extra deep]
    uv run python research/experiments/benchmark.py --config research/configs/baselines.yaml
    uv run python research/experiments/benchmark.py --config research/configs/deep.yaml
                                                    [--only NAME ...] [--no-tracking]

For every experiment of the configuration, evaluates each model under each
protocol, writes the result tables to the configuration's ``output`` file and,
unless --no-tracking is given, logs one MLflow run per experiment (with a
nested run per model and protocol) to $OPENMIND_DATA_DIR/mlflow. Deep models
listed under ``pretrain`` are then trained on all subjects of the experiment and
saved to the local registry ($OPENMIND_DATA_DIR/models). Preprocessed epochs
are cached in $OPENMIND_DATA_DIR/cache/epochs.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.metadata
import json
import logging
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

import openmind_core.datasets.moabb as loader
import openmind_core.preprocessing as preprocessing
from openmind_core.datasets import DATASETS, MONTAGE_A_MI, EpochSet, load_epochs
from openmind_core.evaluation import (
    Score,
    Summary,
    drop_small_subjects,
    evaluate,
    summarize,
)
from openmind_core.models import make_model
from openmind_core.paths import data_dir

ROOT = Path(__file__).resolve().parents[2]
PACKAGES = (
    "openmind-core",
    "numpy",
    "scipy",
    "mne",
    "moabb",
    "pyriemann",
    "scikit-learn",
    "torch",
    "braindecode",
)

logger = logging.getLogger("benchmark")


@dataclass
class Result:
    """Outcome of one experiment."""

    experiment: dict[str, Any]
    epochs: EpochSet
    summaries: list[Summary]
    elapsed: float
    exported: list[str] = field(default_factory=list)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--only", nargs="+", help="run only these experiments")
    parser.add_argument("--no-tracking", action="store_true", help="do not log to MLflow")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    config = yaml.safe_load(args.config.read_text())
    config["config_path"] = args.config.resolve().relative_to(ROOT).as_posix()
    experiments = [e for e in config["experiments"] if not args.only or e["name"] in set(args.only)]
    code = code_version()
    versions = package_versions()
    results: list[Result] = []
    for experiment in experiments:
        epochs, dropped = drop_small_subjects(
            cached_epochs(experiment), config["min_epochs_per_class"]
        )
        experiment = experiment | {"dropped": dropped}
        start = time.perf_counter()
        scores = evaluate(
            epochs,
            {name: factory(config, name, epochs.srate) for name in config["models"]},
            protocols=config["protocols"],
            n_splits=config["n_splits"],
            seed=config["seed"],
        )
        result = Result(
            experiment=experiment,
            epochs=epochs,
            summaries=summarize(scores, n_classes=len(experiment["classes"])),
            elapsed=time.perf_counter() - start,
        )
        for summary in result.summaries:
            logger.info(
                "%s %s %s: %.1f ± %.1f %%",
                experiment["name"],
                summary.protocol,
                summary.pipeline,
                100 * summary.accuracy_mean,
                100 * summary.accuracy_std,
            )
        result.exported = pretrain(config, result, code)
        if not args.no_tracking:
            track(config, result, scores, code, versions)
        results.append(result)

    output = ROOT / config["output"]
    if args.only:
        logger.info("partial run (--only): %s left unchanged", config["output"])
    else:
        write_report(config, results, code, versions, output)
        logger.info("results written to %s", config["output"])


def factory(config: dict[str, Any], name: str, srate: float) -> Any:
    """A function building a fresh model with the configuration's settings."""
    params = (config.get("model_params") or {}).get(name, {})
    return lambda: make_model(name, srate=srate, seed=config["seed"], **params)


def pretrain(config: dict[str, Any], result: Result, code: dict[str, str]) -> list[str]:
    """Train the ``pretrain`` models on every subject and save them to the registry."""
    names = config.get("pretrain") or []
    if not names:
        return []
    from openmind_core.models.registry import save_model

    experiment, epochs = result.experiment, result.epochs
    exported = []
    for name in names:
        model_name = f"{name}-{experiment['name']}"
        logger.info("pretrain %s on %d epochs", model_name, len(epochs))
        model = factory(config, name, epochs.srate)().fit(epochs.X, epochs.y)
        scores = {
            s.protocol: round(s.accuracy_mean, 4) for s in result.summaries if s.pipeline == name
        }
        training = {
            "dataset": experiment["dataset"],
            "subjects": [s for s in experiment["subjects"] if s not in experiment["dropped"]],
            "classes": list(experiment["classes"]),
            "n_epochs": len(epochs),
            "preprocessing": epochs.info.get("filters", {}),
            "benchmark_accuracy": scores,
            "benchmark_config": config["config_path"],
            "git_commit": code["commit"],
            "epochs_trained": len(model.history_),
        }
        save_model(model, model_name, epochs.channels, training=training, overwrite=True)
        exported.append(model_name)
    return exported


def cached_epochs(experiment: dict[str, Any]) -> EpochSet:
    """Load the epochs of an experiment, from the cache when possible.

    The cache key includes the source code of the loader and of the
    preprocessing, so any change to them invalidates the cached epochs.
    """
    channels = list(experiment.get("channels") or MONTAGE_A_MI)
    key = json.dumps(
        {k: experiment[k] for k in ("dataset", "subjects", "classes")}
        # The default montage is left out of the key so existing caches stay valid.
        | ({"channels": channels} if channels != list(MONTAGE_A_MI) else {}),
        sort_keys=True,
    )
    digest_input = key.encode() + source_bytes(loader.__file__)
    for module in sorted(Path(preprocessing.__file__).parent.glob("*.py")):
        digest_input += source_bytes(str(module))
    digest = hashlib.sha256(digest_input).hexdigest()[:12]
    path = data_dir() / "cache" / "epochs" / f"{experiment['dataset']}_{digest}.npz"
    if path.exists():
        logger.info("epochs from cache %s", path)
        return EpochSet.load(path)
    epochs = load_epochs(
        experiment["dataset"], experiment["subjects"], experiment["classes"], channels=channels
    )
    epochs.save(path)
    return epochs


def source_bytes(path: str | None) -> bytes:
    return Path(path).read_bytes() if path else b""


def package_versions() -> dict[str, str]:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def code_version() -> dict[str, str]:
    """Commit of the repository and whether the working tree has changes."""

    def git(*command: str) -> str:
        try:
            return subprocess.run(
                ["git", *command], cwd=ROOT, capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    return {
        "commit": git("rev-parse", "--short", "HEAD") or "unknown",
        # Result files are outputs of these scripts: they do not make the code dirty.
        "dirty": "true"
        if git("status", "--porcelain", "--untracked-files=no", "--", ".", ":!research/results")
        else "false",
    }


def track(
    config: dict[str, Any],
    result: Result,
    scores: list[Score],
    code: dict[str, str],
    versions: dict[str, str],
) -> None:
    """Log one experiment to MLflow: a parent run, one nested run per model and protocol."""
    import mlflow

    store = data_dir(create=True) / "mlflow"
    store.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{store / 'mlflow.db'}")
    name = config["name"]
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(store / "artifacts").as_uri())
    mlflow.set_experiment(name)

    experiment, epochs = result.experiment, result.epochs
    tags = {"git_commit": code["commit"], "git_dirty": code["dirty"], "task": config["task"]}
    tags |= {f"version.{package}": version for package, version in versions.items()}
    data_params = {
        "config": config["config_path"],
        "dataset": experiment["dataset"],
        "subjects": ",".join(map(str, experiment["subjects"])),
        "classes": ",".join(experiment["classes"]),
        "channels": ",".join(epochs.channels),
        "srate": epochs.srate,
        "n_epochs": len(epochs),
        "seed": config["seed"],
        "n_splits": config["n_splits"],
        "min_epochs_per_class": config["min_epochs_per_class"],
        "dropped_subjects": ",".join(map(str, experiment["dropped"])) or "none",
    }
    with mlflow.start_run(run_name=experiment["name"], tags=tags):
        mlflow.log_params(data_params | {"exported_models": ",".join(result.exported) or "none"})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scores.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(Score.__dataclass_fields__))
                writer.writeheader()
                writer.writerows(asdict(score) for score in scores)
            mlflow.log_artifact(str(path))
        for summary in result.summaries:
            model_params = (config.get("model_params") or {}).get(summary.pipeline, {})
            with mlflow.start_run(
                run_name=f"{summary.pipeline}/{summary.protocol}", nested=True, tags=tags
            ):
                mlflow.log_params(
                    data_params
                    | {"model": summary.pipeline, "protocol": summary.protocol}
                    | {f"model.{key}": value for key, value in model_params.items()}
                )
                mlflow.log_metrics(
                    {
                        "accuracy_mean": summary.accuracy_mean,
                        "accuracy_std": summary.accuracy_std,
                        "kappa_mean": summary.kappa_mean,
                        "chance": summary.chance,
                    }
                )
                for score in scores:
                    if (score.pipeline, score.protocol) == (summary.pipeline, summary.protocol):
                        key = f"accuracy_s{score.subject:03d}_{score.session}"
                        mlflow.log_metric(key, score.accuracy)


def write_report(
    config: dict[str, Any],
    results: list[Result],
    code: dict[str, str],
    versions: dict[str, str],
    output: Path,
) -> None:
    """Write the Markdown result tables."""
    protocols = config["protocols"]
    lines = [
        f"# {config['title']}",
        "",
        "Generated by `uv run python research/experiments/benchmark.py --config",
        f"{config['config_path']}`. Do not edit by hand.",
        "",
        "Accuracy in %, mean ± standard deviation over subjects (sessions averaged",
        "within each subject first); κ is Cohen's kappa. Preprocessing: causal",
        "8-30 Hz band-pass and notch, 2 s epochs from cue + 0.5 s, montage A",
        "decoding channels (`FC3, FC4, C3, Cz, C4, CP3, CP4`), 250 Hz.",
        "See `docs/evaluation.md` for the protocols and the discussion.",
        "",
    ]
    for result in results:
        experiment = result.experiment
        spec = DATASETS[experiment["dataset"]]
        chance = 100 / len(experiment["classes"])
        dropped = experiment["dropped"]
        excluded = (
            f" Subjects {', '.join(map(str, dropped))} excluded (fewer than "
            f"{config['min_epochs_per_class']} epochs per class after rejection)."
            if dropped
            else ""
        )
        lines += [
            f"## {experiment['name']}",
            "",
            f"{spec.description} Subjects {format_subjects(experiment['subjects'])}, "
            f"classes {', '.join(experiment['classes'])} (chance {chance:.0f} %), "
            f"{len(result.epochs)} epochs after rejection.{excluded} "
            f"Evaluation time {format_duration(result.elapsed)}.",
            "",
            "| Model | " + " | ".join(protocols) + " |",
            "|---|" + "---|" * len(protocols),
        ]
        for model in config["models"]:
            cells = []
            for protocol in protocols:
                match = [
                    s for s in result.summaries if (s.pipeline, s.protocol) == (model, protocol)
                ]
                if match:
                    s = match[0]
                    cells.append(
                        f"{100 * s.accuracy_mean:.1f} ± {100 * s.accuracy_std:.1f} "
                        f"(κ {s.kappa_mean:.2f})"
                    )
                else:
                    cells.append("n/a")
            lines.append(f"| `{model}` | " + " | ".join(cells) + " |")
        if result.exported:
            names = ", ".join(f"`{name}`" for name in result.exported)
            lines += ["", f"Pre-trained on all subjects and saved to the registry: {names}."]
        lines.append("")
    dirty = " (with uncommitted changes)" if code["dirty"] == "true" else ""
    lines += [
        "## Run",
        "",
        f"- Date: {dt.date.today().isoformat()}",
        f"- Code: commit `{code['commit']}`{dirty}",
        f"- Seed {config['seed']}, {config['n_splits']}-fold within-session",
    ]
    for model, params in (config.get("model_params") or {}).items():
        settings = ", ".join(f"{key}={value}" for key, value in params.items())
        lines.append(f"- `{model}`: {settings}")
    lines += [
        "- Versions: " + ", ".join(f"{name} {version}" for name, version in versions.items()),
        "",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))


def format_subjects(subjects: list[int]) -> str:
    if subjects == list(range(subjects[0], subjects[-1] + 1)):
        return f"{subjects[0]}-{subjects[-1]}"
    return ", ".join(map(str, subjects))


def format_duration(seconds: float) -> str:
    return f"{seconds:.0f} s" if seconds < 120 else f"{seconds / 60:.0f} min"


if __name__ == "__main__":
    main()
