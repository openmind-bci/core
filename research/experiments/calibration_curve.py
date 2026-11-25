"""Calibration curve: how many calibration trials does a new user need?

    uv sync --extra research
    uv run python research/experiments/calibration_curve.py \\
        --config research/configs/calibration-curve.yaml [--only NAME ...] [--no-tracking] [--fresh]

For every experiment of the configuration and every target subject (leave one
subject out), evaluates each model under several conditions (generic, aligned,
aligned + tuned, from scratch, and optionally aligned + fine-tuned with a
strategy, or a similarity-based cold start) for an increasing number of
calibration trials per class. With ``seeds``, deep models are repeated per seed
and averaged. The protocol is documented in ``docs/evaluation.md`` ("Calibration
curve") and implemented in ``openmind_core.calibration``.

Outputs: the Markdown tables of the configuration's ``output`` file, a CSV of
every score in $OPENMIND_DATA_DIR/results/, and, unless --no-tracking is given,
one MLflow run per experiment (a nested run per model) in
$OPENMIND_DATA_DIR/mlflow. Preprocessed epochs are cached like benchmark.py does.
The run is resumable: each finished target subject is checkpointed in
$OPENMIND_DATA_DIR/cache/calibration-curve/.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import logging
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark import (  # a sibling script, not a package
    ROOT,
    cached_epochs,
    code_version,
    factory,
    format_duration,
    format_subjects,
    package_versions,
)

from openmind_core.alignment import align_epochs
from openmind_core.calibration import (
    ALL_CONDITIONS,
    CONDITIONS,
    DEFAULT_COLDSTART_K,
    DEFAULT_N_PER_CLASS,
    GENERIC,
    SCRATCH,
    CurveScore,
    CurveSummary,
    calibration_curve,
    condition_rank,
    summarize_curve,
)
from openmind_core.datasets import DATASETS
from openmind_core.evaluation import drop_small_subjects
from openmind_core.paths import data_dir

logger = logging.getLogger("calibration-curve")

SETTINGS = (
    "models",
    "model_params",
    "conditions",
    "n_per_class",
    "target_weight",
    "alignment",
    "seed",
    "min_epochs_per_class",
    "min_test_per_class",
    "unlabeled_start",
)

#: Settings that only enter the checkpoint key when the configuration sets them,
#: so that adding them did not invalidate the checkpoints of older configurations.
OPTIONAL_SETTINGS = ("seeds", "finetune", "coldstart_k")


@dataclass
class Result:
    """Outcome of one experiment."""

    experiment: dict[str, Any]
    n_epochs: int
    channels: tuple[str, ...]
    srate: float
    scores: list[CurveScore]
    elapsed: float
    summaries: list[CurveSummary] = field(default_factory=list)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--only", nargs="+", help="run only these experiments")
    parser.add_argument("--no-tracking", action="store_true", help="do not log to MLflow")
    parser.add_argument(
        "--fresh", action="store_true", help="ignore the checkpoints of finished subjects"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    config = yaml.safe_load(args.config.read_text())
    config["config_path"] = display_path(args.config.resolve())
    experiments = [e for e in config["experiments"] if not args.only or e["name"] in set(args.only)]
    code = code_version()
    versions = package_versions()
    results = []
    for experiment in experiments:
        result = run_experiment(config, experiment, code, fresh=args.fresh)
        results.append(result)
        if not args.no_tracking:
            track(config, result, code, versions)

    write_csv(config, results, partial=bool(args.only))
    if args.only:
        logger.info("partial run (--only): %s left unchanged", config["output"])
    else:
        output = ROOT / config["output"]
        write_report(config, results, code, versions, output)
        logger.info("results written to %s", output)


def display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def run_experiment(
    config: dict[str, Any], experiment: dict[str, Any], code: dict[str, str], fresh: bool = False
) -> Result:
    """Evaluate every target subject of an experiment, resuming from checkpoints."""
    epochs, dropped = drop_small_subjects(cached_epochs(experiment), config["min_epochs_per_class"])
    experiment = experiment | {"dropped": dropped}
    alignment = config.get("alignment") or {"method": "riemann"}
    conditions = config.get("conditions") or list(CONDITIONS)
    unknown = sorted(set(conditions) - set(ALL_CONDITIONS))
    if unknown:
        raise ValueError(f"unknown conditions {unknown}; available: {list(ALL_CONDITIONS)}")
    aligned = (
        align_epochs(epochs, method=alignment.get("method", "riemann"), by="session")
        if set(conditions) - {GENERIC, SCRATCH}
        else None
    )
    models = {name: factory(config, name, epochs.srate) for name in config["models"]}
    scores: list[CurveScore] = []
    elapsed = 0.0
    # `targets` limits the evaluated subjects (smoke runs); the source is always every other one.
    targets = experiment.get("targets")
    subjects = [s for s in epochs.subjects if not targets or int(s) in targets]
    for i, subject in enumerate(subjects, start=1):
        checkpoint = checkpoint_path(config, experiment, code, subject)
        if checkpoint is not None and checkpoint.exists() and not fresh:
            saved = json.loads(checkpoint.read_text())
            subject_scores = [CurveScore(**s) for s in saved["scores"]]
            took = float(saved["elapsed"])
            logger.info(
                "%s subject %s (%d/%d): from checkpoint",
                experiment["name"],
                subject,
                i,
                len(subjects),
            )
        else:
            start = time.perf_counter()
            subject_scores = calibration_curve(
                epochs,
                models,
                conditions=conditions,
                n_values=config.get("n_per_class") or DEFAULT_N_PER_CLASS,
                alignment=alignment,
                target_weight=config.get("target_weight", 10.0),
                seed=config["seed"],
                subjects=[subject],
                aligned_epochs=aligned,
                min_test_per_class=config.get("min_test_per_class", 10),
                unlabeled_start=config.get("unlabeled_start", 20),
                seeds=config.get("seeds"),
                finetune=config.get("finetune"),
                coldstart_k=config.get("coldstart_k", DEFAULT_COLDSTART_K),
            )
            took = time.perf_counter() - start
            if checkpoint is not None:
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_text(
                    json.dumps({"elapsed": took, "scores": [asdict(s) for s in subject_scores]})
                )
            logger.info(
                "%s subject %s (%d/%d): %d scores in %.0f s",
                experiment["name"],
                subject,
                i,
                len(subjects),
                len(subject_scores),
                took,
            )
        scores += subject_scores
        elapsed += took
    summaries = summarize_curve(scores)
    for s in summaries:
        logger.info(
            "%s %s %s n=%d: %.1f ± %.1f %% (%d subjects)",
            experiment["name"],
            s.model,
            s.condition,
            s.n,
            100 * s.accuracy_mean,
            100 * s.accuracy_std,
            s.n_subjects,
        )
    return Result(
        experiment=experiment,
        n_epochs=len(epochs),
        channels=epochs.channels,
        srate=epochs.srate,
        scores=scores,
        elapsed=elapsed,
        summaries=summaries,
    )


def checkpoint_path(
    config: dict[str, Any], experiment: dict[str, Any], code: dict[str, str], subject: int
) -> Path | None:
    """Where the scores of one target subject are saved, so that a new run can skip them.

    The key covers the experiment, the models and protocol settings and the Git
    tree of ``src/``; without a clean ``src/`` there is no checkpoint.
    """
    if code["src_tree"] == "dirty":
        return None
    settings = {k: config.get(k) for k in SETTINGS}
    settings |= {k: config[k] for k in OPTIONAL_SETTINGS if k in config}
    key = json.dumps(
        {
            "experiment": experiment,
            "settings": settings,
            "src": code["src_tree"],
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(key.encode()).hexdigest()[:12]
    folder = data_dir() / "cache" / "calibration-curve" / config["name"]
    return folder / f"{experiment['name']}-{digest}" / f"s{subject:03d}.json"


def write_csv(config: dict[str, Any], results: list[Result], partial: bool) -> Path:
    """Every score of the run, one row each, outside the repository."""
    suffix = "-partial" if partial else ""
    path = data_dir(create=True) / "results" / f"{config['name']}{suffix}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["experiment", *CurveScore.__dataclass_fields__]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            for score in result.scores:
                writer.writerow({"experiment": result.experiment["name"]} | asdict(score))
    logger.info("scores written to %s", path)
    return path


def track(
    config: dict[str, Any], result: Result, code: dict[str, str], versions: dict[str, str]
) -> None:
    """Log one experiment to MLflow: a parent run, one nested run per model.

    The curves are logged as metrics ``accuracy.<condition>`` with the number of
    trials per class as the step.
    """
    import mlflow

    store = data_dir(create=True) / "mlflow"
    store.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{store / 'mlflow.db'}")
    name = config["name"]
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(store / "artifacts").as_uri())
    mlflow.set_experiment(name)

    experiment = result.experiment
    tags = {"git_commit": code["commit"], "git_dirty": code["dirty"], "task": config["task"]}
    tags |= {f"version.{package}": version for package, version in versions.items()}
    params = {
        "config": config["config_path"],
        "dataset": experiment["dataset"],
        "subjects": ",".join(map(str, experiment["subjects"])),
        "classes": ",".join(experiment["classes"]),
        "channels": ",".join(result.channels),
        "srate": result.srate,
        "n_epochs": result.n_epochs,
        "seed": config["seed"],
        "n_per_class": ",".join(map(str, config.get("n_per_class") or DEFAULT_N_PER_CLASS)),
        "conditions": ",".join(config.get("conditions") or CONDITIONS),
        "target_weight": config.get("target_weight", 10.0),
        "alignment": json.dumps(config.get("alignment") or {"method": "riemann"}),
        "min_test_per_class": config.get("min_test_per_class", 10),
        "unlabeled_start": config.get("unlabeled_start", 20),
        "seeds": ",".join(map(str, config.get("seeds") or [])) or "none",
        "finetune": json.dumps(config.get("finetune") or {}),
        "coldstart_k": config.get("coldstart_k", DEFAULT_COLDSTART_K),
        "dropped_subjects": ",".join(map(str, experiment["dropped"])) or "none",
    }
    with mlflow.start_run(run_name=experiment["name"], tags=tags):
        mlflow.log_params(params)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scores.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(CurveScore.__dataclass_fields__))
                writer.writeheader()
                writer.writerows(asdict(score) for score in result.scores)
            mlflow.log_artifact(str(path))
        for model in config["models"]:
            model_params = (config.get("model_params") or {}).get(model, {})
            with mlflow.start_run(run_name=model, nested=True, tags=tags):
                mlflow.log_params(
                    params
                    | {"model": model}
                    | {f"model.{key}": value for key, value in model_params.items()}
                )
                for s in result.summaries:
                    if s.model != model:
                        continue
                    condition = s.condition.replace("+", "_")
                    mlflow.log_metric(f"accuracy.{condition}", s.accuracy_mean, step=s.n)
                    mlflow.log_metric(f"accuracy_std.{condition}", s.accuracy_std, step=s.n)
                    mlflow.log_metric(f"kappa.{condition}", s.kappa_mean, step=s.n)


def write_report(
    config: dict[str, Any],
    results: list[Result],
    code: dict[str, str],
    versions: dict[str, str],
    output: Path,
) -> None:
    """Write the Markdown result tables: per dataset and model, conditions by n."""
    lines = [
        f"# {config['title']}",
        "",
        "Generated by `uv run python research/experiments/calibration_curve.py --config",
        f"{config['config_path']}`. Do not edit by hand.",
        "",
        "Accuracy in %, mean ± standard deviation over target subjects (sessions",
        "averaged within each subject first), with Cohen's κ in parentheses. Columns:",
        "calibration trials per class. The main tables use the **common test set**, the",
        "trials after the largest feasible calibration set of each session, so that",
        "the columns are comparable; the second table of each model uses the test set",
        "of each column (all trials after its own calibration). `aligned` at 0 trials",
        "builds its reference from the first "
        f"{config.get('unlabeled_start', 20)} trials of the session without",
        "labels (transductive start); those trials are never tested. See",
        "`docs/evaluation.md` (Calibration curve) for the protocol.",
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
            f"{result.n_epochs} epochs after rejection.{excluded} "
            f"Evaluation time {format_duration(result.elapsed)}.",
            "",
        ]
        for model in config["models"]:
            lines += model_tables(model, result)
    dirty = " (with uncommitted changes)" if code["dirty"] == "true" else ""
    alignment = config.get("alignment") or {"method": "riemann"}
    n_values = config.get("n_per_class") or DEFAULT_N_PER_CLASS
    lines += [
        "## Run",
        "",
        f"- Date: {dt.date.today().isoformat()}",
        f"- Code: commit `{code['commit']}`{dirty}",
        f"- Seed {config['seed']}, alignment {alignment.get('method', 'riemann')} per session",
        f"- Calibration trials per class: {', '.join(map(str, n_values))}"
        f"; at least {config.get('min_test_per_class', 10)} test trials per class",
        f"- Weight of the calibration trials in `aligned+tuned`: "
        f"{config.get('target_weight', 10.0)}",
    ]
    if config.get("seeds"):
        lines.append(
            f"- Deep models repeated over seeds {', '.join(map(str, config['seeds']))} "
            "(source pre-training included) and averaged"
        )
    if config.get("finetune"):
        lines.append(f"- Fine-tuning: {json.dumps(config['finetune'])}")
    if any(c.startswith("coldstart") for c in config.get("conditions") or []):
        lines.append(
            f"- `coldstart-knn`: {config.get('coldstart_k', DEFAULT_COLDSTART_K)} nearest source "
            "subjects (Riemannian distance between unlabeled-start covariances), n = 0 only"
        )
    for model, params in (config.get("model_params") or {}).items():
        settings = ", ".join(f"{key}={value}" for key, value in params.items())
        lines.append(f"- `{model}`: {settings}")
    lines += [
        "- Versions: " + ", ".join(f"{name} {version}" for name, version in versions.items()),
        "",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))


def model_tables(model: str, result: Result) -> list[str]:
    summaries = [s for s in result.summaries if s.model == model]
    if not summaries:
        return []
    ns = sorted({s.n for s in summaries})
    conditions = sorted({s.condition for s in summaries}, key=condition_rank)
    by_key = {(s.condition, s.n): s for s in summaries}
    header = ["| Condition | " + " | ".join(f"{n}" for n in ns) + " |", "|---|" + "---|" * len(ns)]
    main_rows, own_rows = [], []
    for condition in conditions:
        cells, own = [], []
        for n in ns:
            s = by_key.get((condition, n))
            cells.append(
                f"{100 * s.accuracy_mean:.1f} ± {100 * s.accuracy_std:.1f} (κ {s.kappa_mean:.2f})"
                if s
                else "-"
            )
            own.append(f"{100 * s.accuracy_own_mean:.1f}" if s else "-")
        main_rows.append(f"| `{condition}` | " + " | ".join(cells) + " |")
        own_rows.append(f"| `{condition}` | " + " | ".join(own) + " |")
    counts = [max((s.n_subjects for s in summaries if s.n == n), default=0) for n in ns]
    tests = [
        sum(s.n_test_mean for s in summaries if s.n == n)
        / max(sum(1 for s in summaries if s.n == n), 1)
        for n in ns
    ]
    main_rows.append("| subjects | " + " | ".join(map(str, counts)) + " |")
    main_rows.append("| test trials (mean) | " + " | ".join(f"{t:.0f}" for t in tests) + " |")
    return [
        f"### `{model}`",
        "",
        "Common test set:",
        "",
        *header,
        *main_rows,
        "",
        "Test set of each column (accuracy only):",
        "",
        *header,
        *own_rows,
        "",
    ]


if __name__ == "__main__":
    main()
