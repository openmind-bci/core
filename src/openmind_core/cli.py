"""Command-line interface for ``openmind-core``.

Subcommands:

* ``info``: versions and data directory;
* ``profile create|list|show|delete``: local pseudonymized user profiles;
* ``models list|show|delete``: the local model registry;
* ``train``: train a model on a public dataset and register it;
* ``bandpower``: publish the band power per electrode for the 3D brain;
* ``online``: decode the EEG stream with a registered model and publish commands.
"""

import json
from typing import Annotated, Any

import typer

from openmind_core import __version__
from openmind_core.profiles import ProfileStore

app = typer.Typer(
    name="openmind-core",
    help="Decode user intent from a standard EEG stream and publish OpenMind commands.",
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    if value:
        from openmind_protocol import __version__ as protocol_version

        typer.echo(f"openmind-core {__version__} (protocol {protocol_version})")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Show the version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """OpenMind Core: EEG decoding, fast calibration and command publishing."""


@app.command("info")
def info() -> None:
    """Show versions and the local data directory."""
    import os
    from pathlib import Path

    from openmind_protocol import __version__ as protocol_version

    value = os.environ.get("OPENMIND_DATA_DIR", "").strip()
    data_dir = Path(value).expanduser() if value else Path.home() / "openmind-data"
    typer.echo(f"openmind-core      {__version__}")
    typer.echo(f"openmind-protocol  {protocol_version}")
    typer.echo(f"data directory     {data_dir}")


profile_app = typer.Typer(
    help="Manage local user profiles (pseudonymized ids such as P01, never names).",
    no_args_is_help=True,
)
app.add_typer(profile_app, name="profile")


def _fail(message: str) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(code=1)


def _store() -> ProfileStore:
    try:
        return ProfileStore()
    except ValueError as exc:
        raise _fail(str(exc)) from exc


@profile_app.command("create")
def profile_create(
    user: Annotated[str, typer.Argument(help="Pseudonymized id, e.g. P01.")],
    model: Annotated[str | None, typer.Option(help="Registry name of the model.")] = None,
    srate: Annotated[float | None, typer.Option(help="Sampling rate in Hz.")] = None,
    channels: Annotated[
        str | None, typer.Option(help="Comma-separated channel labels, e.g. C3,Cz,C4.")
    ] = None,
) -> None:
    """Create an empty profile."""
    from openmind_core.profiles import ProfileExistsError

    store = _store()
    labels = tuple(c.strip() for c in channels.split(",") if c.strip()) if channels else None
    try:
        profile = store.create(user, channels=labels, srate=srate, model=model)
    except (ValueError, ProfileExistsError) as exc:
        raise _fail(str(exc)) from exc
    typer.echo(f"created profile {profile.user}")


@profile_app.command("list")
def profile_list() -> None:
    """List profiles."""
    store = _store()
    profiles = store.list()
    if not profiles:
        typer.echo("no profiles")
        return
    typer.echo(f"{'user':<8}{'created':<27}{'updated':<27}{'model':<20}{'ref':<5}scores")
    for p in profiles:
        has_ref = "yes" if store.load_reference(p.user) is not None else "no"
        n_scores = len(store.scores(p.user))
        typer.echo(
            f"{p.user:<8}{p.created:<27}{p.updated:<27}{p.model or '-':<20}{has_ref:<5}{n_scores}"
        )


@profile_app.command("show")
def profile_show(user: Annotated[str, typer.Argument(help="User id, e.g. P01.")]) -> None:
    """Show the metadata of a profile."""
    from openmind_core.profiles import ProfileNotFoundError

    store = _store()
    try:
        profile = store.get(user)
        reference = store.load_reference(user)
        scores = store.scores(user)
    except (ValueError, ProfileNotFoundError) as exc:
        raise _fail(str(exc.args[0]) if exc.args else str(exc)) from exc
    typer.echo(f"user       {profile.user}")
    typer.echo(f"created    {profile.created}")
    typer.echo(f"updated    {profile.updated}")
    typer.echo(f"model      {profile.model or '-'}")
    typer.echo(f"srate      {profile.srate if profile.srate is not None else '-'}")
    typer.echo(f"channels   {','.join(profile.channels) if profile.channels else '-'}")
    if profile.notes:
        typer.echo(f"notes      {json.dumps(profile.notes, sort_keys=True)}")
    if reference is None:
        typer.echo("reference  none")
    else:
        typer.echo(f"reference  {','.join(reference.channels)}")
    typer.echo(f"scores     {len(scores)}")
    if scores:
        typer.echo(f"last score {json.dumps(scores[-1], sort_keys=True)}")


@profile_app.command("delete")
def profile_delete(
    user: Annotated[str, typer.Argument(help="User id, e.g. P01.")],
    yes: Annotated[bool, typer.Option("--yes", help="Do not ask for confirmation.")] = False,
) -> None:
    """Erase a profile and everything stored about the user."""
    from openmind_core.profiles import ProfileNotFoundError

    store = _store()
    try:
        if not store.exists(user):
            raise ProfileNotFoundError(f"no profile for user {user}")
        if not yes:
            typer.confirm(f"Erase all data of profile {user}? This cannot be undone", abort=True)
        count = store.delete(user)
    except (ValueError, ProfileNotFoundError) as exc:
        raise _fail(str(exc.args[0]) if exc.args else str(exc)) from exc
    typer.echo(f"erased profile {user} ({count} files removed)")


models_app = typer.Typer(
    help="Inspect the local model registry ($OPENMIND_DATA_DIR/models).",
    no_args_is_help=True,
)
app.add_typer(models_app, name="models")


def _size(n_bytes: int) -> str:
    """Human-readable size: B, KB, MB or GB."""
    size = float(n_bytes)
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


@models_app.command("list")
def models_list() -> None:
    """List registered models."""
    from openmind_core.models.registry import list_models, model_size

    try:
        cards = list_models()
    except ValueError as exc:
        raise _fail(str(exc)) from exc
    if not cards:
        typer.echo("no models")
        return
    typer.echo(
        f"{'name':<24}{'architecture':<14}{'classes':<9}{'alignment':<11}"
        f"{'channels':<10}{'size':<11}created"
    )
    for card in cards:
        size = _size(model_size(card.name))
        typer.echo(
            f"{card.name:<24}{card.architecture:<14}{len(card.classes):<9}"
            f"{card.alignment or '-':<11}{len(card.channels):<10}{size:<11}{card.created or '-'}"
        )


@models_app.command("show")
def models_show(name: Annotated[str, typer.Argument(help="Registry name of the model.")]) -> None:
    """Show the card of a registered model."""
    from openmind_core.models.registry import load_card, model_size, validate_name

    try:
        validate_name(name)
        card = load_card(name)
        size = model_size(name)
    except (ValueError, FileNotFoundError) as exc:
        raise _fail(str(exc)) from exc
    typer.echo(f"name          {card.name}")
    typer.echo(f"architecture  {card.architecture}")
    typer.echo(f"format        {card.format}")
    typer.echo(f"classes       {','.join(card.classes)}")
    typer.echo(f"alignment     {card.alignment or '-'}")
    typer.echo(f"channels      {','.join(card.channels)}")
    typer.echo(f"srate         {card.srate:g} Hz")
    typer.echo(f"n_times       {card.n_times} ({card.n_times / card.srate:g} s)")
    typer.echo(f"filters       {json.dumps(card.filters, sort_keys=True) if card.filters else '-'}")
    typer.echo(f"size          {_size(size)}")
    typer.echo(f"created       {card.created or '-'}")
    typer.echo(f"openmind-core {card.openmind_core}")
    if card.training:
        typer.echo(f"training      {json.dumps(card.training, sort_keys=True)}")


@models_app.command("delete")
def models_delete(
    name: Annotated[str, typer.Argument(help="Registry name of the model.")],
    yes: Annotated[bool, typer.Option("--yes", help="Do not ask for confirmation.")] = False,
) -> None:
    """Delete a registered model (its card and weights)."""
    from openmind_core.models.registry import delete_model, load_card, validate_name

    try:
        validate_name(name)
        load_card(name)
        if not yes:
            typer.confirm(f"Delete model {name}? This cannot be undone", abort=True)
        count = delete_model(name)
    except (ValueError, FileNotFoundError) as exc:
        raise _fail(str(exc)) from exc
    typer.echo(f"deleted model {name} ({count} files removed)")


@app.command("train")
def train(
    dataset: Annotated[str, typer.Option(help="Public dataset, e.g. bnci2014_001.")],
    subjects: Annotated[str, typer.Option(help="Subjects: ranges and commas, e.g. 1-9 or 1,3.")],
    model: Annotated[str, typer.Option(help="ts-lr, mdm, csp-lda, eegnet or shallow.")] = "ts-lr",
    classes: Annotated[str, typer.Option(help="Comma-separated commands.")] = "left,right,forward",
    channels: Annotated[
        str, typer.Option(help="Montage letter (A, C) or comma-separated channel labels.")
    ] = "C",
    alignment: Annotated[
        str, typer.Option(help="none, euclid or riemann (per session, like the benchmark).")
    ] = "none",
    name: Annotated[str | None, typer.Option(help="Registry name (default: derived).")] = None,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Replace an existing model.")
    ] = False,
) -> None:
    """Train a model on public data (research extra) and register it."""
    from openmind_core.training import (
        default_name,
        parse_subjects,
        resolve_channels,
        train_and_register,
    )

    try:
        subject_list = parse_subjects(subjects)
        labels, montage = resolve_channels(channels)
        class_list = [c.strip() for c in classes.split(",") if c.strip()]
        model_name = name or default_name(model, dataset, montage, alignment)
        path, training = train_and_register(
            dataset,
            subject_list,
            model,
            class_list,
            labels,
            alignment,
            model_name,
            overwrite=overwrite,
        )
    except ImportError as exc:
        raise _fail(
            f"{exc}. Install the research extra (datasets), plus deep for deep models: "
            "uv sync --extra research --extra deep"
        ) from exc
    except (ValueError, KeyError, FileExistsError) as exc:
        raise _fail(str(exc.args[0]) if exc.args else str(exc)) from exc
    typer.echo(f"trained {model} on {training['n_epochs']} epochs {training['counts']}")
    typer.echo(f"registered {model_name} in {path}")


def _user_id(value: str | None) -> str | None:
    from openmind_core.profiles import InvalidUserIdError, validate_user_id

    if value is None:
        return None
    try:
        return validate_user_id(value)
    except InvalidUserIdError as error:
        raise typer.BadParameter(str(error)) from error


def _broker(value: str) -> tuple[str, int]:
    from openmind_core.bus import parse_broker

    try:
        return parse_broker(value)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error


def _expected_labels(value: str | None) -> list[str] | None:
    """Labels of a montage letter (``C``) or of a comma-separated list."""
    if value is None:
        return None
    if value.upper() in {"A", "B", "C"}:
        from openmind_plug.montage import get_montage

        return [channel.label for channel in get_montage(value.upper()).channels()]
    labels = [label.strip() for label in value.split(",") if label.strip()]
    if not labels:
        raise typer.BadParameter("expected a montage letter (A, B, C) or comma-separated labels")
    return labels


@app.command("bandpower")
def bandpower(
    broker: Annotated[
        str, typer.Option(help="MQTT broker as host[:port].", show_default=True)
    ] = "localhost:1883",
    user: Annotated[
        str | None, typer.Option(help="Pseudonymized user id, e.g. P01.", callback=_user_id)
    ] = None,
    rate: Annotated[float, typer.Option(min=0.1, help="Maximum frames per second.")] = 10.0,
    baseline_s: Annotated[
        float,
        typer.Option(min=0.0, help="Seconds of rest recorded first as baseline (0 = none)."),
    ] = 30.0,
    expected: Annotated[
        str | None,
        typer.Option(
            help="Electrodes expected by the 3D brain: a montage letter (C) or comma-separated "
            "labels. Absent ones are sent with ok=false."
        ),
    ] = None,
    source_id: Annotated[
        str | None, typer.Option(help="LSL source_id of the EEG stream to read.")
    ] = None,
    timeout: Annotated[float, typer.Option(min=0.1, help="Seconds to wait for the stream.")] = 10.0,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print one line per frame; no broker needed.")
    ] = False,
) -> None:
    """Publish the band power per electrode (mu, beta, alpha) for the 3D brain."""
    try:
        from openmind_plug.inlet import EEGInlet
    except ImportError as error:
        typer.echo(
            "openmind-plug is required to read the EEG stream. Install it with "
            "`uv sync --extra plug` (or `pip install openmind-plug`).",
            err=True,
        )
        raise typer.Exit(1) from error

    from collections.abc import Callable, Iterator

    import numpy.typing as npt

    from openmind_core.bus import BrokerUnavailableError, Publisher
    from openmind_core.online.bandpower_service import format_frame, run_bandpower
    from openmind_protocol import BandPower

    host, port = _broker(broker)
    expected_labels = _expected_labels(expected)

    def print_frame(message: BandPower) -> None:
        typer.echo(format_frame(message))

    publisher: Publisher | None = None
    publish: Callable[[BandPower], None] = print_frame
    if not dry_run:
        publisher = Publisher(host, port)
        try:
            publisher.connect()
        except BrokerUnavailableError as error:
            typer.echo(f"Error: {error}", err=True)
            raise typer.Exit(1) from error
        publish = publisher.publish

    try:
        inlet = EEGInlet(timeout=timeout, source_id=source_id, with_markers=False)
    except TimeoutError as error:
        typer.echo(f"Error: no EEG stream found ({error}). Start `openmind-plug stream`.", err=True)
        if publisher is not None:
            publisher.close()
        raise typer.Exit(1) from error

    def blocks() -> Iterator[tuple[npt.ArrayLike, npt.ArrayLike]]:
        while True:
            samples, stamps = inlet.read(timeout=0.1)
            if len(stamps):
                yield samples, stamps

    typer.echo(
        f"Reading {inlet.source_id} ({inlet.info.srate:g} Hz, "
        f"{', '.join(inlet.info.labels)}); Ctrl+C to stop."
    )
    try:
        run_bandpower(
            blocks(),
            publish,
            inlet.info.labels,
            inlet.info.srate,
            user=user,
            rate_hz=rate,
            baseline_s=baseline_s,
            expected_labels=expected_labels,
            announce=typer.echo,
        )
    except KeyboardInterrupt:
        typer.echo("Stopped.")
    finally:
        inlet.close()
        if publisher is not None:
            publisher.close()


def _parse_thresholds(values: list[str]) -> dict[str, float]:
    """``["left=0.6", "forward=0.55"]`` to a dict, validated."""
    from openmind_core.online.evidence import MOVEMENTS

    out: dict[str, float] = {}
    for item in values:
        name, separator, number = item.partition("=")
        name = name.strip()
        if not separator or name not in MOVEMENTS:
            raise typer.BadParameter(
                f"expected <command>=<value> with command in {list(MOVEMENTS)}"
            )
        try:
            value = float(number)
        except ValueError:
            raise typer.BadParameter(f"{item!r}: {number!r} is not a number") from None
        if not 0.0 < value <= 1.0:
            raise typer.BadParameter(f"{item!r}: threshold must be in ]0, 1]")
        out[name] = value
    return out


@app.command("online")
def online(
    profile: Annotated[
        str, typer.Option(help="Pseudonymized profile id, e.g. P01.", callback=_user_id)
    ],
    model: Annotated[str, typer.Option(help="Registry name of the model (see `train`).")],
    broker: Annotated[
        str, typer.Option(help="MQTT broker as host[:port].", show_default=True)
    ] = "localhost:1883",
    source_id: Annotated[
        str | None, typer.Option(help="LSL source_id of the EEG stream to read.")
    ] = None,
    timeout: Annotated[float, typer.Option(min=0.1, help="Seconds to wait for the stream.")] = 10.0,
    warmup_s: Annotated[
        float,
        typer.Option(min=1.0, help="Seconds of clean signal to estimate the alignment reference."),
    ] = 30.0,
    threshold: Annotated[
        list[str] | None,
        typer.Option(help="Per-command threshold, repeatable: --threshold left=0.6."),
    ] = None,
    windows: Annotated[
        int, typer.Option(min=1, help="Predictions averaged per decision (4 is about 1 s).")
    ] = 4,
    save_reference: Annotated[
        bool,
        typer.Option(
            "--save-reference/--no-save-reference",
            help="Store the warm-up reference in the profile.",
        ),
    ] = True,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print the decisions; no broker needed.")
    ] = False,
    quiet: Annotated[bool, typer.Option("--quiet", help="No live line.")] = False,
) -> None:
    """Decode the EEG stream with a model and publish commands and status over MQTT."""
    try:
        from openmind_plug.inlet import EEGInlet
    except ImportError as error:
        typer.echo(
            "openmind-plug is required to read the EEG stream. Install it with "
            "`uv sync --extra plug` (or `pip install openmind-plug`).",
            err=True,
        )
        raise typer.Exit(1) from error

    import time
    from collections.abc import Callable, Iterator

    import numpy.typing as npt

    from openmind_core.bus import BrokerUnavailableError, Publisher
    from openmind_core.models.registry import list_models, load_model
    from openmind_core.online import (
        CommandEmitter,
        EvidenceAccumulator,
        EvidenceConfig,
        HealthMonitor,
        LatencyMeter,
        OnlineConfig,
        OnlineDecoder,
        StatusEmitter,
    )
    from openmind_core.online.decoder import FloatArray
    from openmind_core.online.service import run_online
    from openmind_core.profiles import ProfileNotFoundError
    from openmind_protocol import Command, Message, Status

    host, port = _broker(broker)
    thresholds = dict(EvidenceConfig().thresholds) | _parse_thresholds(threshold or [])
    store = _store()
    try:
        store.get(profile)
    except ProfileNotFoundError as error:
        raise _fail(
            f"no profile for user {profile}. Create it with `openmind-core profile create "
            f"{profile}`."
        ) from error
    try:
        net, card = load_model(model)
    except FileNotFoundError as error:
        known = ", ".join(c.name for c in list_models()) or "none"
        raise _fail(
            f"no model named {model!r} (registered: {known}). Train one with "
            "`openmind-core train --dataset bnci2014_001 --subjects 1-9 --channels C`."
        ) from error
    except (ImportError, ValueError) as error:
        raise _fail(f"cannot load model {model!r}: {error}") from error

    reference = None
    stored = store.load_reference(profile)
    if card.alignment is not None and stored is not None:
        if tuple(stored.channels) == tuple(card.channels):
            reference = stored
        else:
            typer.echo(
                f"note: the stored reference of {profile} has other channels "
                f"({','.join(stored.channels)}) than the model ({','.join(card.channels)}); "
                "a new one will be estimated.",
                err=True,
            )

    state: dict[str, Any] = {"command": "-", "confidence": 0.0, "printed": 0.0}

    def show(message: Message) -> None:
        if isinstance(message, Command):
            state["command"], state["confidence"] = message.command, message.confidence
            if dry_run:
                typer.echo(
                    f"{message.ts:.2f} {message.command:<8} {message.confidence:.2f} "
                    f"seq={message.seq}"
                )
        elif isinstance(message, Status) and not quiet and not dry_run:
            now = time.monotonic()
            if now - state["printed"] >= 0.25:
                state["printed"] = now
                text = f"{message.state:<11} {state['command']:<8} {state['confidence']:.2f}"
                if message.message:
                    text += f"  ({message.message})"
                typer.echo(f"\r{text:<70}", nl=False)

    publisher: Publisher | None = None
    publish: Callable[[Message], None] = show
    if not dry_run:
        publisher = Publisher(host, port)
        try:
            publisher.connect()
        except BrokerUnavailableError as error:
            typer.echo(f"Error: {error}", err=True)
            raise typer.Exit(1) from error
        sent = publisher.publish

        def publish(message: Message) -> None:
            sent(message)
            show(message)

    def cleanup(inlet: object | None = None) -> None:
        if inlet is not None:
            inlet.close()  # type: ignore[attr-defined]
        if publisher is not None:
            time.sleep(0.2)  # let the network thread send the last messages
            publisher.close()

    try:
        inlet = EEGInlet(timeout=timeout, source_id=source_id, with_markers=False)
    except TimeoutError as error:
        typer.echo(f"Error: no EEG stream found ({error}). Start `openmind-plug stream`.", err=True)
        cleanup()
        raise typer.Exit(1) from error

    try:
        decoder = OnlineDecoder(
            net,
            card,
            list(inlet.info.labels),
            inlet.info.srate,
            OnlineConfig(warmup_s=warmup_s),
            reference=reference,
        )
    except ValueError as error:
        cleanup(inlet)
        raise _fail(f"stream {inlet.source_id} does not fit model {model!r}: {error}") from error

    def keep_reference(covariance: FloatArray) -> None:
        if save_reference:
            store.save_reference(profile, covariance, card.channels)
            typer.echo(f"\nReference saved in profile {profile}.")

    try:
        store.update(profile, model=model)
        accumulator = EvidenceAccumulator(
            card.classes,
            EvidenceConfig(windows=windows, min_ok=min(2, windows), thresholds=thresholds),
        )
        typer.echo(
            f"Decoding {inlet.source_id} ({inlet.info.srate:g} Hz) with {model} for {profile}"
            f"{' (dry run)' if dry_run else f' -> {host}:{port}'}; Ctrl+C to stop."
        )
        if decoder.state == "calibrating":
            typer.echo(f"Warm-up: stay relaxed, eyes open, for {warmup_s:g} s...")

        def blocks() -> Iterator[tuple[npt.ArrayLike, npt.ArrayLike]]:
            while True:
                yield inlet.read(timeout=0.1)  # empty on timeout: keeps the status flowing

        stats = run_online(
            blocks(),
            publish,
            decoder=decoder,
            accumulator=accumulator,
            command_emitter=CommandEmitter(profile),
            status_emitter=StatusEmitter(model, profile),
            health=HealthMonitor(),
            latency=LatencyMeter(),
            on_reference=keep_reference if reference is None else None,
        )
        typer.echo(f"\nStopped after {stats.duration_s:.0f} s: {stats.commands}")
    except KeyboardInterrupt:
        typer.echo("\nStopped (rest published, status idle).")
    finally:
        cleanup(inlet)
