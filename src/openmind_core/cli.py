"""Command-line interface for ``openmind-core``.

Subcommands:

* ``info``: versions and data directory;
* ``profile create|list|show|delete``: local pseudonymized user profiles;
* ``train``: train a model on a public dataset and register it;
* ``bandpower``: publish the band power per electrode for the 3D brain.

``online`` is added by a later task.
"""

import json
from typing import Annotated

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
