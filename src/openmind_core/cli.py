"""Command-line interface for ``openmind-core``.

Subcommands (``online``, ``profile``...) are added by later tasks.
"""

import re
from typing import Annotated

import typer

from openmind_core import __version__

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


_USER_ID = re.compile(r"^P[0-9]{2,}$")


def _user_id(value: str | None) -> str | None:
    if value is not None and not _USER_ID.fullmatch(value):
        raise typer.BadParameter("expected a pseudonym such as P01 (letter P and 2+ digits)")
    return value


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
