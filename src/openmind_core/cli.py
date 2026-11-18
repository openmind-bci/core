"""Command-line interface for ``openmind-core``.

Subcommands (``online``, ``profile``, ``bandpower``...) are added by later tasks.
"""

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
