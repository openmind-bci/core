"""Command-line interface for ``openmind-core``.

Subcommands:

* ``info``: versions and data directory;
* ``profile create|list|show|delete``: local pseudonymized user profiles.

Others (``online``, ``bandpower``...) are added by later tasks.
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
