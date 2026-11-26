"""Tests of the user profile store and its CLI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from openmind_core.cli import app
from openmind_core.profiles import (
    InvalidUserIdError,
    ProfileExistsError,
    ProfileNotFoundError,
    ProfileStore,
    profiles_dir,
    validate_user_id,
)

CHANNELS = ("C3", "Cz", "C4")


@pytest.fixture
def store(tmp_path: Path) -> ProfileStore:
    root = tmp_path / "profiles"
    if any((p / ".git").exists() for p in (root, *root.parents)):
        pytest.skip("pytest tmp directory is inside a Git repository")
    return ProfileStore(root)


def _spd(n: int = 3) -> np.ndarray:
    a = np.random.default_rng(0).normal(size=(n, n))
    return a @ a.T + n * np.eye(n)


@pytest.mark.parametrize("user", ["P01", "P12", "P100"])
def test_valid_ids(user: str) -> None:
    assert validate_user_id(user) == user


@pytest.mark.parametrize("user", ["alice", "Thomas", "P1", "p01", "P01 ", "../P01", "P01/x", ""])
def test_names_refused(user: str) -> None:
    with pytest.raises(InvalidUserIdError, match="pseudonymized"):
        validate_user_id(user)


def test_every_method_validates_id(store: ProfileStore) -> None:
    for call in (
        lambda: store.create("alice"),
        lambda: store.get("alice"),
        lambda: store.exists("alice"),
        lambda: store.update("alice", model="m"),
        lambda: store.delete("alice"),
        lambda: store.load_reference("alice"),
        lambda: store.scores("alice"),
        lambda: store.append_score("alice", {}),
        lambda: store.weights_dir("alice"),
    ):
        with pytest.raises(InvalidUserIdError):
            call()


def test_round_trip(store: ProfileStore) -> None:
    created = store.create("P02", channels=["C3", "Cz"], srate=250.0, model="base")
    store.create("P01")
    assert store.exists("P02")
    assert store.get("P02") == created
    assert created.channels == ("C3", "Cz")
    assert [p.user for p in store.list()] == ["P01", "P02"]
    with pytest.raises(ProfileExistsError):
        store.create("P02")
    assert store.create("P02", exist_ok=True) == created
    updated = store.update("P02", model="personal", notes={"a": 1})
    assert updated.model == "personal"
    assert updated.updated >= created.updated
    assert updated.created == created.created
    assert store.get("P02").notes == {"a": 1}
    with pytest.raises(TypeError):
        store.update("P02", user="P09")
    with pytest.raises(ProfileNotFoundError):
        store.get("P09")


def test_list_empty_root(tmp_path: Path) -> None:
    assert ProfileStore(tmp_path / "missing").list() == []


def test_delete_removes_everything(store: ProfileStore) -> None:
    store.create("P01")
    store.save_reference("P01", _spd(), CHANNELS)
    store.append_score("P01", {"score": 1})
    (store.weights_dir("P01", create=True) / "w.pt").write_bytes(b"x")
    count = store.delete("P01")
    assert count == 4
    assert not (store.root / "P01").exists()
    assert not store.exists("P01")
    with pytest.raises(ProfileNotFoundError):
        store.delete("P01")


def test_reference_round_trip(store: ProfileStore) -> None:
    store.create("P01")
    assert store.load_reference("P01") is None
    cov = _spd()
    power = np.arange(6, dtype=float).reshape(3, 2)
    store.save_reference("P01", cov, CHANNELS, rest_power=power, rest_bands=["mu", "beta"])
    ref = store.load_reference("P01")
    assert ref is not None
    np.testing.assert_allclose(ref.covariance, cov)
    assert ref.channels == CHANNELS
    assert ref.rest_bands == ("mu", "beta")
    assert ref.rest_power is not None
    np.testing.assert_allclose(ref.rest_power, power)
    store.save_reference("P01", cov, CHANNELS)
    ref = store.load_reference("P01")
    assert ref is not None and ref.rest_power is None and ref.rest_bands is None


def test_invalid_reference_refused(store: ProfileStore) -> None:
    store.create("P01")
    bad = {
        "not square": (np.ones((3, 2)), CHANNELS),
        "channel mismatch": (_spd(3), ("C3", "Cz")),
        "asymmetric": (np.triu(_spd()) + 1.0, CHANNELS),
        "not positive": (-np.eye(3), CHANNELS),
        "nan": (np.full((3, 3), np.nan), CHANNELS),
    }
    for cov, names in bad.values():
        with pytest.raises(ValueError):
            store.save_reference("P01", cov, names)
    with pytest.raises(ValueError):
        store.save_reference("P01", _spd(), CHANNELS, rest_power=np.ones((3, 2)))
    with pytest.raises(ValueError):
        store.save_reference(
            "P01", _spd(), CHANNELS, rest_power=np.ones((2, 2)), rest_bands=["mu", "beta"]
        )
    assert store.load_reference("P01") is None


def test_scores(store: ProfileStore) -> None:
    store.create("P01")
    assert store.scores("P01") == []
    store.append_score("P01", {"score": 10, "ts": 1.0})
    store.append_score("P01", {"score": 20})
    scores = store.scores("P01")
    assert [s["score"] for s in scores] == [10, 20]
    assert scores[0]["ts"] == 1.0
    assert "ts" in scores[1]
    lines = (store.root / "P01" / "scores.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_refuses_root_inside_git_repo(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    with pytest.raises(ValueError, match="Git"):
        ProfileStore(repo / "data" / "profiles")


def test_atomic_write_leaves_no_tmp(store: ProfileStore) -> None:
    store.create("P01")
    store.update("P01", model="m")
    store.save_reference("P01", _spd(), CHANNELS)
    names = sorted(p.name for p in (store.root / "P01").iterdir())
    assert names == ["profile.json", "reference.npz"]
    assert (
        json.loads((store.root / "P01" / "profile.json").read_text(encoding="utf-8"))["model"]
        == "m"
    )


def test_profiles_dir_uses_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENMIND_DATA_DIR", str(tmp_path))
    assert profiles_dir() == tmp_path / "profiles"


@pytest.fixture
def cli_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    if any((p / ".git").exists() for p in (tmp_path, *tmp_path.parents)):
        pytest.skip("pytest tmp directory is inside a Git repository")
    monkeypatch.setenv("OPENMIND_DATA_DIR", str(tmp_path))
    return tmp_path


def test_cli_flow(cli_env: Path) -> None:
    runner = CliRunner()
    assert "no profiles" in runner.invoke(app, ["profile", "list"]).output
    result = runner.invoke(
        app,
        ["profile", "create", "P01", "--model", "base", "--srate", "250", "--channels", "C3,Cz"],
    )
    assert result.exit_code == 0, result.output
    store = ProfileStore()
    store.save_reference("P01", _spd(2), ("C3", "Cz"))
    store.append_score("P01", {"score": 42})
    listing = runner.invoke(app, ["profile", "list"]).output
    assert "P01" in listing and "base" in listing and "yes" in listing
    shown = runner.invoke(app, ["profile", "show", "P01"]).output
    assert "C3,Cz" in shown and '"score": 42' in shown
    # Declining the confirmation keeps the profile.
    declined = runner.invoke(app, ["profile", "delete", "P01"], input="n\n")
    assert declined.exit_code != 0
    assert store.exists("P01")
    result = runner.invoke(app, ["profile", "delete", "P01", "--yes"])
    assert result.exit_code == 0
    assert "3 files removed" in result.output
    assert not store.exists("P01")


def test_cli_errors(cli_env: Path) -> None:
    runner = CliRunner()
    for args in (
        ["profile", "create", "alice"],
        ["profile", "show", "P09"],
        ["profile", "delete", "P09", "--yes"],
    ):
        result = runner.invoke(app, args)
        assert result.exit_code == 1
        assert "error:" in result.output
        assert "Traceback" not in result.output
    runner.invoke(app, ["profile", "create", "P01"])
    assert runner.invoke(app, ["profile", "create", "P01"]).exit_code == 1


def test_cli_refuses_git_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    monkeypatch.setenv("OPENMIND_DATA_DIR", str(tmp_path))
    result = CliRunner().invoke(app, ["profile", "list"])
    assert result.exit_code == 1
    assert "Git" in result.output
