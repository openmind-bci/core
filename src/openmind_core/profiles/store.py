"""On-disk store of user profiles.

A profile holds what is needed to resume a session without a long
calibration: metadata, a reference covariance (and optional resting band
power), a score history and the weights of a personal model.

EEG data are health data (GDPR). Therefore:

* users are identified by pseudonymized ids only (``P01``, ``P02``...);
* profiles live under ``$OPENMIND_DATA_DIR/profiles``, never in a Git
  repository (the store refuses such a root);
* :meth:`ProfileStore.delete` erases everything about a user (right to
  erasure).

Layout of ``<root>/<user>/``::

    profile.json     metadata (JSON)
    reference.npz    covariance, channels, optional rest_bands / rest_power
    scores.jsonl     one JSON object per line
    weights/         personal model weights

Encryption at rest is not implemented; it could be added later as an
optional feature.
"""

from __future__ import annotations

import builtins
import contextlib
import json
import os
import re
import shutil
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from openmind_core.paths import data_dir

#: Same pattern as ``openmind_protocol.messages.UserId``.
USER_ID_PATTERN = re.compile(r"^P[0-9]{2,}$")

_PROFILE_FILE = "profile.json"
_REFERENCE_FILE = "reference.npz"
_SCORES_FILE = "scores.jsonl"
_WEIGHTS_DIR = "weights"
_SYMMETRY_TOL = 1e-8


class InvalidUserIdError(ValueError):
    """The user id is not a pseudonymized id."""


class ProfileExistsError(FileExistsError):
    """A profile already exists for this user."""


class ProfileNotFoundError(KeyError):
    """No profile exists for this user."""


def validate_user_id(user: str) -> str:
    """Return ``user`` if it is a pseudonymized id, else raise."""
    if not isinstance(user, str) or not USER_ID_PATTERN.fullmatch(user):
        raise InvalidUserIdError(
            "invalid user id: use a pseudonymized id such as P01, P02... "
            "(the letter P followed by at least two digits), never a name"
        )
    return user


def profiles_dir() -> Path:
    """Default profile directory, ``<data dir>/profiles``."""
    return data_dir() / "profiles"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Profile:
    """Metadata of a user profile."""

    user: str
    created: str
    updated: str
    channels: tuple[str, ...] | None = None
    srate: float | None = None
    model: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        data = {
            "user": self.user,
            "created": self.created,
            "updated": self.updated,
            "channels": list(self.channels) if self.channels is not None else None,
            "srate": self.srate,
            "model": self.model,
            "notes": self.notes,
        }
        return json.dumps(data, indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, text: str) -> Profile:
        """Parse a JSON string produced by :meth:`to_json`."""
        data = json.loads(text)
        channels = data.get("channels")
        srate = data.get("srate")
        return cls(
            user=validate_user_id(data["user"]),
            created=data["created"],
            updated=data["updated"],
            channels=tuple(channels) if channels is not None else None,
            srate=float(srate) if srate is not None else None,
            model=data.get("model"),
            notes=dict(data.get("notes") or {}),
        )


@dataclass(frozen=True)
class Reference:
    """Reference covariance of a user, with optional resting band power."""

    covariance: npt.NDArray[np.float64]
    channels: tuple[str, ...]
    rest_power: npt.NDArray[np.float64] | None = None
    rest_bands: tuple[str, ...] | None = None


def _in_git_tree(path: Path) -> bool:
    resolved = path.expanduser().resolve()
    return any((p / ".git").exists() for p in (resolved, *resolved.parents))


def _atomic_write_text(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _chmod_private(path: Path) -> None:
    with contextlib.suppress(OSError, NotImplementedError):
        path.chmod(0o700)


class ProfileStore:
    """Read and write user profiles under a root directory."""

    def __init__(self, root: Path | None = None) -> None:
        resolved = Path(root) if root is not None else profiles_dir()
        if _in_git_tree(resolved):
            raise ValueError(
                f"refusing profile root {resolved}: it is inside a Git working tree; "
                "profiles are health data and must never be stored in a repository "
                "(set OPENMIND_DATA_DIR to a directory outside any repository)"
            )
        self.root = resolved

    # -- paths -----------------------------------------------------------

    def _user_dir(self, user: str) -> Path:
        return self.root / validate_user_id(user)

    def _require(self, user: str) -> Path:
        path = self._user_dir(user)
        if not (path / _PROFILE_FILE).is_file():
            raise ProfileNotFoundError(f"no profile for user {user}")
        return path

    def weights_dir(self, user: str, create: bool = False) -> Path:
        """Directory of the personal model weights of ``user``."""
        path = self._user_dir(user) / _WEIGHTS_DIR
        if create:
            path.mkdir(parents=True, exist_ok=True)
            _chmod_private(path)
        return path

    # -- profile metadata ------------------------------------------------

    def exists(self, user: str) -> bool:
        """Whether a profile exists for ``user``."""
        return (self._user_dir(user) / _PROFILE_FILE).is_file()

    def _write(self, profile: Profile) -> None:
        path = self._user_dir(profile.user)
        path.mkdir(parents=True, exist_ok=True)
        _chmod_private(self.root)
        _chmod_private(path)
        _atomic_write_text(path / _PROFILE_FILE, profile.to_json())

    def create(
        self,
        user: str,
        *,
        channels: tuple[str, ...] | builtins.list[str] | None = None,
        srate: float | None = None,
        model: str | None = None,
        exist_ok: bool = False,
    ) -> Profile:
        """Create a profile (or return the existing one when ``exist_ok``)."""
        validate_user_id(user)
        if self.exists(user):
            if exist_ok:
                return self.get(user)
            raise ProfileExistsError(f"a profile already exists for user {user}")
        now = _now()
        profile = Profile(
            user=user,
            created=now,
            updated=now,
            channels=tuple(channels) if channels is not None else None,
            srate=srate,
            model=model,
        )
        self._write(profile)
        return profile

    def get(self, user: str) -> Profile:
        """Return the profile of ``user``."""
        path = self._require(user)
        return Profile.from_json((path / _PROFILE_FILE).read_text(encoding="utf-8"))

    def list(self) -> list[Profile]:
        """Return all profiles sorted by user id."""
        if not self.root.is_dir():
            return []
        users = sorted(
            p.name
            for p in self.root.iterdir()
            if USER_ID_PATTERN.fullmatch(p.name) and (p / _PROFILE_FILE).is_file()
        )
        return [self.get(user) for user in users]

    def update(self, user: str, **fields: Any) -> Profile:
        """Update profile fields and bump ``updated``."""
        profile = self.get(user)
        allowed = {"channels", "srate", "model", "notes"}
        unknown = set(fields) - allowed
        if unknown:
            raise TypeError(f"cannot update fields: {sorted(unknown)}")
        if fields.get("channels") is not None:
            fields["channels"] = tuple(fields["channels"])
        # Keep strictly increasing timestamps even within the same second.
        updated = _now()
        if updated <= profile.updated:
            updated = profile.updated
        new = replace(profile, updated=updated, **fields)
        self._write(new)
        return new

    def delete(self, user: str) -> int:
        """Erase everything about ``user`` and return the number of files removed."""
        path = self._require(user)
        count = sum(1 for p in path.rglob("*") if p.is_file())
        shutil.rmtree(path)
        return count

    # -- reference ---------------------------------------------------------

    def save_reference(
        self,
        user: str,
        covariance: npt.ArrayLike,
        channels: tuple[str, ...] | builtins.list[str],
        rest_power: npt.ArrayLike | None = None,
        rest_bands: tuple[str, ...] | builtins.list[str] | None = None,
    ) -> None:
        """Store the reference covariance (and optional resting band power)."""
        path = self._require(user)
        cov = np.asarray(covariance, dtype=np.float64)
        names = tuple(channels)
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ValueError(f"covariance must be square, got shape {cov.shape}")
        if cov.shape[0] != len(names):
            raise ValueError(
                f"covariance is {cov.shape[0]}x{cov.shape[0]} but {len(names)} channels given"
            )
        if not np.all(np.isfinite(cov)):
            raise ValueError("covariance contains non-finite values")
        scale = max(float(np.abs(cov).max()), 1.0)
        if not np.allclose(cov, cov.T, atol=_SYMMETRY_TOL * scale):
            raise ValueError("covariance must be symmetric")
        cov = (cov + cov.T) / 2
        if float(np.linalg.eigvalsh(cov)[0]) <= 0:
            raise ValueError("covariance must be positive definite")
        arrays: dict[str, Any] = {"covariance": cov, "channels": np.array(names, dtype=str)}
        if rest_power is not None:
            power = np.asarray(rest_power, dtype=np.float64)
            if rest_bands is None:
                raise ValueError("rest_bands is required with rest_power")
            if power.shape != (len(names), len(rest_bands)):
                raise ValueError(
                    f"rest_power must have shape (n_channels, n_bands) = "
                    f"({len(names)}, {len(rest_bands)}), got {power.shape}"
                )
            arrays["rest_power"] = power
            arrays["rest_bands"] = np.array(tuple(rest_bands), dtype=str)
        fd, tmp = tempfile.mkstemp(dir=path, prefix=f".{_REFERENCE_FILE}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as handle:
                np.savez(handle, **arrays)
            os.replace(tmp, path / _REFERENCE_FILE)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    def load_reference(self, user: str) -> Reference | None:
        """Return the stored reference, or ``None`` if there is none."""
        path = self._require(user) / _REFERENCE_FILE
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as data:
            power = data["rest_power"].astype(np.float64) if "rest_power" in data else None
            bands = tuple(str(b) for b in data["rest_bands"]) if "rest_bands" in data else None
            return Reference(
                covariance=data["covariance"].astype(np.float64),
                channels=tuple(str(c) for c in data["channels"]),
                rest_power=power,
                rest_bands=bands,
            )

    # -- scores ------------------------------------------------------------

    def append_score(self, user: str, score: Mapping[str, Any]) -> None:
        """Append one score record (``ts`` is added if missing)."""
        path = self._require(user)
        record = dict(score)
        record.setdefault("ts", time.time())
        line = json.dumps(record, sort_keys=True) + "\n"
        with (path / _SCORES_FILE).open("a", encoding="utf-8") as handle:
            handle.write(line)

    def scores(self, user: str) -> builtins.list[dict[str, Any]]:
        """Return all score records, oldest first."""
        path = self._require(user) / _SCORES_FILE
        if not path.is_file():
            return []
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
