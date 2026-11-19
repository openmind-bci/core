"""Tests for epoch extraction from continuous runs, and the epoch container."""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pytest

from openmind_core.datasets import MONTAGE_A_MI, EpochSet, epochs_from_raw, load_epochs
from openmind_core.datasets.moabb import _spec

CHANNELS = ["Fz", "Fc3.", "FC4", "C3", "CZ", "C4", "CP3", "CP4", "EOG1"]


def make_raw(sfreq: float = 160.0, seconds: float = 60.0, line_freq: float = 60.0) -> mne.io.Raw:
    rng = np.random.default_rng(0)
    n = int(seconds * sfreq)
    t = np.arange(n) / sfreq
    data = rng.normal(0, 5e-6, (len(CHANNELS), n))
    data[3] += 10e-6 * np.sin(2 * np.pi * 10 * t)  # mu on C3
    types = ["eog" if name.startswith("EOG") else "eeg" for name in CHANNELS]
    info = mne.create_info(CHANNELS, sfreq, types)
    info["line_freq"] = line_freq
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_annotations(
        mne.Annotations(
            onset=[5.0, 12.0, 19.0, 26.0, 33.0, 40.0, 59.0],
            duration=[3.0] * 7,
            description=[
                "left_hand",
                "right_hand",
                "feet",
                "tongue",
                "rest",
                "left_hand",
                "right_hand",
            ],
        )
    )
    return raw


def test_epochs_have_montage_channels_rate_and_labels() -> None:
    epochs = epochs_from_raw(make_raw(), 3, "0", "test", classes=("left", "right", "forward"))
    assert epochs.channels == MONTAGE_A_MI
    assert epochs.srate == 250.0
    assert epochs.X.shape == (4, 7, 500)  # tongue and rest dropped, last cue too late
    assert epochs.X.dtype == np.float32
    assert epochs.y.tolist() == ["left", "right", "forward", "left"]
    assert epochs.subject.tolist() == [3] * 4
    assert epochs.session.tolist() == ["0"] * 4
    assert epochs.info["filters"]["notch_hz"] == 60.0  # dataset power line
    assert epochs.info["rejected"] == 0


def test_epochs_are_in_microvolts_and_filtered() -> None:
    epochs = epochs_from_raw(make_raw(), 1, "0", "test", classes=("left",))
    c3 = epochs.channels.index("C3")
    # The 10 Hz rhythm (10 µV) passes the 8-30 Hz band-pass; units are µV, not V.
    assert 4 < np.abs(epochs.X[:, c3]).max() < 40
    assert np.abs(epochs.X.mean(axis=2)).max() < 2  # no DC left


def test_cue_offset_shifts_epochs() -> None:
    base = epochs_from_raw(make_raw(), 1, "0", "test", classes=("left",), reject_uv=None)
    shifted = epochs_from_raw(
        make_raw(), 1, "0", "test", classes=("left",), reject_uv=None, cue_offset_s=1.0
    )
    np.testing.assert_allclose(shifted.X[0, :, :250], base.X[0, :, 250:], atol=1e-4)


def test_stim_channel_events_take_precedence_over_annotations() -> None:
    """Like MOABB's BNCI2014-001: trial starts on the stim channel, cues 2 s later."""
    raw = make_raw()
    stim = np.zeros((1, raw.n_times))
    for onset, code in [(3.0, 1), (10.0, 2)]:  # trial starts, 2 s before the annotated cues
        stim[0, int(onset * raw.info["sfreq"])] = code
    info = mne.create_info(["STI"], raw.info["sfreq"], "stim")
    info["line_freq"] = raw.info["line_freq"]
    raw.add_channels([mne.io.RawArray(stim, info, verbose=False)])
    kwargs = {"classes": ("left", "right"), "reject_uv": None}
    from_stim = epochs_from_raw(
        raw, 1, "0", "test", cue_offset_s=2.0, event_id={"left_hand": 1, "right_hand": 2}, **kwargs
    )
    from_annotations = epochs_from_raw(raw, 1, "0", "test", **kwargs)
    assert from_stim.y.tolist() == ["left", "right"]
    np.testing.assert_allclose(from_stim.X, from_annotations.X[:2], atol=1e-4)


def test_rejection_by_amplitude() -> None:
    raw = make_raw()
    raw._data[4, int(6.5 * 160)] += 500e-6  # spike on Cz in the first epoch
    epochs = epochs_from_raw(raw, 1, "0", "test", classes=("left", "right", "forward"))
    assert epochs.info["rejected"] == 1
    assert epochs.y.tolist() == ["right", "forward", "left"]


def test_missing_channel_is_reported() -> None:
    raw = make_raw().drop_channels(["C4"])
    with pytest.raises(ValueError, match=r"lacks channels \['C4'\]"):
        epochs_from_raw(raw, 1, "0", "test", classes=("left",))


def test_unknown_dataset() -> None:
    with pytest.raises(KeyError, match="supported"):
        _spec("nope")
    assert _spec("BNCI2014-001").key == "bnci2014_001"


# --- EpochSet -----------------------------------------------------------------------


def small_set(subject: int, n: int = 4) -> EpochSet:
    return EpochSet(
        X=np.ones((n, 2, 10), dtype=np.float32) * subject,
        y=np.array(["left", "right"] * (n // 2)),
        subject=np.full(n, subject),
        session=np.array(["0train"] * n),
        channels=("C3", "C4"),
        srate=250.0,
        dataset="test",
    )


def test_epoch_set_helpers(tmp_path: Path) -> None:
    joined = EpochSet.concatenate([small_set(1), small_set(2)])
    assert len(joined) == 8
    assert joined.subjects == (1, 2)
    assert joined.classes == ("left", "right")
    assert joined.counts() == {"left": 4, "right": 4}
    left = joined.select(joined.y == "left")
    assert len(left) == 4 and left.classes == ("left",)

    path = joined.save(tmp_path / "cache" / "epochs.npz")
    loaded = EpochSet.load(path)
    np.testing.assert_array_equal(loaded.X, joined.X)
    assert loaded.y.tolist() == joined.y.tolist()
    assert loaded.session.tolist() == ["0train"] * 8  # names are not truncated
    assert (loaded.channels, loaded.srate, loaded.dataset) == (("C3", "C4"), 250.0, "test")


def test_epoch_set_validation() -> None:
    with pytest.raises(ValueError, match="n_epochs"):
        EpochSet(
            np.zeros((2, 3, 4), np.float32),
            np.array(["a", "b"]),
            np.zeros(2, int),
            np.array(["s", "s"]),
            ("C3", "C4"),
            250.0,
            "x",
        )
    with pytest.raises(ValueError, match="one entry per epoch"):
        EpochSet(
            np.zeros((2, 2, 4), np.float32),
            np.array(["a"]),
            np.zeros(2, int),
            np.array(["s", "s"]),
            ("C3", "C4"),
            250.0,
            "x",
        )
    with pytest.raises(ValueError, match="share"):
        other = small_set(1)
        other.srate = 160.0
        EpochSet.concatenate([small_set(1), other])


@pytest.mark.slow
def test_bnci2014_001_subject_1() -> None:
    pytest.importorskip("moabb")
    epochs = load_epochs("bnci2014_001", [1])
    assert epochs.X.shape == (432, 7, 500)  # 2 sessions x 3 classes x 72 trials
    assert epochs.counts() == {"forward": 144, "left": 144, "right": 144}
    assert sorted(set(epochs.session.tolist())) == ["0train", "1test"]
    assert 5 < float(np.median(np.ptp(epochs.X, axis=2))) < 100
