"""Tests for the calibration trials rebuilt from markers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from openmind_core.io import Trial, TrialCollector


@dataclass
class Marker:
    """Same shape as ``openmind_plug.sources.Marker``."""

    timestamp: float
    payload: dict[str, Any]


def sandbox(kind: str, trial: int | None = None, label: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"v": 1, "type": kind, "source": "sandbox", "scene": "bird"}
    if trial is not None:
        payload["trial"] = trial
    if label is not None:
        payload["label"] = label
    return payload


def session(*trials: tuple[int, str], t0: float = 100.0) -> list[Marker]:
    markers = [Marker(t0, sandbox("event", label="session_start"))]
    t = t0 + 1
    for number, label in trials:
        markers += [
            Marker(t, sandbox("trial_start", number)),
            Marker(t + 2, sandbox("cue", number, label)),
            Marker(t + 6, sandbox("trial_end", number)),
        ]
        t += 8
    return markers


def test_complete_trials_are_rebuilt() -> None:
    collector = TrialCollector()
    trials = collector.add(session((1, "left"), (2, "right")))
    assert trials == [
        Trial(1, "left", cue_time=103.0, start=101.0, end=107.0, scene="bird"),
        Trial(2, "right", cue_time=111.0, start=109.0, end=115.0, scene="bird"),
    ]
    assert collector.trials == trials
    assert collector.interrupted == 0
    assert collector.events == [(100.0, "session_start")]


def test_markers_of_other_producers_are_ignored() -> None:
    markers = session((1, "left"))
    # Cues of a dataset replayed by Plug, merged in the same flow.
    markers += [Marker(102.5, {"type": "cue", "label": "right", "trial": 1, "source": "replay"})]
    markers += [Marker(104.0, {"type": "state", "label": "rest", "source": "synthetic"})]
    assert [t.label for t in TrialCollector().add(markers)] == ["left"]
    assert len(TrialCollector(source=None).add(session((1, "left")))) == 1


def test_interrupted_trials_are_discarded() -> None:
    markers = session((1, "left"), (2, "right"), (3, "forward"))
    markers = [
        m for m in markers if not (m.payload["type"] == "trial_end" and m.payload["trial"] == 2)
    ]
    collector = TrialCollector()
    assert [t.trial for t in collector.add(markers)] == [1, 3]
    assert collector.interrupted == 1


def test_trial_without_cue_is_interrupted() -> None:
    markers = [
        m
        for m in session((1, "left"), (2, "right"))
        if m.payload["type"] != "cue" or m.payload["trial"] != 1
    ]
    collector = TrialCollector()
    assert [t.trial for t in collector.add(markers)] == [2]
    assert collector.interrupted == 1


def test_session_end_interrupts_an_open_trial() -> None:
    markers = session((1, "left"))[:-1]  # no trial_end
    markers.append(Marker(200.0, sandbox("event", label="session_end")))
    collector = TrialCollector()
    assert collector.add(markers) == []
    assert collector.interrupted == 1
    assert collector.events[-1] == (200.0, "session_end")


def test_markers_may_arrive_in_several_batches_and_out_of_order() -> None:
    markers = session((1, "left"), (2, "right"))
    collector = TrialCollector()
    first = collector.add(reversed(markers[:4]))  # sorted by timestamp before use
    second = collector.add([(m.timestamp, m.payload) for m in markers[4:]])
    assert [t.trial for t in first + second] == [1, 2]


def test_cue_of_another_trial_is_ignored() -> None:
    markers = session((1, "left"))
    markers.insert(2, Marker(101.5, sandbox("cue", 7, "right")))
    (trial,) = TrialCollector().add(markers)
    assert trial.label == "left"


@pytest.mark.parametrize("payload", [{"type": "trial_start"}, {"type": "cue"}, {"no": "type"}])
def test_malformed_markers_are_ignored(payload: dict[str, Any]) -> None:
    collector = TrialCollector(source=None)
    assert collector.add([Marker(1.0, payload)]) == []
    assert collector.interrupted == 0


def test_calibration_markers_from_sandbox_reach_core_through_lsl() -> None:
    """Plug EEG stream + a Sandbox markers stream started later → complete trials."""
    pytest.importorskip("openmind_plug")
    import time
    import uuid

    import pylsl
    from openmind_plug.inlet import EEGInlet
    from openmind_plug.montage import get_montage
    from openmind_plug.outlet import EEGOutlet
    from openmind_plug.sources import StreamInfo

    if not hasattr(EEGInlet, "marker_sources"):
        pytest.skip("this version of openmind-plug reads one markers stream only")
    info = StreamInfo(
        srate=250.0,
        channels=get_montage("C").channels(),
        manufacturer="OpenMind",
        model="test",
        driver="test",
        reference="unknown",
        montage="C",
    )
    eeg_id = f"openmind-core-test:{uuid.uuid4()}"
    sandbox_id = f"openmind-sandbox:{uuid.uuid4()}:markers"
    with (
        EEGOutlet(info, source_id=eeg_id),
        EEGInlet(timeout=5, source_id=eeg_id, markers_from=[sandbox_id], rescan_s=0.2) as inlet,
    ):
        header = pylsl.StreamInfo(
            "openmind-markers", "Markers", 1, pylsl.IRREGULAR_RATE, pylsl.cf_string, sandbox_id
        )
        header.desc().append_child_value("spec_version", "1.2.0")
        bridge = pylsl.StreamOutlet(header)
        deadline = time.monotonic() + 5
        while sandbox_id not in inlet.marker_sources and time.monotonic() < deadline:
            inlet.read_markers(timeout=0.2)
        time.sleep(0.3)
        now = pylsl.local_clock()
        for offset, kind, label in (
            (0.0, "trial_start", None),
            (0.5, "cue", "left"),
            (1.0, "trial_end", None),
        ):
            payload = sandbox(kind, 1, label)
            bridge.push_sample([json.dumps(payload)], now + offset)
        collector = TrialCollector()
        deadline = time.monotonic() + 5
        while not collector.trials and time.monotonic() < deadline:
            collector.add(inlet.read_markers(timeout=0.2))
    (trial,) = collector.trials
    assert trial.label == "left"
    assert trial.cue_time == pytest.approx(now + 0.5, abs=1e-3)
