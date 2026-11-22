"""Tests of evidence accumulation."""

from __future__ import annotations

import pytest

from openmind_core.online import (
    Decision,
    DecoderStep,
    EvidenceAccumulator,
    EvidenceConfig,
    Prediction,
)

CLASSES = ("forward", "left", "right")


def pred(t: float, left: float = 0.0, right: float = 0.0, forward: float = 0.0) -> Prediction:
    return Prediction(t, {"forward": forward, "left": left, "right": right}, True)


def bad(t: float, reason: str = "muscle") -> Prediction:
    return Prediction(t, None, False, reason)


def run(
    acc: EvidenceAccumulator, preds: list[Prediction], actions: list[float] | None = None
) -> list[Decision]:
    return acc.update(DecoderStep(preds, actions or []))


def test_steady_left_gives_left_after_enough_windows() -> None:
    acc = EvidenceAccumulator(CLASSES)
    out = run(acc, [pred(i * 0.25, left=0.8, right=0.1, forward=0.1) for i in range(5)])
    assert out[0].command == "rest"  # a single ok window: below min_ok
    assert [d.command for d in out[1:]] == ["left"] * 4
    assert out[-1].confidence == pytest.approx(0.8)
    assert acc.current == "left"


def test_weak_noisy_probabilities_give_rest() -> None:
    acc = EvidenceAccumulator(CLASSES)
    out = run(acc, [pred(i * 0.25, left=0.4, right=0.3, forward=0.3) for i in range(6)])
    assert {d.command for d in out} == {"rest"}
    assert out[-1].confidence == pytest.approx(0.6)  # 1 - max(mean)


def test_hysteresis_prevents_flicker() -> None:
    steady = [pred(i * 0.25, left=0.8, right=0.1, forward=0.1) for i in range(4)]
    # 0.57 is under the threshold 0.6 but above 0.55
    wobble = [pred(1 + i * 0.25, left=0.57, right=0.23, forward=0.2) for i in range(8)]
    acc = EvidenceAccumulator(CLASSES)
    out = run(acc, steady + wobble)
    assert out[-1].command == "left"
    cold = EvidenceAccumulator(CLASSES)  # same wobble without a current command
    assert {d.command for d in run(cold, wobble)} == {"rest"}
    below = [pred(3 + i * 0.25, left=0.5, right=0.25, forward=0.25) for i in range(8)]
    assert run(acc, below)[-1].command == "rest"


def test_muscle_and_rejected_give_rest_with_full_confidence() -> None:
    acc = EvidenceAccumulator(CLASSES)
    out = run(acc, [bad(0.0), bad(0.25, "rejected"), bad(0.5, "warmup")])
    assert [(d.command, d.confidence, d.probabilities) for d in out] == [("rest", 1.0, None)] * 3


def test_enough_bad_windows_cancel_the_evidence() -> None:
    acc = EvidenceAccumulator(CLASSES)
    good = [pred(i * 0.25, left=0.9) for i in range(4)]
    assert run(acc, good)[-1].command == "left"
    out = run(acc, [bad(1.0), bad(1.25), bad(1.5)])
    assert out[0].command == "left"  # three ok windows are still enough
    assert out[1].command == "left"  # two ok windows: min_ok reached
    assert (out[-1].command, out[-1].confidence) == ("rest", 1.0)  # one ok window left


def test_actions_are_one_shot_and_in_time_order() -> None:
    acc = EvidenceAccumulator(CLASSES)
    preds = [pred(i * 0.25, left=0.9) for i in range(4)]
    out = run(acc, preds, actions=[0.6, 5.0])
    kinds = [(d.command, d.timestamp) for d in out]
    assert kinds == [
        ("rest", 0.0),
        ("left", 0.25),
        ("left", 0.5),
        ("action", 0.6),
        ("left", 0.75),
        ("action", 5.0),
    ]
    assert out[3].confidence == 0.95
    assert acc.current == "left"  # action does not change the movement


def test_forward_has_its_own_threshold() -> None:
    config = EvidenceConfig(thresholds={"left": 0.6, "right": 0.6, "forward": 0.45})
    acc = EvidenceAccumulator(CLASSES, config)
    out = run(acc, [pred(i * 0.25, left=0.3, right=0.2, forward=0.5) for i in range(4)])
    assert out[-1].command == "forward"
    strict = EvidenceAccumulator(CLASSES)
    assert (
        run(strict, [pred(i * 0.25, left=0.3, right=0.2, forward=0.5) for i in range(4)])[
            -1
        ].command
        == "rest"
    )


def test_class_without_threshold_maps_to_rest() -> None:
    acc = EvidenceAccumulator(("left", "rest", "right"))
    p = [Prediction(i * 0.25, {"left": 0.1, "rest": 0.8, "right": 0.1}, True) for i in range(4)]
    out = run(acc, p)
    assert out[-1].command == "rest" and out[-1].confidence == pytest.approx(0.8)


def test_reset_and_config_validation() -> None:
    acc = EvidenceAccumulator(CLASSES)
    run(acc, [pred(i * 0.25, left=0.9) for i in range(4)])
    acc.reset()
    assert acc.current == "rest"
    assert run(acc, [pred(0.0, left=0.9)])[0].command == "rest"  # history gone
    with pytest.raises(ValueError):
        EvidenceConfig(windows=2, min_ok=3)
