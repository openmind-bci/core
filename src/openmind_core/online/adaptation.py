"""Online adaptation with safeguards: slow re-centering, pseudo-labels, rollback.

EEG drifts within a session (electrode impedance, fatigue, posture). Three
mechanisms follow it, each bounded and reversible:

* :class:`RecenteringAdapter` (unsupervised, on by default when used): the
  alignment reference covariance moves slowly towards the covariance of each
  decoded window, along the geodesic of the Riemannian manifold (``riemann``)
  or as a moving average (``euclid``), and the whitening is rebuilt. Each step
  moves the reference by at most ``max_step`` (affine-invariant distance) and
  the total drift from the initial reference is capped at ``max_drift``:
  beyond, the adapter freezes and reports it.
* :class:`PseudoLabelTuner` (off by default): windows decoded with a smoothed
  confidence above ``threshold`` join the training set of the personal model
  with their predicted label, as a minority (at most ``max_share`` of the set).
  Risk: confirmation bias, since the model learns its own mistakes.
* :class:`AdaptationGuard`: never touches the initial model nor the initial
  reference, snapshots the adaptive state before each block and restores it
  when the progress score falls by more than ``max_drop`` below the best
  recent block.

None of this is wired into the command line yet: a session builds the pieces
and calls ``guard.begin_block()`` / ``guard.end_block(score)`` itself.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from openmind_core.alignment import covariances
from openmind_core.online.decoder import OnlineDecoder, Prediction
from openmind_core.preprocessing.windows import STEP_S

if TYPE_CHECKING:
    from openmind_core.calibration.adaptive import PersonalModel, PersonalState

__all__ = [
    "AdaptationGuard",
    "PseudoLabelTuner",
    "RecenteringAdapter",
    "geodesic_step",
    "riemann_distance",
]

FloatArray = npt.NDArray[np.float64]

logger = logging.getLogger(__name__)

_EPS = 1e-12


def _sym(matrix: FloatArray) -> FloatArray:
    return np.asarray((matrix + matrix.T) / 2.0)


def _powers(matrix: FloatArray) -> tuple[FloatArray, FloatArray]:
    """``(matrix^(1/2), matrix^(-1/2))`` of a symmetric positive definite matrix."""
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, _EPS)
    root = np.sqrt(values)
    return np.asarray((vectors * root) @ vectors.T), np.asarray((vectors / root) @ vectors.T)


def riemann_distance(a: FloatArray, b: FloatArray) -> float:
    """Affine-invariant distance between two symmetric positive definite matrices."""
    _, inv_root = _powers(a)
    values = np.linalg.eigvalsh(_sym(inv_root @ b @ inv_root))
    return float(np.sqrt(np.sum(np.log(np.maximum(values, _EPS)) ** 2)))


def geodesic_step(a: FloatArray, b: FloatArray, t: float) -> FloatArray:
    """Point at fraction ``t`` of the geodesic from ``a`` (``t=0``) to ``b`` (``t=1``)."""
    root, inv_root = _powers(a)
    values, vectors = np.linalg.eigh(_sym(inv_root @ b @ inv_root))
    values = np.maximum(values, _EPS)
    return _sym(root @ ((vectors * values**t) @ vectors.T) @ root)


@dataclass(frozen=True)
class _AdapterState:
    reference: FloatArray | None
    seen: int
    updates: int
    frozen: bool


class RecenteringAdapter:
    """Unsupervised, bounded re-centering of the alignment reference.

    Args:
        method: ``"riemann"`` or ``"euclid"``; ``None`` takes the decoder's.
        rate: Weight of each window in the reference (``alpha``).
        half_life_s: Alternative to ``rate``: after this many seconds a past
            window weighs half (``alpha = 1 - 0.5 ** (step_s / half_life_s)``).
        step_s: Time between two windows (for ``half_life_s``).
        min_windows: Decoded windows to see before the first update.
        max_step: Largest move of the reference per window (affine-invariant
            distance).
        max_drift: Largest distance between the current and the initial
            reference; an update that would exceed it freezes the adapter.

    Raises:
        ValueError: On inconsistent settings.
    """

    def __init__(
        self,
        method: str | None = None,
        rate: float | None = None,
        half_life_s: float | None = None,
        step_s: float = STEP_S,
        min_windows: int = 20,
        max_step: float = 0.05,
        max_drift: float = 1.5,
    ) -> None:
        if method not in (None, "riemann", "euclid"):
            raise ValueError(f"unknown method {method!r}")
        if rate is not None and half_life_s is not None:
            raise ValueError("give rate or half_life_s, not both")
        if half_life_s is not None:
            if half_life_s <= 0 or step_s <= 0:
                raise ValueError("half_life_s and step_s must be > 0")
            rate = 1.0 - 0.5 ** (step_s / half_life_s)
        rate = 0.01 if rate is None else rate
        if not 0.0 < rate <= 1.0:
            raise ValueError("rate must be in (0, 1]")
        if min_windows < 0 or max_step <= 0 or max_drift <= 0:
            raise ValueError("need min_windows >= 0, max_step > 0 and max_drift > 0")
        self.method = method
        self.rate = rate
        self.min_windows = min_windows
        self.max_step = max_step
        self.max_drift = max_drift
        self.paused = False
        self.frozen = False
        self.seen = 0
        self.updates = 0
        self._initial: FloatArray | None = None
        self._reference: FloatArray | None = None
        self._decoder: OnlineDecoder | None = None
        self._listeners: list[Callable[[FloatArray], None]] = []

    @property
    def reference(self) -> FloatArray | None:
        """Current reference (``None`` before the first window)."""
        return self._reference

    @property
    def initial_reference(self) -> FloatArray | None:
        """Reference at the time of the first window (never modified)."""
        return self._initial

    @property
    def drift(self) -> float:
        """Distance between the current and the initial reference."""
        if self._initial is None or self._reference is None:
            return 0.0
        return riemann_distance(self._initial, self._reference)

    def add_listener(self, listener: Callable[[FloatArray], None]) -> None:
        """Call ``listener(reference)`` after each update (e.g. ``PersonalModel.set_reference``)."""
        self._listeners.append(listener)

    def attach(self, decoder: OnlineDecoder) -> None:
        """Follow the windows of ``decoder`` and update its reference.

        Raises:
            ValueError: If the model does not use alignment, or its method
                differs from the adapter's.
        """
        alignment = decoder.card.alignment
        if alignment is None:
            raise ValueError("online re-centering needs a model trained on aligned epochs")
        if self.method is None:
            self.method = alignment
        elif self.method != alignment:
            raise ValueError(
                f"adapter method {self.method!r} differs from the model's {alignment!r}"
            )
        self._decoder = decoder
        decoder.add_window_hook(self._on_window)

    def observe(self, window: npt.ArrayLike, reference: FloatArray | None = None) -> bool:
        """Account for one decoded window ``(n_channels, n_times)``.

        Args:
            window: Filtered, not aligned window.
            reference: Reference in use, to start from (default: the adapter's).

        Returns:
            Whether the reference moved.
        """
        if self._reference is None:
            if reference is None:
                raise ValueError("no reference yet")
            self._reference = np.array(reference, dtype=np.float64)
            self._initial = self._reference.copy()
        self.seen += 1
        if self.frozen or self.paused or self.seen <= self.min_windows:
            return False
        cov = covariances(np.asarray(window, dtype=np.float64)[None])[0]
        return self._update(cov)

    def snapshot(self) -> _AdapterState:
        """State to restore with :meth:`restore`."""
        ref = self._reference
        return _AdapterState(
            ref if ref is None else ref.copy(), self.seen, self.updates, self.frozen
        )

    def restore(self, state: _AdapterState) -> None:
        """Restore the reference (and counters) of a snapshot; also in the decoder."""
        self._reference = None if state.reference is None else state.reference.copy()
        self.seen, self.updates, self.frozen = state.seen, state.updates, state.frozen
        if self._reference is not None:
            self._publish(self._reference)

    # -- internals -------------------------------------------------------------

    def _on_window(self, window: FloatArray, prediction: Prediction) -> None:
        decoder = self._decoder
        if decoder is None or decoder.reference_covariance is None:
            return
        if self.observe(window, decoder.reference_covariance):
            assert self._reference is not None
            self._publish(self._reference)

    def _publish(self, reference: FloatArray) -> None:
        if self._decoder is not None:
            self._decoder.set_reference(reference)
        for listener in self._listeners:
            listener(reference)

    def _update(self, cov: FloatArray) -> bool:
        ref, initial = self._reference, self._initial
        assert ref is not None and initial is not None
        if self.method == "euclid":
            candidate = _sym((1.0 - self.rate) * ref + self.rate * cov)
            moved = riemann_distance(ref, candidate)
            if moved > self.max_step:
                candidate = geodesic_step(ref, candidate, self.max_step / moved)
        else:
            distance = riemann_distance(ref, cov)
            if distance < _EPS:
                return False
            candidate = geodesic_step(ref, cov, min(self.rate, self.max_step / distance))
        if riemann_distance(initial, candidate) > self.max_drift:
            self.frozen = True
            logger.warning(
                "re-centering frozen: drift would exceed %.2f after %d updates",
                self.max_drift,
                self.updates,
            )
            return False
        self._reference = candidate
        self.updates += 1
        return True


@dataclass(frozen=True)
class _PseudoItem:
    window: FloatArray
    label: str
    confidence: float


class PseudoLabelTuner:
    """Collect confident windows as pseudo-labelled training data (off by default).

    The predictions of the model are used as labels: the model then learns from
    its own decisions, which reinforces its mistakes (confirmation bias). It is
    therefore disabled by default, limited to a minority of the training set and
    kept apart from the true labels of the calibration.

    Args:
        threshold: Smoothed top probability above which a window is kept.
        enabled: Turn collection on.
        smoothing: Weight of each new probability vector in the running average.
        max_share: Largest share of pseudo-labels in the training set, overall
            and per class.
        max_stored: Windows kept (the most recent ones).
    """

    def __init__(
        self,
        threshold: float = 0.9,
        enabled: bool = False,
        smoothing: float = 0.3,
        max_share: float = 0.3,
        max_stored: int = 500,
    ) -> None:
        if not 0.5 <= threshold < 1.0:
            raise ValueError("threshold must be in [0.5, 1)")
        if not 0.0 < smoothing <= 1.0 or not 0.0 <= max_share < 1.0 or max_stored < 1:
            raise ValueError("invalid smoothing, max_share or max_stored")
        self.threshold = threshold
        self.enabled = enabled
        self.paused = False
        self.smoothing = smoothing
        self.max_share = max_share
        self.max_stored = max_stored
        self._items: list[_PseudoItem] = []
        self._smooth: dict[str, float] | None = None

    def __len__(self) -> int:
        return len(self._items)

    def attach(self, decoder: OnlineDecoder) -> None:
        """Follow the windows of ``decoder``."""

        def hook(window: FloatArray, prediction: Prediction) -> None:
            self.observe(window, prediction.probabilities)

        decoder.add_window_hook(hook)

    def observe(self, window: npt.ArrayLike, probabilities: Mapping[str, float] | None) -> bool:
        """Smooth the probabilities and keep the window if confident; returns whether kept."""
        if probabilities is None:
            return False
        if self._smooth is None:
            self._smooth = dict(probabilities)
        else:
            a = self.smoothing
            self._smooth = {c: (1 - a) * self._smooth[c] + a * p for c, p in probabilities.items()}
        if not self.enabled or self.paused:
            return False
        smooth = self._smooth
        label = max(smooth, key=lambda c: smooth[c])
        confidence = smooth[label]
        if confidence <= self.threshold:
            return False
        self._items.append(_PseudoItem(np.array(window, dtype=np.float64), label, confidence))
        del self._items[: -self.max_stored]
        return True

    def training_set(self, true_counts: Mapping[str, int]) -> tuple[FloatArray, npt.NDArray[Any]]:
        """Pseudo-labelled windows to add to a training set with ``true_counts`` per class.

        Per class, at most ``max_share / (1 - max_share) * true_count`` windows
        (the most confident ones), so pseudo-labels stay under ``max_share`` of the set.
        """
        if not self.enabled or not self._items:
            return np.empty((0, 0, 0)), np.empty((0,), dtype=np.str_)
        ratio = self.max_share / (1.0 - self.max_share)
        chosen: list[_PseudoItem] = []
        for label, count in true_counts.items():
            items = sorted(
                (i for i in self._items if i.label == label),
                key=lambda i: i.confidence,
                reverse=True,
            )
            chosen += items[: math.floor(ratio * count + 1e-9)]
        if not chosen:
            return np.empty((0, 0, 0)), np.empty((0,), dtype=np.str_)
        return np.stack([i.window for i in chosen]), np.array([i.label for i in chosen])

    def snapshot(self) -> list[_PseudoItem]:
        """Collected windows (restore with :meth:`restore`)."""
        return list(self._items)

    def restore(self, items: list[_PseudoItem]) -> None:
        self._items = list(items)


@dataclass(frozen=True)
class _GuardEntry:
    score: float
    state: _GuardState


@dataclass(frozen=True)
class _GuardState:
    adapter: _AdapterState | None
    personal: PersonalState | None
    pseudo: list[_PseudoItem] | None
    reference: FloatArray | None


class AdaptationGuard:
    """Snapshots and automatic rollback of the adaptive parts.

    The initial reference (:attr:`initial_reference`, read-only) and the
    generic model (``personal.generic``) are never modified by adaptation;
    :meth:`reset` goes back to them.

    Protocol: call :meth:`begin_block` when a block of trials starts (it takes
    the snapshot), and :meth:`end_block` with the block's score when it ends.
    The score is the ``overall`` of the progress tracker's ``BlockScore``; without
    cues, omit it: the mean confidence (top probability) of the windows decoded
    during the block is used, an unsupervised proxy that drops when the
    decoder becomes indecisive.

    Args:
        decoder: Decoder whose reference is adapted (hooked to follow the
            confidence).
        adapter: Re-centering adapter, if used.
        personal: Personal model, if used.
        tuner: Pseudo-label tuner, if used.
        max_drop: Fall of the score, relative to the best of the last
            ``history`` blocks, that triggers a rollback.
        history: Blocks considered for the best recent score.
        cooldown_blocks: Blocks during which adaptation is paused after a rollback.
    """

    def __init__(
        self,
        decoder: OnlineDecoder,
        adapter: RecenteringAdapter | None = None,
        personal: PersonalModel | None = None,
        tuner: PseudoLabelTuner | None = None,
        max_drop: float = 0.10,
        history: int = 3,
        cooldown_blocks: int = 1,
    ) -> None:
        if max_drop <= 0 or history < 1 or cooldown_blocks < 0:
            raise ValueError("need max_drop > 0, history >= 1 and cooldown_blocks >= 0")
        self.decoder = decoder
        self.adapter = adapter
        self.personal = personal
        self.tuner = tuner
        self.max_drop = max_drop
        self.history = history
        self.cooldown_blocks = cooldown_blocks
        self.rollbacks: list[dict[str, float]] = []
        self.blocks = 0
        self._initial: FloatArray | None = None
        self._capture_initial()
        self._entries: list[_GuardEntry] = []
        self._pending: _GuardState | None = None
        self._cooldown = 0
        self._message_blocks = 0
        self._confidences: list[float] = []
        decoder.add_window_hook(self._on_window)
        if adapter is not None and personal is not None:
            adapter.add_listener(personal.set_reference)

    @property
    def initial_reference(self) -> FloatArray | None:
        """Reference the session started with (read-only; ``None`` while warming up)."""
        return self._initial

    def begin_block(self) -> None:
        """Take the snapshot of the state the block starts with."""
        self._capture_initial()
        self._pending = self._snapshot()
        self._confidences = []

    def running_score(self) -> float | None:
        """Mean confidence of the windows decoded since :meth:`begin_block`."""
        return float(np.mean(self._confidences)) if self._confidences else None

    def end_block(self, score: float | None = None) -> bool:
        """Close a block; roll back if its score fell too far. Returns whether it did.

        Args:
            score: Progress score of the block (``BlockScore.overall``); ``None``
                uses :meth:`running_score`.
        """
        if score is None:
            score = self.running_score()
        if score is None:
            return False
        if self._pending is None:
            self._pending = self._snapshot()
        self.blocks += 1
        recent = self._entries[-self.history :]
        if recent:
            best = max(recent, key=lambda e: e.score)
            if score < best.score - self.max_drop:
                self._restore(best.state)
                event = {"block": float(self.blocks), "score": score, "best": best.score}
                self.rollbacks.append(event)
                logger.warning(
                    "adaptation rolled back at block %d: score %.3f vs best recent %.3f",
                    self.blocks,
                    score,
                    best.score,
                )
                self._pending = None
                self._set_paused(self.cooldown_blocks > 0)
                self._cooldown = self.cooldown_blocks
                self._message_blocks = 1
                return True
        self._entries.append(_GuardEntry(score, self._pending))
        self._pending = None
        if self._cooldown > 0:
            self._cooldown -= 1
            if self._cooldown == 0:
                self._set_paused(False)
        self._message_blocks = max(0, self._message_blocks - 1)
        return False

    def reset(self) -> None:
        """Go back to the initial reference and drop every adaptive part."""
        if self._initial is not None:
            self.decoder.set_reference(self._initial)
            if self.personal is not None:
                self.personal.set_reference(self._initial)
        if self.adapter is not None and self._initial is not None:
            self.adapter.restore(_AdapterState(self._initial.copy(), 0, 0, False))
        if self.personal is not None:
            from openmind_core.calibration.adaptive import PersonalState

            self.personal.set_state(PersonalState(None, 0.0, 0))
        if self.tuner is not None:
            self.tuner.restore([])

    def status_message(self) -> str | None:
        """Short text for the decoder status, ``None`` when all is well."""
        if self._message_blocks > 0:
            return "adaptation rolled back"
        if self.adapter is not None and self.adapter.frozen:
            return "adaptation frozen (drift limit)"
        return None

    # -- internals -------------------------------------------------------------

    def _capture_initial(self) -> None:
        if self._initial is not None:
            return
        ref = None if self.adapter is None else self.adapter.initial_reference
        if ref is None:
            ref = self.decoder.reference_covariance
        if ref is not None:
            self._initial = np.array(ref, dtype=np.float64)
            self._initial.setflags(write=False)

    def _on_window(self, window: FloatArray, prediction: Prediction) -> None:
        if prediction.probabilities:
            self._confidences.append(max(prediction.probabilities.values()))

    def _set_paused(self, paused: bool) -> None:
        if self.adapter is not None:
            self.adapter.paused = paused
        if self.tuner is not None:
            self.tuner.paused = paused

    def _snapshot(self) -> _GuardState:
        ref = self.decoder.reference_covariance
        return _GuardState(
            self.adapter.snapshot() if self.adapter is not None else None,
            self.personal.get_state() if self.personal is not None else None,
            self.tuner.snapshot() if self.tuner is not None else None,
            None if ref is None else ref.copy(),
        )

    def _restore(self, state: _GuardState) -> None:
        if state.reference is not None:
            self.decoder.set_reference(state.reference)
            if self.personal is not None:
                self.personal.set_reference(state.reference)
        if self.adapter is not None and state.adapter is not None:
            self.adapter.restore(state.adapter)
        if self.personal is not None and state.personal is not None:
            self.personal.set_state(state.personal)
        if self.tuner is not None and state.pseudo is not None:
            self.tuner.restore(state.pseudo)
