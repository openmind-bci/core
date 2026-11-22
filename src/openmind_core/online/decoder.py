"""Online decoder: blocks of EEG in, class probabilities per window out.

The decoder reuses the code of offline training, step by step:

.. code-block:: text

    block (µV) ─ pick card channels ─┬─ StreamFilter ─ SlidingWindow ─ amplitude rejection
                                     │                                        │
                                     └─ JawClenchDetector ── "muscle" pause ──┤
                                                          └─ double clench → action
        ─ Aligner (if the model was trained on aligned epochs) ─ predict_proba

* Windows have the shape of an offline epoch ``(n_channels, n_times)`` and go
  through the same ``StreamFilter``, the same ``amplitude_mask`` and the same
  ``Aligner`` as the training epochs, so a window gives the same probabilities
  as the same epoch cut offline (tested).
* While the jaw detector sees muscle activity, windows are not decoded
  (``reason="muscle"``): muscle also contaminates the beta band.
* When the model was trained on aligned epochs, the session needs a reference
  covariance: given (a profile's :class:`~openmind_core.profiles.Reference`) or
  estimated on the first ``warmup_s`` seconds of clean windows (state
  ``"calibrating"``, predictions ``reason="warmup"``).

Out of scope here: resampling (the stream must already have the sampling rate
of the model), evidence accumulation over successive windows and MQTT
publication, which come next.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from openmind_core.alignment import Aligner
from openmind_core.models.registry import ModelCard
from openmind_core.preprocessing.artifacts import REJECT_PTP_UV, amplitude_mask
from openmind_core.preprocessing.filters import FilterConfig, StreamFilter
from openmind_core.preprocessing.jaw import JawClenchDetector
from openmind_core.preprocessing.windows import STEP_S, WINDOW_S, SlidingWindow
from openmind_core.profiles import Reference

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class OnlineConfig:
    """Settings of :class:`OnlineDecoder`.

    Args:
        window_s: Decoding window; must match the model's epoch length.
        step_s: Step between two windows.
        filters: Band-pass and notch; ``None`` uses those of the model card
            (the ones its training epochs went through), else the defaults.
        reject_uv: Peak-to-peak amplitude above which a window is rejected.
        warmup_s: Seconds of clean signal used to estimate the alignment
            reference when none is given.
    """

    window_s: float = WINDOW_S
    step_s: float = STEP_S
    filters: FilterConfig | None = None
    reject_uv: float = REJECT_PTP_UV
    warmup_s: float = 30.0


@dataclass(frozen=True)
class Prediction:
    """Result for one window.

    Attributes:
        timestamp: LSL time of the last sample of the window, or the time in
            seconds since the first sample when no timestamps were given.
        probabilities: Class → probability, in the card's class order, or
            ``None`` when the window was not decoded.
        ok: Whether the window was decoded.
        reason: Why it was not: ``"rejected"`` (amplitude), ``"muscle"`` (jaw
            activity in progress) or ``"warmup"`` (alignment reference not yet
            known).
    """

    timestamp: float
    probabilities: dict[str, float] | None
    ok: bool
    reason: str | None = None


@dataclass(frozen=True)
class DecoderStep:
    """Everything produced by one :meth:`OnlineDecoder.push`.

    Attributes:
        predictions: One per window completed by the block, oldest first.
        actions: Times of the double jaw clenches recognized (same clock as
            the prediction timestamps).
    """

    predictions: list[Prediction]
    actions: list[float]


def _normalize(label: str) -> str:
    """Case-insensitive channel key (``Fc3.``, ``EEG FC3`` and ``fc3`` match ``FC3``)."""
    return label.strip().rstrip(".").removeprefix("EEG ").strip().casefold()


class OnlineDecoder:
    """Decode a live EEG stream with a registered model.

    Args:
        model: Trained classifier with ``predict_proba`` (from ``load_model``).
        card: Its registry card (channels, sampling rate, alignment, filters).
        stream_labels: Channel labels of the stream, in the order of the blocks
            given to :meth:`push`.
        srate: Sampling rate of the stream, which must equal the card's.
        config: Settings.
        reference: Reference covariance of the user, used at once when the
            model needs alignment; its channels must be the card's.

    Raises:
        ValueError: If a card channel is missing from the stream, the sampling
            rate or the window length does not match the model, or the
            reference has other channels.
    """

    def __init__(
        self,
        model: Any,
        card: ModelCard,
        stream_labels: list[str] | tuple[str, ...],
        srate: float,
        config: OnlineConfig | None = None,
        reference: Reference | None = None,
    ) -> None:
        self.model = model
        self.card = card
        self.config = config = config or OnlineConfig()
        if not math.isclose(srate, card.srate, abs_tol=1e-6):
            raise ValueError(
                f"the stream runs at {srate} Hz but model {card.name!r} expects {card.srate} Hz "
                "(resampling is not supported)"
            )
        if round(config.window_s * srate) != card.n_times:
            raise ValueError(
                f"windows of {config.window_s} s are {round(config.window_s * srate)} samples "
                f"but model {card.name!r} takes {card.n_times}"
            )
        self.srate = srate
        self.n_stream_channels = len(stream_labels)
        keys: dict[str, int] = {}
        for index, label in enumerate(stream_labels):
            keys.setdefault(_normalize(label), index)
        missing = [c for c in card.channels if _normalize(c) not in keys]
        if missing:
            raise ValueError(
                f"model {card.name!r} needs channels {missing} missing from the stream "
                f"{list(stream_labels)}"
            )
        self._picked = [keys[_normalize(c)] for c in card.channels]
        n_channels = len(card.channels)
        filters = config.filters or (FilterConfig(**card.filters) if card.filters else None)
        self._filter = StreamFilter(srate, n_channels, filters)
        self._window = SlidingWindow(n_channels, srate, config.window_s, config.step_s)
        self._jaw: JawClenchDetector | None
        try:
            self._jaw = JawClenchDetector(srate, n_channels)
        except ValueError:  # sampling rate too low for the muscle band
            self._jaw = None
        self._aligner: Aligner | None = None
        self._needs_alignment = card.alignment is not None
        self._warmup: list[FloatArray] = []
        if self._needs_alignment and reference is not None:
            if tuple(_normalize(c) for c in reference.channels) != tuple(
                _normalize(c) for c in card.channels
            ):
                raise ValueError(
                    f"reference channels {list(reference.channels)} differ from the model's "
                    f"{list(card.channels)}"
                )
            self._aligner = Aligner.from_reference(reference.covariance, str(card.alignment))
        self._count = 0  # samples pushed so far

    @property
    def state(self) -> str:
        """``"calibrating"`` while the alignment reference is estimated, else ``"running"``."""
        return "calibrating" if self._needs_alignment and self._aligner is None else "running"

    @property
    def reference_covariance(self) -> FloatArray | None:
        """Reference covariance in use (to store in the profile), ``None`` before it is known."""
        return None if self._aligner is None else np.asarray(self._aligner.reference_)

    def push(self, block: npt.ArrayLike, timestamps: npt.ArrayLike | None = None) -> DecoderStep:
        """Process the next samples.

        Args:
            block: ``(n_samples, n_stream_channels)`` in µV.
            timestamps: LSL time of each sample, ``(n_samples,)``. Without
                them, times are seconds since the first sample.
        """
        data = np.asarray(block, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != self.n_stream_channels:
            raise ValueError(f"expected shape (n, {self.n_stream_channels}), got {data.shape}")
        n = data.shape[0]
        times = None if timestamps is None else np.asarray(timestamps, dtype=np.float64)
        if times is not None and times.shape != (n,):
            raise ValueError(f"expected {n} timestamps, got shape {times.shape}")
        picked = data[:, self._picked]
        predictions: list[Prediction] = []
        actions: list[float] = []
        start = self._count
        pos = 0
        while pos < n:
            stop = min(n, pos + self._until_next_window(start + pos))
            segment = picked[pos:stop]
            muscle = False
            if self._jaw is not None:
                for t in self._jaw.update(segment):
                    local = min(max(round(t * self.srate) - start, 0), n - 1)
                    actions.append(t if times is None else float(times[local]))
                muscle = self._jaw.active
            for window in self._window.push(self._filter(segment)):
                end = stop - 1
                stamp = (start + end) / self.srate if times is None else float(times[end])
                predictions.append(self._decode(window, stamp, muscle))
            pos = stop
        self._count += n
        return DecoderStep(predictions, actions)

    # -- internals -------------------------------------------------------------

    def _until_next_window(self, count: int) -> int:
        """Samples from ``count`` up to and including the next window end."""
        first, step = self._window.n_times, self._window.step
        if count < first:
            return first - count
        return step - (count - first) % step

    def _decode(self, window: FloatArray, stamp: float, muscle: bool) -> Prediction:
        if muscle:
            return Prediction(stamp, None, False, "muscle")
        if not amplitude_mask(window[None], self.config.reject_uv)[0]:
            return Prediction(stamp, None, False, "rejected")
        if self._needs_alignment and self._aligner is None:
            self._warmup.append(window)
            span = (len(self._warmup) - 1) * self._window.step + self._window.n_times
            if span < self.config.warmup_s * self.srate:
                return Prediction(stamp, None, False, "warmup")
            self._aligner = Aligner(str(self.card.alignment)).fit(np.stack(self._warmup))
            self._warmup = []
        x = window if self._aligner is None else self._aligner.transform(window)
        proba = np.asarray(self.model.predict_proba(np.asarray(x)[None]))[0]
        return Prediction(
            stamp, {c: float(p) for c, p in zip(self.card.classes, proba, strict=True)}, True
        )
