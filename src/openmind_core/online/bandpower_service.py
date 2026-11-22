"""Band power service: EEG blocks in, ``bandpower`` messages out.

The loop is independent of LSL and MQTT: it takes an iterator of
``(samples, timestamps)`` blocks and a ``publish`` callable, so it runs the same
on a live stream, a replay or synthetic blocks in tests.

Rate limiting and the baseline follow the **data clock** (sample timestamps and
counts), not the wall clock, so the result does not depend on how fast the
blocks are read.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from openmind_core.preprocessing.bandpower import BandPowerConfig, BandPowerEstimator
from openmind_protocol import BandPower

Block = tuple[npt.ArrayLike, npt.ArrayLike]

#: Tolerance on the publication interval (frames come every 100 ms, on a grid).
_EPS = 1e-6


@dataclass(frozen=True)
class ServiceStats:
    """What a run of :func:`run_bandpower` did."""

    frames: int
    published: int
    baseline_done: bool


def format_frame(message: BandPower) -> str:
    """One compact line per frame, for ``--dry-run``.

    Shows the change from the baseline of the mu band (``rel_mu``) when there
    is one, else the mu power in µV²/Hz; ``--`` for a channel that is not ok.
    """
    relative = any(channel.rel_mu is not None for channel in message.channels)
    parts = []
    for channel in message.channels:
        number = channel.rel_mu if relative else channel.mu
        if number is None:
            parts.append(f"{channel.label}=--")
        else:
            parts.append(
                f"{channel.label}={number:+.2f}" if relative else f"{channel.label}={number:.2f}"
            )
    unit = "rel_mu" if relative else "mu"
    return f"{message.ts:.2f} {unit} " + " ".join(parts)


def run_bandpower(
    blocks: Iterable[Block],
    publish: Callable[[BandPower], None],
    labels: Sequence[str],
    srate: float,
    *,
    user: str | None = None,
    rate_hz: float = 10.0,
    baseline_s: float = 0.0,
    expected_labels: Sequence[str] | None = None,
    announce: Callable[[str], None] | None = None,
    config: BandPowerConfig | None = None,
) -> ServiceStats:
    """Estimate band power from ``blocks`` and publish at most ``rate_hz`` frames/s.

    Args:
        blocks: ``(samples, timestamps)`` with samples ``(n, n_channels)`` in µV.
        publish: Called with each ``bandpower`` message.
        labels: Channel labels of the blocks.
        srate: Sampling rate in Hz.
        user: Pseudonym of the user, if any.
        rate_hz: Maximum publication rate (estimator frames come at about 10 Hz).
        baseline_s: Seconds of rest recorded at the start to build the baseline;
            0 means no baseline (``rel_*`` are null). Frames are published during
            the baseline, without ``rel_*``.
        expected_labels: Electrodes the viewer expects; absent ones are sent
            with ``ok = False``.
        announce: Receives progress messages (baseline start and end).
        config: Estimator settings.

    Returns:
        Counters of the run, once ``blocks`` is exhausted.
    """
    if rate_hz <= 0:
        raise ValueError(f"rate_hz must be positive, got {rate_hz}")
    if baseline_s < 0:
        raise ValueError(f"baseline_s must be >= 0, got {baseline_s}")
    say = announce or (lambda _text: None)
    estimator = BandPowerEstimator(labels, srate, config)
    interval = 1.0 / rate_hz
    recording = baseline_s > 0
    if recording:
        estimator.start_baseline()
        say(f"Baseline: stay relaxed, eyes open, for {baseline_s:g} s...")
    n_samples = n_frames = n_published = n_baseline_frames = 0
    last_published: float | None = None

    for samples, timestamps in blocks:
        frames = estimator.push(samples, timestamps)
        n_samples += int(np.shape(samples)[0])
        for frame in frames:
            n_frames += 1
            n_baseline_frames += recording
            stamp = frame.timestamp
            if (
                stamp is not None
                and last_published is not None
                and stamp - last_published < interval - _EPS
            ):
                continue
            message = frame.to_message(user, srate, expected_labels)
            publish(message)
            n_published += 1
            last_published = stamp
        if recording and n_samples >= baseline_s * srate and n_baseline_frames > 0:
            estimator.stop_baseline()
            recording = False
            say("Baseline ready")
    return ServiceStats(n_frames, n_published, estimator.baseline is not None)
