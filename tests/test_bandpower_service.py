"""Tests of the band power service and of the ``bandpower`` command."""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Iterator

import numpy as np
import numpy.typing as npt
import pytest
from typer.testing import CliRunner

from openmind_core.cli import app
from openmind_core.online.bandpower_service import format_frame, run_bandpower
from openmind_protocol import BandPower, decode, encode

SRATE = 250.0
LABELS = ["C3", "Cz", "C4"]
T0 = 1000.0


def _blocks(
    seconds: float, erd_after_s: float | None = None, block: int = 50
) -> Iterator[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
    """Noise plus a 10 Hz rhythm; the rhythm of C4 collapses after ``erd_after_s``."""
    rng = np.random.default_rng(0)
    n = int(seconds * SRATE)
    t = np.arange(n) / SRATE
    data = rng.normal(0.0, 2.0, (n, 3))
    amp = np.full((n, 3), 8.0)
    if erd_after_s is not None:
        amp[t >= erd_after_s, 2] = 1.0
    data += amp * np.sin(2 * np.pi * 10.0 * t)[:, None]
    for start in range(0, n, block):
        yield data[start : start + block], T0 + t[start : start + block]


def _run(
    blocks: Iterator[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]], **kwargs: object
):  # type: ignore[no-untyped-def]
    messages: list[BandPower] = []
    announced: list[str] = []
    stats = run_bandpower(
        blocks,
        messages.append,
        LABELS,
        SRATE,
        announce=announced.append,
        **kwargs,  # type: ignore[arg-type]
    )
    return stats, messages, announced


def test_erd_after_baseline_gives_negative_rel_mu() -> None:
    stats, messages, announced = _run(
        _blocks(16.0, erd_after_s=8.0), user="P01", baseline_s=6.0, rate_hz=10.0
    )
    assert announced[0].startswith("Baseline: stay relaxed")
    assert announced[-1] == "Baseline ready"
    assert stats.baseline_done
    for message in messages:
        decode(encode(message), BandPower)  # validates against the schema
    first = messages[0]
    assert all(channel.rel_mu is None for channel in first.channels)
    last = messages[-1]
    by_label = {channel.label: channel for channel in last.channels}
    assert by_label["C4"].rel_mu is not None
    assert by_label["C4"].rel_mu < -0.5
    assert abs(by_label["C3"].rel_mu or 0.0) < 0.5
    assert last.user == "P01"


def test_no_baseline_leaves_relative_null() -> None:
    stats, messages, announced = _run(_blocks(5.0), baseline_s=0.0)
    assert announced == []
    assert not stats.baseline_done
    assert messages
    assert all(c.rel_mu is None and c.rel_beta is None for m in messages for c in m.channels)
    assert all(c.mu is not None for c in messages[-1].channels)


@pytest.mark.parametrize("rate", [10.0, 4.0, 2.0])
def test_rate_limit(rate: float) -> None:
    _, messages, _ = _run(_blocks(8.0), rate_hz=rate)
    # 8 s of data, first frame once the 2 s window is full: 6 s of frames,
    # limited on the data clock (message ts is the Unix time of publication).
    expected = 6.0 * rate
    # frames come every 100 ms, so the rate is a bound, rounded down to it
    assert 0.8 * expected <= len(messages) <= expected + 1


def test_expected_missing_label_is_not_ok() -> None:
    _, messages, _ = _run(_blocks(4.0), expected_labels=["C3", "Cz", "C4", "Fp1"])
    channels = {c.label: c for c in messages[-1].channels}
    assert channels["Fp1"].ok is False
    assert channels["Fp1"].mu is None and channels["Fp1"].rel_mu is None
    assert channels["C3"].ok is True


def test_short_stream_without_frames_publishes_nothing() -> None:
    stats, messages, _ = _run(_blocks(1.0), baseline_s=5.0)
    assert messages == []
    assert not stats.baseline_done


def test_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="rate_hz"):
        run_bandpower(iter(()), lambda m: None, LABELS, SRATE, rate_hz=0.0)
    with pytest.raises(ValueError, match="baseline_s"):
        run_bandpower(iter(()), lambda m: None, LABELS, SRATE, baseline_s=-1.0)


def test_format_frame() -> None:
    _, messages, _ = _run(_blocks(4.0), expected_labels=["Fp1"])
    line = format_frame(messages[-1])
    assert " mu " in line and "C3=" in line and "Fp1=--" in line
    _, with_base, _ = _run(_blocks(10.0), baseline_s=3.0)
    assert " rel_mu " in format_frame(with_base[-1])


# --- command line ---------------------------------------------------------------

runner = CliRunner()


def test_cli_help() -> None:
    result = runner.invoke(app, ["bandpower", "--help"])
    assert result.exit_code == 0
    for option in ("--broker", "--user", "--rate", "--baseline-s", "--expected", "--dry-run"):
        assert option in result.output


def test_cli_missing_plug(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "openmind_plug", None)
    monkeypatch.setitem(sys.modules, "openmind_plug.inlet", None)
    result = runner.invoke(app, ["bandpower", "--dry-run"])
    assert result.exit_code == 1
    assert "uv sync --extra plug" in result.output


def test_cli_rejects_bad_user() -> None:
    result = runner.invoke(app, ["bandpower", "--user", "Thomas", "--dry-run"])
    assert result.exit_code != 0
    assert "P01" in result.output


# --- integration with Plug (skipped without the extra) --------------------------


def test_synthetic_stream_through_lsl() -> None:
    pytest.importorskip("openmind_plug")
    from openmind_plug.inlet import EEGInlet
    from openmind_plug.outlet import EEGOutlet, stream
    from openmind_plug.sources.synthetic import SyntheticSource

    source = SyntheticSource(montage="C", seed=1)
    stop = threading.Event()
    with EEGOutlet(source.info, source_id="openmind-plug:test:bandpower") as outlet:
        thread = threading.Thread(
            target=stream, args=(source,), kwargs={"outlet": outlet, "stop_event": stop}
        )
        thread.start()
        try:
            with EEGInlet(
                timeout=10.0, source_id="openmind-plug:test:bandpower", with_markers=False
            ) as inlet:
                deadline = time.monotonic() + 4.0

                def blocks() -> Iterator[tuple[npt.ArrayLike, npt.ArrayLike]]:
                    while time.monotonic() < deadline:
                        samples, stamps = inlet.read(timeout=0.1)
                        if len(stamps):
                            yield samples, stamps

                messages: list[BandPower] = []
                run_bandpower(
                    blocks(), messages.append, inlet.info.labels, inlet.info.srate, rate_hz=10.0
                )
        finally:
            stop.set()
            thread.join(timeout=5.0)
    assert messages
    assert {c.label for c in messages[-1].channels} == set(inlet.info.labels)
    decode(encode(messages[-1]), BandPower)
