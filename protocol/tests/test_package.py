"""Smoke tests for the protocol package skeleton."""

import openmind_protocol


def test_version_is_exposed() -> None:
    assert isinstance(openmind_protocol.__version__, str)
    assert openmind_protocol.__version__
