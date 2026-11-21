"""Reading the standard LSL streams: markers now, EEG ring buffer with CORE-080.

Core reads the streams through ``openmind_plug.inlet.EEGInlet`` (``plug``
extra), which follows the LSL stream specification of OpenMind Plug (1.2.0):
markers come from **every** ``openmind-markers`` stream (Plug, Sandbox...),
merged by timestamp on the local LSL clock.
"""

from openmind_core.io.markers import Trial, TrialCollector

__all__ = ["Trial", "TrialCollector"]
