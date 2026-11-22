"""Real-time decoding: from EEG blocks to class probabilities and commands."""

from openmind_core.online.decoder import DecoderStep, OnlineConfig, OnlineDecoder, Prediction
from openmind_core.online.emitter import (
    CommandEmitter,
    HealthMonitor,
    LatencyMeter,
    StatusEmitter,
    decoder_state,
)
from openmind_core.online.evidence import Decision, EvidenceAccumulator, EvidenceConfig

__all__ = [
    "CommandEmitter",
    "Decision",
    "DecoderStep",
    "EvidenceAccumulator",
    "EvidenceConfig",
    "HealthMonitor",
    "LatencyMeter",
    "OnlineConfig",
    "OnlineDecoder",
    "Prediction",
    "StatusEmitter",
    "decoder_state",
]
