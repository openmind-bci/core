"""Decoders: classical baselines now, deep models and the registry later."""

from openmind_core.models.baselines import BASELINES, csp_lda, make_baseline, mdm, ts_lr

__all__ = ["BASELINES", "csp_lda", "make_baseline", "mdm", "ts_lr"]
