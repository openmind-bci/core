"""Alignment of recording sessions, to reduce the differences between users and days.

Each session has its own spatial signature (electrode positions, contact
quality, head shape). Alignment whitens the signals of a session by a reference
covariance of that session, so that every session is centred on the same
point (the identity matrix). It uses no labels, only the signals.

- ``method="euclid"``: the reference is the arithmetic mean of the epoch
  covariances (Euclidean alignment, He and Wu, 2019).
- ``method="riemann"``: the reference is their Riemannian mean. Whitening the
  signals by it makes the Riemannian mean of the aligned covariances exactly the
  identity: this is Riemannian re-centering (Zanini et al., 2018), applied to
  the signals so that it works with every decoder, deep networks included.

Online, the reference of a session comes from its first epochs (calibration or
rest); :class:`Aligner` keeps it and transforms the following windows.
"""

from openmind_core.alignment.align import (
    METHODS,
    Aligner,
    align_epochs,
    covariances,
    reference_covariance,
)

__all__ = ["METHODS", "Aligner", "align_epochs", "covariances", "reference_covariance"]
