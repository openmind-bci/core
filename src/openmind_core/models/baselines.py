"""Classical motor-imagery decoders: the baselines every new model must beat.

All baselines are scikit-learn pipelines that take epochs shaped
``(n_epochs, n_channels, n_times)`` in µV (already band-passed by
:mod:`openmind_core.preprocessing`) and predict command labels. Each one starts
by estimating a spatial covariance matrix per epoch, with the Oracle
Approximating Shrinkage (OAS) estimator, which stays well conditioned on short
epochs and few channels.

=============  ==============================================================
``csp-lda``    Common Spatial Patterns (log-variance of 6 spatial filters)
               followed by linear discriminant analysis. The historical
               reference of motor-imagery BCIs.
``mdm``        Minimum Distance to Mean: each class is a Riemannian mean of
               covariance matrices; an epoch goes to the closest mean. No
               hyper-parameter, robust, and the basis of Riemannian
               re-centering (``CORE-050``).
``ts-lr``      Tangent space projection at the Riemannian mean of the training
               covariances, then L2-regularized logistic regression. Usually
               the strongest classical baseline.
=============  ==============================================================
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

import numpy as np
import numpy.typing as npt
from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from pyriemann.tangentspace import TangentSpace
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#: Covariance estimator shared by every baseline.
COVARIANCE_ESTIMATOR: Final = "oas"


class AsFloat64(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """Cast epochs to float64 before covariance estimation.

    Epochs are stored as float32 to save memory; covariance estimation and
    matrix logarithms need double precision.
    """

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None) -> AsFloat64:
        """Nothing to learn."""
        return self

    def transform(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Return ``X`` as a float64 array of epochs."""
        data = np.asarray(X, dtype=np.float64)
        if data.ndim != 3:
            raise ValueError(f"expected (n_epochs, n_channels, n_times), got shape {data.shape}")
        return data


def csp_lda(n_filters: int = 6) -> Pipeline:
    """CSP (log-variance of ``n_filters`` filters) + shrinkage LDA."""
    return Pipeline(
        [
            ("float64", AsFloat64()),
            ("covariances", Covariances(estimator=COVARIANCE_ESTIMATOR)),
            ("csp", CSP(nfilter=n_filters, log=True)),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]
    )


def mdm() -> Pipeline:
    """Minimum Distance to Riemannian Mean."""
    return Pipeline(
        [
            ("float64", AsFloat64()),
            ("covariances", Covariances(estimator=COVARIANCE_ESTIMATOR)),
            ("mdm", MDM(metric="riemann")),
        ]
    )


def ts_lr(c: float = 1.0, seed: int = 0) -> Pipeline:
    """Tangent space + standardized L2 logistic regression."""
    return Pipeline(
        [
            ("float64", AsFloat64()),
            ("covariances", Covariances(estimator=COVARIANCE_ESTIMATOR)),
            ("tangent", TangentSpace(metric="riemann")),
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(C=c, max_iter=1000, random_state=seed)),
        ]
    )


#: Baselines by name, each built with its default settings.
BASELINES: Final[dict[str, Callable[[], Pipeline]]] = {
    "csp-lda": csp_lda,
    "mdm": mdm,
    "ts-lr": ts_lr,
}


def make_baseline(name: str) -> Pipeline:
    """Build a fresh, unfitted baseline by name.

    Raises:
        KeyError: If ``name`` is not in :data:`BASELINES`.
    """
    try:
        factory = BASELINES[name]
    except KeyError:
        raise KeyError(f"unknown baseline {name!r}; available: {sorted(BASELINES)}") from None
    return factory()
