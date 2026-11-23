"""A pre-trained EEG foundation model (LaBraM) as a calibration competitor (``CORE-060``).

LaBraM (Jiang et al., ICLR 2024) is a transformer pre-trained on about 2 500 hours
of EEG. Braindecode ships its architecture and republishes the weights on the
Hugging Face Hub (``braindecode/labram-pretrained``); they are **downloaded at run
time** (Hugging Face cache, ``HF_HOME``) and never stored in a repository.

What the model expects, and what :class:`FoundationClassifier` does about it so
that the rest of the pipeline (epochs in µV at the headset rate, alignment, the
calibration curve) stays unchanged:

* **Sampling rate 200 Hz**: epochs are resampled from ``srate`` with
  ``scipy.signal.resample_poly`` (250 Hz -> 200 Hz is 4/5).
* **Patches of one second** (200 samples): a 2 s epoch gives two time patches per
  channel, every (channel, patch) pair being a token.
* **Named channels**: each channel selects its own position embedding among the
  128 of the 10-20/10-10 list, so ``ch_names`` must name the epoch channels
  (montage C: ``FC3 FC4 C3 Cz C4 CP3 CP4 CPz``, all known).
* **Amplitude**: LaBraM is trained on µV divided by 100. Every channel is divided
  by its standard deviation on the training set, then multiplied by
  ``target_std`` (0.2, i.e. about 20 µV). This also holds for aligned epochs,
  whose scale is arbitrary.
* **Output layer**: the checkpoint has none; a new linear layer is created for the
  classes of the task (``fc_norm`` and the transformer are pre-trained).

Training reuses :class:`~openmind_core.models.deep.DeepClassifier`: ``fit`` trains
the strategy ``fit_strategy`` on the source subjects (``full`` by default, with a
low learning rate), and ``fine_tune`` offers the same strategies as for EEGNet,
``head`` (new output layer), ``final`` (last transformer block, final norms and
output layer), ``adapter`` (a 1x1 spatial layer before the encoder) and ``full``.
Frozen modules stay in evaluation mode.

These models are research only: :func:`openmind_core.models.registry.save_model`
refuses them.

Needs the ``foundation`` extra (``torch``, ``braindecode``, ``huggingface_hub``).
"""

from __future__ import annotations

import logging
from fractions import Fraction
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import torch
from braindecode.models import Labram
from braindecode.models.labram import LABRAM_CHANNEL_ORDER
from scipy.signal import resample_poly
from torch import nn

from openmind_core.models.deep import DeepClassifier, _as_epochs

logger = logging.getLogger(__name__)

#: Hugging Face repository of the pre-trained LaBraM weights.
LABRAM_REPO: Final = "braindecode/labram-pretrained"

#: Sampling rate LaBraM was pre-trained at, in Hz.
LABRAM_SRATE: Final = 200.0

#: Strategies usable to train the model on the source subjects (``fit``).
FIT_STRATEGIES: Final[tuple[str, ...]] = ("head", "final", "full")


class _LabramNet(nn.Module):
    """LaBraM with the channel names bound: ``(batch, channels, times) -> logits``."""

    def __init__(self, net: Labram, ch_names: tuple[str, ...]) -> None:
        super().__init__()
        self.net = net
        self.ch_names = list(ch_names)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x, ch_names=self.ch_names)  # type: ignore[no-any-return]


class FoundationClassifier(DeepClassifier):
    """LaBraM with a classification head, behind the ``DeepClassifier`` interface.

    Args:
        ch_names: Names of the epoch channels, in order (10-20/10-10 names known
            to LaBraM). Required.
        srate: Sampling rate of the epochs, in Hz (resampled to 200 Hz).
        pretrained: Load the pre-trained weights (``False`` builds a randomly
            initialized network, for tests).
        repo_id: Hugging Face repository of the weights.
        fit_strategy: What ``fit`` trains: ``head``, ``final`` or ``full``.
        target_std: Standard deviation of the normalized inputs (LaBraM works on
            µV / 100).
        encoder_params: Architecture overrides, only with ``pretrained=False``
            (for example ``{"num_layers": 2}`` for a tiny test network).
        max_epochs, batch_size, lr, weight_decay, valid_fraction, patience,
        device, seed: As in :class:`DeepClassifier` (the default ``lr`` is lower:
            the encoder is pre-trained).

    Attributes (after ``fit``): those of :class:`DeepClassifier`; ``n_times_`` is
        the number of samples at 200 Hz.
    """

    #: Not registrable (see :func:`openmind_core.models.registry.save_model`).
    research_only = True

    def __init__(
        self,
        ch_names: tuple[str, ...] | list[str] | None = None,
        srate: float = 250.0,
        pretrained: bool = True,
        repo_id: str = LABRAM_REPO,
        fit_strategy: str = "full",
        target_std: float = 0.2,
        encoder_params: dict[str, Any] | None = None,
        max_epochs: int = 30,
        batch_size: int = 64,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        valid_fraction: float = 0.2,
        patience: int = 8,
        device: str = "auto",
        seed: int = 0,
    ) -> None:
        super().__init__(
            architecture="labram",
            srate=srate,
            max_epochs=max_epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            valid_fraction=valid_fraction,
            patience=patience,
            device=device,
            seed=seed,
        )
        self.ch_names = ch_names
        self.pretrained = pretrained
        self.repo_id = repo_id
        self.fit_strategy = fit_strategy
        self.target_std = target_std
        self.encoder_params = encoder_params

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> FoundationClassifier:
        """Build the network and train ``fit_strategy`` on epochs ``X``, labels ``y``."""
        data = _as_epochs(X)
        labels = np.asarray(y)
        names = self._check_channels(data.shape[1])
        if self.fit_strategy not in FIT_STRATEGIES:
            raise ValueError(
                f"unknown fit_strategy {self.fit_strategy!r}; available: {list(FIT_STRATEGIES)}"
            )
        self.classes_ = np.unique(labels)
        if len(self.classes_) < 2:
            raise ValueError("need at least two classes to train")
        self.scale_ = data.std(axis=(0, 2)).astype(np.float32)
        self.scale_[self.scale_ == 0] = 1.0
        torch.manual_seed(self.seed)
        inputs = self._normalize(data)
        self.n_times_ = int(inputs.shape[2])
        self.module_ = _LabramNet(
            self._build_encoder(data.shape[1], len(self.classes_), self.n_times_), names
        )
        names_to_train = self._trainable(self.fit_strategy)
        for name, parameter in self.module_.named_parameters():
            parameter.requires_grad = names_to_train is None or any(
                name == p or name.startswith(p + ".") for p in names_to_train
            )
        target = np.searchsorted(self.classes_, labels)
        self._optimize(
            self.module_,
            inputs,
            target,
            max_epochs=self.max_epochs,
            lr=self.lr,
            patience=self.patience,
            valid_fraction=self.valid_fraction,
            train_modules=names_to_train,
        )
        for parameter in self.module_.parameters():
            parameter.requires_grad = True
        return self

    # -- internals -----------------------------------------------------------

    def _check_channels(self, n_chans: int) -> tuple[str, ...]:
        if self.ch_names is None:
            raise ValueError(
                "FoundationClassifier needs ch_names (the names of the epoch channels)"
            )
        names = tuple(self.ch_names)
        if len(names) != n_chans:
            raise ValueError(f"{len(names)} channel names given for {n_chans} channels")
        unknown = [n for n in names if n.upper() not in LABRAM_CHANNEL_ORDER]
        if unknown:
            raise ValueError(f"channels unknown to LaBraM: {unknown}")
        return names

    def _trainable(self, strategy: str) -> tuple[str, ...] | None:
        if strategy == "head":
            return ("net.final_layer",)
        if strategy == "final":
            last = len(self.module_.net.blocks) - 1  # type: ignore[union-attr, arg-type]
            return (f"net.blocks.{last}", "net.norm", "net.fc_norm", "net.final_layer")
        if strategy == "adapter":
            return ()
        return None

    def _normalize(self, data: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Per-channel standardization, scaled to ``target_std``, resampled to 200 Hz."""
        scaled = super()._normalize(data) * np.float32(self.target_std)
        ratio = Fraction(LABRAM_SRATE / self.srate).limit_denominator(1000)
        if ratio != 1:
            scaled = resample_poly(scaled, ratio.numerator, ratio.denominator, axis=-1)
        return np.ascontiguousarray(scaled, dtype=np.float32)

    def _build_encoder(self, n_chans: int, n_outputs: int, n_times: int) -> Labram:
        """LaBraM for ``n_times`` samples at 200 Hz, with a new output layer."""
        if not self.pretrained:
            return Labram(
                n_chans=n_chans,
                n_times=n_times,
                n_outputs=n_outputs,
                sfreq=LABRAM_SRATE,
                **(self.encoder_params or {}),
            )
        try:
            return Labram.from_pretrained(
                self.repo_id,
                n_chans=n_chans,
                n_times=n_times,
                n_outputs=n_outputs,
                sfreq=LABRAM_SRATE,
            )
        except ImportError as error:
            raise ImportError(
                "loading LaBraM needs the foundation extra "
                "(uv sync --extra foundation: huggingface_hub, safetensors)"
            ) from error
        except Exception as error:
            raise RuntimeError(
                f"could not load the pre-trained weights {self.repo_id!r} from the "
                "Hugging Face Hub (offline? set HF_HOME to a cache that already holds "
                f"them): {error}"
            ) from error


__all__ = ["FIT_STRATEGIES", "LABRAM_REPO", "LABRAM_SRATE", "FoundationClassifier"]
