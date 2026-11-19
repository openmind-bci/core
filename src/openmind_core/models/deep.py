"""Deep motor-imagery decoders (EEGNet, ShallowConvNet) with a scikit-learn interface.

The architectures come from Braindecode; training is a small PyTorch loop kept
here on purpose: fast calibration (``CORE-060``) needs to freeze layers, swap the
output layer and fine-tune on a few epochs, which a hand-written loop makes
explicit.

=============  ==============================================================
``eegnet``     EEGNet (Lawhern et al., 2018): temporal convolution, depthwise
               spatial filters, separable convolution. Compact (a few thousand
               weights), designed to transfer across subjects.
``shallow``    ShallowConvNet (Schirrmeister et al., 2017): temporal then
               spatial convolution, squaring, average pooling and log: a
               learned, end-to-end version of filter-bank CSP.
=============  ==============================================================

Inputs are epochs ``(n_epochs, n_channels, n_times)`` in µV, already
band-passed by :mod:`openmind_core.preprocessing`. Each channel is divided by
its standard deviation on the training set (stored in ``scale_``), so the
networks see values around 1 whatever the headset gain.

Needs the ``deep`` extra (``torch``, ``braindecode``).
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import torch
from braindecode.models import EEGNet, ShallowFBCSPNet
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from torch import nn

logger = logging.getLogger(__name__)

#: Supported architectures.
ARCHITECTURES: Final[tuple[str, ...]] = ("eegnet", "shallow")


def build_network(
    architecture: str, n_chans: int, n_outputs: int, n_times: int, srate: float
) -> nn.Module:
    """Build an untrained network.

    EEGNet's temporal kernel is set to half a second, as in the original paper
    (64 samples at 128 Hz). ShallowConvNet's defaults are already tuned for
    250 Hz.

    Raises:
        KeyError: If the architecture is unknown.
    """
    network: nn.Module
    if architecture == "eegnet":
        network = EEGNet(
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_times=n_times,
            kernel_length=round(srate / 2),
            final_layer_with_constraint=True,  # max-norm on the classifier, as in the paper
        )
    elif architecture == "shallow":
        network = ShallowFBCSPNet(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times)
    else:
        raise KeyError(f"unknown architecture {architecture!r}; available: {list(ARCHITECTURES)}")
    return network


def resolve_device(device: str) -> torch.device:
    """``"auto"`` picks CUDA, then Apple MPS, then the CPU."""
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DeepClassifier(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    """Train a Braindecode network like a scikit-learn classifier.

    Training uses AdamW with a cosine learning-rate schedule. A stratified
    ``valid_fraction`` of the training epochs is held out; the weights with the
    lowest validation loss are kept, and training stops after ``patience``
    epochs without improvement.

    Args:
        architecture: One of :data:`ARCHITECTURES`.
        srate: Sampling rate of the epochs, in Hz.
        max_epochs: Maximum number of passes over the training data.
        batch_size: Mini-batch size.
        lr: Initial learning rate.
        weight_decay: AdamW weight decay.
        valid_fraction: Share of training epochs held out for early stopping
            (0 trains on everything for ``max_epochs``).
        patience: Epochs without validation improvement before stopping.
        device: ``"auto"``, ``"cpu"``, ``"cuda"`` or ``"mps"``.
        seed: Seed of the weight initialization, the split and the batches.

    Attributes (after ``fit``):
        classes_: Sorted class labels.
        scale_: Per-channel standard deviation used to normalize inputs.
        n_times_: Samples per training epoch.
        module_: The trained network (on the CPU).
        history_: Per-epoch ``train_loss``, ``valid_loss`` and ``valid_accuracy``.
    """

    def __init__(
        self,
        architecture: str = "eegnet",
        srate: float = 250.0,
        max_epochs: int = 200,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        valid_fraction: float = 0.2,
        patience: int = 40,
        device: str = "auto",
        seed: int = 0,
    ) -> None:
        self.architecture = architecture
        self.srate = srate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.valid_fraction = valid_fraction
        self.patience = patience
        self.device = device
        self.seed = seed

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> DeepClassifier:
        """Train a new network on epochs ``X`` and labels ``y``."""
        data = _as_epochs(X)
        labels = np.asarray(y)
        self.classes_ = np.unique(labels)
        if len(self.classes_) < 2:
            raise ValueError("need at least two classes to train")
        self.scale_ = data.std(axis=(0, 2)).astype(np.float32)
        self.scale_[self.scale_ == 0] = 1.0
        torch.manual_seed(self.seed)
        n_chans, n_times = data.shape[1:]
        self.n_times_ = int(n_times)
        self.module_ = build_network(
            self.architecture, n_chans, len(self.classes_), n_times, self.srate
        )
        target = np.searchsorted(self.classes_, labels)
        self._train(self._normalize(data), target)
        return self

    def predict_proba(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Class probabilities, columns in the order of ``classes_``."""
        logits = self._logits(self._normalize(_as_epochs(X)))
        return np.asarray(torch.softmax(logits, dim=1).numpy(), dtype=np.float64)

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[Any]:
        """Most probable class of each epoch."""
        return np.asarray(self.classes_[self.predict_proba(X).argmax(axis=1)])

    # -- internals -----------------------------------------------------------

    def _normalize(self, data: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        if data.shape[1] != len(self.scale_):
            raise ValueError(f"expected {len(self.scale_)} channels, got {data.shape[1]}")
        return np.asarray(data / self.scale_[None, :, None], dtype=np.float32)

    @torch.no_grad()
    def _logits(self, data: npt.NDArray[np.float32]) -> torch.Tensor:
        self.module_.eval()
        outputs = [
            self.module_(torch.from_numpy(data[i : i + 256])) for i in range(0, len(data), 256)
        ]
        return torch.cat(outputs) if outputs else torch.empty(0, len(self.classes_))

    def _train(self, data: npt.NDArray[np.float32], target: npt.NDArray[np.int64]) -> None:
        device = resolve_device(self.device)
        rng = np.random.default_rng(self.seed)
        train_idx = np.arange(len(data))
        valid_idx = np.array([], dtype=np.int64)
        counts = np.bincount(target)
        if self.valid_fraction > 0 and int(np.min(counts)) >= 2:
            train_idx, valid_idx = train_test_split(
                train_idx, test_size=self.valid_fraction, stratify=target, random_state=self.seed
            )
        x_train = torch.from_numpy(data[train_idx]).to(device)
        y_train = torch.from_numpy(target[train_idx]).to(device)
        x_valid = torch.from_numpy(data[valid_idx]).to(device)
        y_valid = torch.from_numpy(target[valid_idx]).to(device)

        module = self.module_.to(device)
        optimizer = torch.optim.AdamW(
            module.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        loss_fn = nn.CrossEntropyLoss()
        best_loss, best_state, since_best = np.inf, copy.deepcopy(module.state_dict()), 0
        self.history_: list[dict[str, float]] = []

        for epoch in range(self.max_epochs):
            module.train()
            order = rng.permutation(len(x_train))
            total = 0.0
            for start in range(0, len(order), self.batch_size):
                batch = torch.from_numpy(order[start : start + self.batch_size]).to(device)
                if len(batch) < 2:  # batch norm needs two samples
                    continue
                optimizer.zero_grad()
                loss = loss_fn(module(x_train[batch]), y_train[batch])
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(batch)
            schedule.step()
            record = {"epoch": float(epoch), "train_loss": total / max(len(order), 1)}
            if len(valid_idx):
                module.eval()
                with torch.no_grad():
                    logits = module(x_valid)
                    valid_loss = float(loss_fn(logits, y_valid))
                    accuracy = float((logits.argmax(dim=1) == y_valid).float().mean())
                record |= {"valid_loss": valid_loss, "valid_accuracy": accuracy}
                if valid_loss < best_loss:
                    best_loss, best_state, since_best = (
                        valid_loss,
                        copy.deepcopy(module.state_dict()),
                        0,
                    )
                else:
                    since_best += 1
            self.history_.append(record)
            if len(valid_idx) and since_best >= self.patience:
                logger.debug("early stop at epoch %d", epoch)
                break

        if len(valid_idx):
            module.load_state_dict(best_state)
        self.module_ = module.to("cpu").eval()


def _as_epochs(X: npt.ArrayLike) -> npt.NDArray[np.float32]:
    data = np.asarray(X, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"expected (n_epochs, n_channels, n_times), got shape {data.shape}")
    return data
