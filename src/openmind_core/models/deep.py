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

from openmind_core.calibration.finetune import DEFAULT_LR, STRATEGIES

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


#: Modules (children of the network) that each strategy trains, by architecture.
#: ``None`` stands for every module; the ``adapter`` strategy trains none of them.
_TRAINABLE: Final[dict[str, dict[str, tuple[str, ...] | None]]] = {
    "eegnet": {
        "head": ("final_layer",),
        "final": ("conv_separable_depth", "conv_separable_point", "bnorm_2", "final_layer"),
        "adapter": (),
        "full": None,
    },
    "shallow": {
        "head": ("final_layer",),
        "final": ("bnorm", "final_layer"),
        "adapter": (),
        "full": None,
    },
}


class ChannelAdapter(nn.Module):
    """A per-user 1x1 spatial layer: ``x + delta @ x``, mixing the channels.

    ``delta`` starts at zero, so the adapter is the identity before fine-tuning
    and weight decay pulls it back towards the identity.
    """

    def __init__(self, n_chans: int) -> None:
        super().__init__()
        self.delta = nn.Parameter(torch.zeros(n_chans, n_chans))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.einsum("cd,bdt->bct", self.delta, x)


class _Adapted(nn.Module):
    """A network preceded by an adapter (the parameters are shared, not copied)."""

    def __init__(self, adapter: nn.Module, network: nn.Module) -> None:
        super().__init__()
        self.adapter = adapter
        self.network = network

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(self.adapter(x))  # type: ignore[no-any-return]


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
        adapter_: The per-user input layer added by ``fine_tune("adapter")``
            (absent otherwise).
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

    def fine_tune(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        strategy: str = "head",
        epochs: int = 100,
        lr: float | None = None,
        sample_weight: npt.ArrayLike | None = None,
        patience: int = 20,
        valid_fraction: float = 0.2,
        min_validation_trials: int = 30,
    ) -> DeepClassifier:
        """Adapt this (pre-trained) model to a few trials of a new user.

        The model itself is left untouched: a fine-tuned **copy** is returned.
        Frozen layers keep their weights and their batch-norm statistics.

        Args:
            X: Calibration epochs ``(n, n_channels, n_times)``, in the same
                space as the pre-training data (same alignment).
            y: Their labels, among ``classes_``.
            strategy: ``"head"`` (output layer), ``"final"`` (last block and
                output layer), ``"adapter"`` (a new 1x1 spatial layer before the
                frozen network) or ``"full"`` (every weight).
            epochs: Maximum number of passes over the trials.
            lr: Learning rate (default: the strategy's, see
                :data:`openmind_core.calibration.finetune.DEFAULT_LR`).
            sample_weight: Optional weight of each trial in the loss.
            patience: Epochs without validation improvement before stopping.
            valid_fraction: Share of the trials held out for early stopping.
            min_validation_trials: With fewer trials nothing is held out and
                training runs for ``epochs`` (the best epoch cannot be told
                from so little data).

        Returns:
            A new classifier. With no trials (``n = 0``) it is an unchanged copy.

        Raises:
            ValueError: For an unknown strategy, a model that is not trained or
                labels outside ``classes_``.
        """
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}; available: {list(STRATEGIES)}")
        if not hasattr(self, "module_"):
            raise ValueError("the model is not trained")
        tuned = copy.deepcopy(self)
        data = _as_epochs(X)
        if len(data) == 0:
            return tuned
        labels = np.asarray(y)
        if not np.isin(labels, self.classes_).all():
            raise ValueError("calibration labels must be among the classes of the model")
        weights = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float32)
        if weights is not None and len(weights) != len(data):
            raise ValueError("sample_weight must have one value per epoch")
        torch.manual_seed(self.seed)
        n_chans = data.shape[1]
        if strategy == "adapter" and getattr(tuned, "adapter_", None) is None:
            tuned.adapter_ = ChannelAdapter(n_chans)
        target = np.searchsorted(self.classes_, labels)
        trainable = _TRAINABLE[self.architecture][strategy]
        runner = tuned._runner()
        prefix = "network." if isinstance(runner, _Adapted) else ""
        if strategy == "adapter":
            names: tuple[str, ...] | None = ("adapter",)
        elif trainable is None:
            names = None
        else:
            names = tuple(f"{prefix}{name}" for name in trainable)
        for name, parameter in runner.named_parameters():
            parameter.requires_grad = names is None or any(
                name == p or name.startswith(p + ".") for p in names
            )
        use_valid = len(data) >= min_validation_trials and valid_fraction > 0
        tuned._optimize(
            runner,
            tuned._normalize(data),
            target,
            max_epochs=epochs,
            lr=DEFAULT_LR[strategy] if lr is None else lr,
            patience=patience,
            valid_fraction=valid_fraction if use_valid else 0.0,
            sample_weight=weights,
            train_modules=names,
        )
        for parameter in runner.parameters():
            parameter.requires_grad = True
        return tuned

    # -- internals -----------------------------------------------------------

    def _runner(self) -> nn.Module:
        """The module that maps normalized epochs to logits (adapter included)."""
        adapter = getattr(self, "adapter_", None)
        return self.module_ if adapter is None else _Adapted(adapter, self.module_)

    def _normalize(self, data: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        if data.shape[1] != len(self.scale_):
            raise ValueError(f"expected {len(self.scale_)} channels, got {data.shape[1]}")
        return np.asarray(data / self.scale_[None, :, None], dtype=np.float32)

    @torch.no_grad()
    def _logits(self, data: npt.NDArray[np.float32]) -> torch.Tensor:
        runner = self._runner().eval()
        outputs = [runner(torch.from_numpy(data[i : i + 256])) for i in range(0, len(data), 256)]
        return torch.cat(outputs) if outputs else torch.empty(0, len(self.classes_))

    def _train(self, data: npt.NDArray[np.float32], target: npt.NDArray[np.int64]) -> None:
        self._optimize(
            self.module_,
            data,
            target,
            max_epochs=self.max_epochs,
            lr=self.lr,
            patience=self.patience,
            valid_fraction=self.valid_fraction,
        )

    def _optimize(
        self,
        module: nn.Module,
        data: npt.NDArray[np.float32],
        target: npt.NDArray[np.int64],
        max_epochs: int,
        lr: float,
        patience: int,
        valid_fraction: float,
        sample_weight: npt.NDArray[np.float32] | None = None,
        train_modules: tuple[str, ...] | None = None,
    ) -> None:
        """Train the parameters of ``module`` that require a gradient.

        Args:
            train_modules: Names (prefixes) of the submodules left in training
                mode; every other one stays in evaluation mode, so that frozen
                batch-norm layers keep their statistics. ``None`` trains all.
        """
        device = resolve_device(self.device)
        rng = np.random.default_rng(self.seed)
        train_idx = np.arange(len(data))
        valid_idx = np.array([], dtype=np.int64)
        counts = np.bincount(target)
        if valid_fraction > 0 and int(np.min(counts)) >= 2:
            train_idx, valid_idx = train_test_split(
                train_idx, test_size=valid_fraction, stratify=target, random_state=self.seed
            )
        x_train = torch.from_numpy(data[train_idx]).to(device)
        y_train = torch.from_numpy(target[train_idx]).to(device)
        x_valid = torch.from_numpy(data[valid_idx]).to(device)
        y_valid = torch.from_numpy(target[valid_idx]).to(device)
        w_train = (
            None
            if sample_weight is None
            else torch.from_numpy(np.asarray(sample_weight, dtype=np.float32)[train_idx]).to(device)
        )

        module = module.to(device)
        parameters = [p for p in module.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=self.weight_decay)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        loss_fn = nn.CrossEntropyLoss()
        weighted_loss_fn = nn.CrossEntropyLoss(reduction="none")
        best_loss, best_state, since_best = np.inf, copy.deepcopy(module.state_dict()), 0
        self.history_: list[dict[str, float]] = []

        for epoch in range(max_epochs):
            _set_training(module, train_modules)
            order = rng.permutation(len(x_train))
            total = 0.0
            for start in range(0, len(order), self.batch_size):
                batch = torch.from_numpy(order[start : start + self.batch_size]).to(device)
                if len(batch) < 2:  # batch norm needs two samples
                    continue
                optimizer.zero_grad()
                if w_train is None:
                    loss = loss_fn(module(x_train[batch]), y_train[batch])
                else:
                    weight = w_train[batch]
                    per_trial = weighted_loss_fn(module(x_train[batch]), y_train[batch])
                    loss = (per_trial * weight).sum() / weight.sum()
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
            if len(valid_idx) and since_best >= patience:
                logger.debug("early stop at epoch %d", epoch)
                break

        if len(valid_idx):
            module.load_state_dict(best_state)
        module.to("cpu").eval()


def _set_training(module: nn.Module, names: tuple[str, ...] | None) -> None:
    """Training mode for the submodules under ``names`` (all if ``None``), evaluation otherwise."""
    if names is None:
        module.train()
        return
    module.eval()
    for name, child in module.named_modules():
        if any(name == p or name.startswith(p + ".") for p in names):
            child.train()


def _as_epochs(X: npt.ArrayLike) -> npt.NDArray[np.float32]:
    data = np.asarray(X, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"expected (n_epochs, n_channels, n_times), got shape {data.shape}")
    return data
