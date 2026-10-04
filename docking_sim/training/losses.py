from __future__ import annotations

from contextlib import contextmanager
from typing import Callable

import torch
import torch.nn.functional as F


def canonical_loss(name: str) -> str:
    key = str(name or "mse").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "mse": "mse",
        "l2": "mse",
        "huber": "huber",
        "smooth_l1": "smooth_l1",
        "smoothl1": "smooth_l1",
    }
    if key not in aliases:
        known = "mse, huber, smooth_l1"
        raise ValueError(f"Unknown critic loss '{name}'. Expected one of: {known}")
    return aliases[key]


def critic_loss(pred: torch.Tensor, target: torch.Tensor, name: str, *, reduction: str = "mean") -> torch.Tensor:
    key = canonical_loss(name)
    if key == "huber":
        return F.huber_loss(pred, target, reduction=reduction, delta=1.0)
    if key == "smooth_l1":
        return F.smooth_l1_loss(pred, target, reduction=reduction, beta=1.0)
    return F.mse_loss(pred, target, reduction=reduction)


class _FProxy:
    def __init__(self, real: object, mse_loss: Callable) -> None:
        self._real = real
        self.mse_loss = mse_loss

    def __getattr__(self, name: str):
        return getattr(self._real, name)


def _as_mse(name: str) -> Callable:
    def mse_loss(pred, target, size_average=None, reduce=None, reduction: str = "mean"):
        del size_average, reduce
        return critic_loss(pred, target, name, reduction=reduction)

    return mse_loss


@contextmanager
def use_critic_loss(train_fn, name: str):
    """Route the algorithm's F.mse_loss call through the selected critic loss."""
    globals_ = train_fn.__globals__
    original = globals_["F"]
    real = original._real if isinstance(original, _FProxy) else original
    globals_["F"] = _FProxy(real, _as_mse(name))
    try:
        yield
    finally:
        globals_["F"] = original


def _install_loss_hook(model) -> None:
    owner = type(model)
    for klass in type(model).__mro__:
        if "train" in klass.__dict__:
            owner = klass
            break
    if getattr(owner, "_docking_loss_hook", False):
        return
    raw = owner.__dict__["train"]

    def train(self, *args, **kwargs):
        name = getattr(self, "critic_loss_name", "mse")
        if canonical_loss(name) == "mse":
            return raw(self, *args, **kwargs)
        with use_critic_loss(raw, name):
            return raw(self, *args, **kwargs)

    owner.train = train
    owner._docking_loss_hook = True


def attach_critic_loss(model, name: str) -> None:
    _install_loss_hook(model)
    model.critic_loss_name = canonical_loss(name)
