from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Any, Callable


class TorchaxNotInstalledError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _load_modules() -> tuple[Any, Any]:
    try:
        torchax_mod = importlib.import_module("torchax")
        interop_mod = importlib.import_module("torchax.interop")
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in runtime usage
        raise TorchaxNotInstalledError(
            "thrml.th requires torchax>=0.0.10. Install it with `pip install torchax`."
        ) from exc
    return torchax_mod, interop_mod


def require_torchax():
    return _load_modules()[0]


def require_interop():
    return _load_modules()[1]


def torch_view(tree):
    return require_interop().torch_view(tree)


def jax_view(tree):
    return require_interop().jax_view(tree)


def call_jax(func: Callable[..., Any], *args, **kwargs):
    return require_interop().call_jax(func, *args, **kwargs)


def call_torch(func: Callable[..., Any], *args, **kwargs):
    return require_interop().call_torch(func, *args, **kwargs)
