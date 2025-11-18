from __future__ import annotations

from threading import Lock
from typing import Sequence

from thrml.block_management import Block
from thrml.block_sampling import BlockSamplingProgram, SamplingSchedule

from .config import CompileConfig, EnableConfig
from .interop import call_jax, call_torch, jax_view, require_torchax, torch_view
from .keys import KeyStream, make_key
from .sampler import ThrmlSampler, compile_sampler

__all__ = [
    "call_jax",
    "call_torch",
    "compile",
    "CompileConfig",
    "EnableConfig",
    "enable",
    "disable",
    "is_enabled",
    "jax_view",
    "KeyStream",
    "make_key",
    "ThrmlSampler",
    "torch_view",
]

__version__ = "0.1.0"

_ENABLE_LOCK = Lock()
_ENABLED = False


def enable(config: EnableConfig | None = None):
    global _ENABLED
    with _ENABLE_LOCK:
        torchax_mod = require_torchax()
        if config and config.mode == "accuracy":
            torchax_mod.enable_accuracy_mode()
        elif config and config.mode == "performance":
            torchax_mod.enable_performance_mode()
        env = torchax_mod.enable_globally()
        if env is None:
            env = torchax_mod.default_env()
        if config and config.respect_torch_return_dtypes is not None:
            env.config.internal_respect_torch_return_dtypes = config.respect_torch_return_dtypes
        _ENABLED = True
        return env


def disable():
    global _ENABLED
    with _ENABLE_LOCK:
        if not _ENABLED:
            return
        torchax_mod = require_torchax()
        torchax_mod.disable_globally()
        _ENABLED = False


def is_enabled() -> bool:
    return _ENABLED


def compile(
    program: BlockSamplingProgram,
    schedule: SamplingSchedule,
    *,
    nodes_to_sample: Sequence[Block] | None = None,
    clamp_state=None,
    seed=0,
    jit: bool = True,
    block_until_ready: bool = True,
    flatten_single: bool = True,
    name: str | None = None,
) -> ThrmlSampler:
    config = CompileConfig(
        nodes_to_sample=nodes_to_sample,
        clamp_state=clamp_state,
        seed=seed,
        jit=jit,
        block_until_ready=block_until_ready,
        flatten_single=flatten_single,
        name=name,
    )
    return compile_sampler(program, schedule, config)
