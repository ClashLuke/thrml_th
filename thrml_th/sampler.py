from __future__ import annotations

from collections import OrderedDict
from typing import Any, Sequence

import equinox as eqx
import jax
import torch
from jax import tree_util as jtu

from thrml.block_management import Block
from thrml.block_sampling import BlockSamplingProgram, SamplingSchedule, sample_states

from .config import CompileConfig
from .interop import jax_view, torch_view
from .keys import KeyStream, make_key


def _default_nodes(program: BlockSamplingProgram) -> list[Block]:
    grouped: "OrderedDict[type[Block], list]" = OrderedDict()
    for block in program.gibbs_spec.free_blocks:
        key = block.node_type
        grouped.setdefault(key, [])
        grouped[key].extend(block.nodes)
    if not grouped:
        raise ValueError("Program does not expose any free blocks; pass nodes_to_sample explicitly.")
    return [Block(nodes) for nodes in grouped.values()]


def _build_sampler(
    program: BlockSamplingProgram,
    schedule: SamplingSchedule,
    nodes_to_sample: Sequence[Block],
    jit: bool,
):
    nodes_tuple = tuple(nodes_to_sample)

    def _sampler(key, init_state, clamp_state):
        init_list = list(init_state)
        clamp_list = list(clamp_state)
        return sample_states(
            key,
            program,
            schedule,
            init_list,
            clamp_list,
            list(nodes_tuple),
        )

    if jit:
        return eqx.filter_jit(_sampler)
    return _sampler


class ThrmlSampler(torch.nn.Module):
    """torch.nn.Module wrapper that forwards to a THRML sampler."""

    def __init__(self, program: BlockSamplingProgram, schedule: SamplingSchedule, config: CompileConfig):
        super().__init__()
        nodes = list(config.nodes_to_sample) if config.nodes_to_sample is not None else _default_nodes(program)
        if not nodes:
            raise ValueError("nodes_to_sample cannot be empty")
        self._nodes = tuple(nodes)
        clamp_state = list(config.clamp_state) if config.clamp_state is not None else []
        self._default_clamp = tuple(clamp_state)
        self._flatten_single = config.flatten_single
        self._block_until_ready = config.block_until_ready
        self._name = config.name or program.__class__.__name__
        self._jit = config.jit
        self._sampler = _build_sampler(program, schedule, self._nodes, jit=self._jit)
        self._key_stream = KeyStream(config.seed)
        self._program = program
        self._schedule = schedule

    @property
    def nodes(self) -> tuple[Block, ...]:
        return self._nodes

    @property
    def program(self) -> BlockSamplingProgram:
        return self._program

    @property
    def schedule(self) -> SamplingSchedule:
        return self._schedule

    def reset(self, seed=None) -> None:
        self._key_stream.reset(seed)

    def with_options(self, **overrides) -> "ThrmlSampler":
        config = CompileConfig(
            nodes_to_sample=overrides.get("nodes_to_sample", self._nodes),
            clamp_state=overrides.get("clamp_state", self._default_clamp),
            seed=overrides.get("seed", self._key_stream.key),
            jit=overrides.get("jit", self._jit),
            block_until_ready=overrides.get("block_until_ready", self._block_until_ready),
            flatten_single=overrides.get("flatten_single", self._flatten_single),
            name=overrides.get("name", self._name),
        )
        return ThrmlSampler(self._program, self._schedule, config)

    @property
    def jax_callable(self):
        """Return the underlying JAX callable for advanced integrations."""

        return self._sampler

    def _prepare_key(self, key) -> jax.Array:
        if key is None:
            return self._key_stream()
        if isinstance(key, int):
            return make_key(key)
        key_jax = jax_view(key)
        if isinstance(key_jax, jax.Array):
            return key_jax
        raise TypeError("key must be an int, JAX key, or torch tensor")

    @staticmethod
    def _ensure_list(tree, label: str):
        if isinstance(tree, list):
            return tree
        if isinstance(tree, tuple):
            return list(tree)
        raise TypeError(f"{label} must be a list or tuple")

    def forward(self, init_state, *, key=None, clamp_state=None, sync: bool | None = None):  # type: ignore[override]
        init_jax = self._ensure_list(jax_view(init_state), "init_state")
        clamp_source = clamp_state if clamp_state is not None else self._default_clamp
        clamp_jax = self._ensure_list(jax_view(clamp_source), "clamp_state")
        jax_key = self._prepare_key(key)
        results = self._sampler(jax_key, tuple(init_jax), tuple(clamp_jax))
        if sync if sync is not None else self._block_until_ready:
            results = jtu.tree_map(lambda leaf: jax.block_until_ready(leaf), results)
        flattened = results
        if self._flatten_single and isinstance(results, (list, tuple)) and len(results) == 1:
            flattened = results[0]
        return torch_view(flattened)


def compile_sampler(
    program: BlockSamplingProgram,
    schedule: SamplingSchedule,
    config: CompileConfig,
) -> ThrmlSampler:
    return ThrmlSampler(program, schedule, config)
