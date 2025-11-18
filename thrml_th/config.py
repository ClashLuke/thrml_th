from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence, TypeAlias

import jax

from thrml.block_management import Block

SeedLike: TypeAlias = int | jax.Array


@dataclass(frozen=True)
class EnableConfig:

    mode: Literal["auto", "performance", "accuracy"] = "auto"
    respect_torch_return_dtypes: bool | None = None


@dataclass(frozen=True)
class CompileConfig:

    nodes_to_sample: Sequence[Block] | None = None
    clamp_state: Sequence[Any] | None = None
    seed: SeedLike | None = 0
    jit: bool = True
    block_until_ready: bool = True
    flatten_single: bool = True
    name: str | None = None
