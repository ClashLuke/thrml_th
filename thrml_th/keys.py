from __future__ import annotations

import time
from dataclasses import dataclass

import jax

from .config import SeedLike


def make_key(seed: SeedLike | None = None) -> jax.Array:
    if seed is None:
        seed = int(time.time_ns() & 0xFFFFFFFF)
    if isinstance(seed, jax.Array):
        return seed
    if isinstance(seed, int):
        return jax.random.key(seed)
    raise TypeError(f"Unsupported seed type: {type(seed)!r}")


@dataclass
class KeyStream:

    _key: jax.Array

    def __init__(self, seed: SeedLike | None = None):
        object.__setattr__(self, "_key", make_key(seed))

    def reset(self, seed: SeedLike | None = None) -> None:
        object.__setattr__(self, "_key", make_key(seed))

    def next(self) -> jax.Array:
        new_key, sample_key = jax.random.split(self._key)
        object.__setattr__(self, "_key", new_key)
        return sample_key

    __call__ = next

    @property
    def key(self) -> jax.Array:
        return self._key
