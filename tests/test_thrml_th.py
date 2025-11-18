from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import thrml.th as thrml_th
from thrml import Block, SamplingSchedule, SpinNode, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init


@pytest.fixture(scope="module")
def ising_problem():
    nodes = [SpinNode() for _ in range(4)]
    edges = [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]
    biases = jnp.zeros((len(nodes),), dtype=jnp.float32)
    weights = jnp.ones((len(edges),), dtype=jnp.float32) * 0.3
    beta = jnp.array(1.0, dtype=jnp.float32)
    model = IsingEBM(nodes, edges, biases, weights, beta)
    free_blocks = [Block(nodes)]
    program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])
    schedule = SamplingSchedule(n_warmup=4, n_samples=8, steps_per_sample=2)
    init_key = jax.random.key(0)
    init_state = hinton_init(init_key, model, free_blocks, ())
    return program, schedule, init_state, nodes


def test_enable_is_idempotent():
    thrml_th.disable()  # ensure clean slate
    env = thrml_th.enable()
    assert env is not None
    assert thrml_th.is_enabled()
    env2 = thrml_th.enable()
    assert env2 is not None
    thrml_th.disable()
    assert not thrml_th.is_enabled()


def test_compile_matches_thrml_reference(ising_problem):
    thrml_th.enable()
    program, schedule, init_state, nodes = ising_problem
    sampler = thrml_th.compile(program, schedule, nodes_to_sample=[Block(nodes)], seed=123, flatten_single=True)
    key = thrml_th.make_key(7)
    init_state_torch = thrml_th.torch_view(tuple(init_state))
    torch_output = sampler(init_state_torch, key=key)

    jax_output = thrml_th.jax_view(torch_output)
    expected = sample_states(key, program, schedule, list(init_state), [], [Block(nodes)])[0]
    assert jnp.all(jnp.equal(jax_output, expected))


def test_can_return_full_tree(ising_problem):
    thrml_th.enable()
    program, schedule, init_state, nodes = ising_problem
    sampler = thrml_th.compile(program, schedule, nodes_to_sample=[Block(nodes)], flatten_single=False)
    result = sampler(thrml_th.torch_view(tuple(init_state)), key=thrml_th.make_key(42))
    assert isinstance(result, (list, tuple))
    assert len(result) == 1
