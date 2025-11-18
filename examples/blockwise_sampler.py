from __future__ import annotations

import argparse
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import torch

import thrml.th as thrml_th
from thrml import Block, SamplingSchedule, SpinNode
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

from .shared import configure_example_runtime, to_host_torch

def build_program(length: int, beta: float, coupling: float) -> tuple[IsingSamplingProgram, list, list[SpinNode]]:

    if length < 4:
        raise ValueError("length must be at least 4 to form multiple blocks")

    nodes = [SpinNode() for _ in range(length)]
    edges = [(nodes[i], nodes[i + 1]) for i in range(length - 1)]
    biases = jnp.zeros((length,), dtype=jnp.float32)
    weights = jnp.ones((len(edges),), dtype=jnp.float32) * coupling
    beta_arr = jnp.array(beta, dtype=jnp.float32)
    model = IsingEBM(nodes, edges, biases, weights, beta_arr)
    free_block = Block(nodes)
    program = IsingSamplingProgram(model, free_blocks=[free_block], clamped_blocks=[])
    init_state = hinton_init(jax.random.key(0), model, [free_block], ())
    return program, init_state, nodes

def make_partitions(nodes: Sequence[SpinNode]) -> dict[str, list[Block]]:

    mid = len(nodes) // 2
    contiguous = [Block(nodes[:mid]), Block(nodes[mid:])]
    if not contiguous[0].nodes or not contiguous[1].nodes:
        raise ValueError("contiguous partition produced an empty block; increase length")

    checker_even = Block(nodes[::2])
    checker_odd = Block(nodes[1::2])
    if not checker_even.nodes or not checker_odd.nodes:
        raise ValueError("checkerboard partition requires at least two nodes per parity")

    return {
        "contiguous halves": contiguous,
        "checkerboard": [checker_even, checker_odd],
    }

def summarize_block(samples: torch.Tensor) -> dict[str, float]:

    stats = {
        "mag": float(samples.mean().item()),
        "up_ratio": float((samples > 0).float().mean().item()),
        "size": samples.shape[1],
        "draws": samples.shape[0],
        "nn_corr": float("nan"),
    }
    if samples.shape[1] > 1:
        stats["nn_corr"] = float(torch.mean(samples[:, :-1] * samples[:, 1:]).item())
    return stats


def _to_tensor_list(tree) -> list[torch.Tensor]:
    if isinstance(tree, torch.Tensor):
        return [tree]
    return [leaf for leaf in tree]
def collect_samples(
    sampler: thrml_th.ThrmlSampler,
    init_state_torch,
    *,
    chains: int,
    seed: int,
) -> list[torch.Tensor]:
    if chains < 1:
        raise ValueError("chains must be >= 1")

    key_stream = thrml_th.KeyStream(seed)
    per_block: list[list[torch.Tensor]] | None = None
    for _ in range(chains):
        draws = sampler(init_state_torch, key=key_stream())
        tensor_tree = _to_tensor_list(to_host_torch(draws))
        tensor_tree = [tensor.to(torch.float32) for tensor in tensor_tree]
        if per_block is None:
            per_block = [[] for _ in range(len(tensor_tree))]
        for idx, tensor in enumerate(tensor_tree):
            per_block[idx].append(tensor)

    assert per_block is not None, "Sampler produced no blocks"
    return [torch.cat(block_samples, dim=0) for block_samples in per_block]

def display_partition(name: str, block_samples: list[torch.Tensor], show_samples: int) -> None:

    print(f"Partition: {name}")
    for idx, samples in enumerate(block_samples, start=1):
        stats = summarize_block(samples)
        print(
            f"  block {idx:02d} | spins = {stats['size']} | draws = {stats['draws']} | "
            f"mag = {stats['mag']:.4f} | up-ratio = {stats['up_ratio'] * 100:.1f}% | "
            f"nn-corr = {stats['nn_corr']:.4f}"
        )
        if show_samples > 0:
            preview = samples[:show_samples]
            print(f"    first {min(show_samples, preview.shape[0])} samples:\n{preview}")
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare contiguous and checkerboard block sampling with thrml.th"
    )
    parser.add_argument("--length", type=int, default=12, help="Number of spins in the chain")
    parser.add_argument("--beta", type=float, default=1.0, help="Inverse temperature")
    parser.add_argument("--coupling", type=float, default=0.3, help="Edge coupling magnitude")
    parser.add_argument("--n-warmup", type=int, default=8, help="Gibbs warmup iterations")
    parser.add_argument("--n-samples", type=int, default=16, help="Samples per chain call")
    parser.add_argument("--steps-per-sample", type=int, default=2, help="Gibbs steps per draw")
    parser.add_argument("--chains", type=int, default=2, help="Independent chains to average")
    parser.add_argument("--seed", type=int, default=0, help="Seed for KeyStreams")
    parser.add_argument(
        "--show-samples",
        type=int,
        default=1,
        help="Number of raw block samples to print for each partition",
    )
    parser.add_argument(
        "--suppress-warnings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Suppress torchax duplicate op warnings",
    )
    return parser

def run(args: argparse.Namespace) -> None:
    configure_example_runtime(args.suppress_warnings)
    thrml_th.enable()

    program, init_state, nodes = build_program(args.length, args.beta, args.coupling)
    partitions = make_partitions(nodes)

    schedule = SamplingSchedule(
        n_warmup=args.n_warmup,
        n_samples=args.n_samples,
        steps_per_sample=args.steps_per_sample,
    )

    sampler = thrml_th.compile(
        program,
        schedule,
        nodes_to_sample=partitions["contiguous halves"],
        flatten_single=False,
        seed=args.seed,
    )
    init_state_torch = thrml_th.torch_view(tuple(init_state))

    contiguous_samples = collect_samples(
        sampler,
        init_state_torch,
        chains=args.chains,
        seed=args.seed,
    )

    checker_sampler = sampler.with_options(
        nodes_to_sample=partitions["checkerboard"],
        flatten_single=False,
        seed=args.seed + 1,
    )
    checker_samples = collect_samples(
        checker_sampler,
        init_state_torch,
        chains=args.chains,
        seed=args.seed + 1,
    )

    display_partition("contiguous halves", contiguous_samples, args.show_samples)
    print()
    display_partition("checkerboard", checker_samples, args.show_samples)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
