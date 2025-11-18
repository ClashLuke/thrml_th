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

def build_program(length: int, beta: float, coupling: float) -> tuple[IsingSamplingProgram, list]:

    nodes = [SpinNode() for _ in range(length)]
    edges = [(nodes[i], nodes[i + 1]) for i in range(length - 1)]
    biases = jnp.zeros((length,), dtype=jnp.float32)
    weights = jnp.ones((len(edges),), dtype=jnp.float32) * coupling
    model = IsingEBM(nodes, edges, biases, weights, jnp.array(beta, dtype=jnp.float32))
    program = IsingSamplingProgram(model, free_blocks=[Block(nodes)], clamped_blocks=[])
    init_state = hinton_init(jax.random.key(0), model, [Block(nodes)], ())
    return program, init_state

def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(description="Estimate Ising statistics using thrml.th")
    parser.add_argument("--length", type=int, default=32, help="Number of spins in the chain")
    parser.add_argument("--beta", type=float, default=1.0, help="Inverse temperature")
    parser.add_argument("--coupling", type=float, default=0.35, help="Edge coupling magnitude")
    parser.add_argument("--seed", type=int, default=0, help="Seed for sampler + KeyStream")
    parser.add_argument(
        "--n-warmup", type=int, default=16, help="Number of warmup samples for the Gibbs chain"
    )
    parser.add_argument("--n-samples", type=int, default=64, help="Samples per chain")
    parser.add_argument("--steps-per-sample", type=int, default=2, help="Gibbs steps per draw")
    parser.add_argument(
        "--chains",
        type=int,
        default=3,
        help="Independent chains (calls to the sampler with fresh keys)",
    )
    parser.add_argument(
        "--show-samples",
        type=int,
        default=3,
        help="Number of raw samples to print for inspection",
    )
    parser.add_argument(
        "--suppress-warnings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Suppress torchax and numpy warnings",
    )
    return parser

def summarize(samples: torch.Tensor) -> dict[str, torch.Tensor | float]:

    overall_mag = samples.mean().item()
    per_site = samples.mean(dim=0)
    nn_corr = float("nan")
    if samples.shape[1] > 1:
        nn_corr = torch.mean(samples[:, :-1] * samples[:, 1:]).item()
    up_ratio = (samples > 0).float().mean().item()
    return {
        "overall_mag": overall_mag,
        "per_site": per_site,
        "nn_corr": nn_corr,
        "up_ratio": up_ratio,
    }

def run(args: argparse.Namespace) -> None:

    configure_example_runtime(args.suppress_warnings)
    thrml_th.enable()

    program, init_state = build_program(args.length, args.beta, args.coupling)
    schedule = SamplingSchedule(
        n_warmup=args.n_warmup, n_samples=args.n_samples, steps_per_sample=args.steps_per_sample
    )
    sampler = thrml_th.compile(
        program, schedule, nodes_to_sample=program.gibbs_spec.free_blocks, seed=args.seed
    )
    init_state_torch = thrml_th.torch_view(tuple(init_state))

    key_stream = thrml_th.KeyStream(args.seed)
    batches: list[torch.Tensor] = []
    for _ in range(args.chains):
        chain_samples = sampler(init_state_torch, key=key_stream())
        batches.append(to_host_torch(chain_samples).to(torch.float32))
    samples = torch.cat(batches, dim=0)

    stats = summarize(samples)
    total = samples.shape[0]
    print(f"Collected {total} samples over {args.chains} chains (spin count = {args.length}).")
    print(f"  overall magnetization: {stats['overall_mag']:.4f}")
    print(f"  up-spin ratio: {stats['up_ratio'] * 100:.2f}%")
    if args.length <= 12:
        print(f"  per-site magnetization: {stats['per_site'].tolist()}")
    else:
        head = stats["per_site"][:6].tolist()
        tail = stats["per_site"][-6:].tolist()
        print(f"  per-site magnetization (head/tail): {head} … {tail}")
    if args.length > 1:
        print(f"  nearest-neighbor correlation: {stats['nn_corr']:.4f}")

    if args.show_samples > 0:
        to_show = min(args.show_samples, total)
        print(f"\nFirst {to_show} samples:\n{samples[:to_show]}")

def main(argv: Sequence[str] | None = None) -> int:

    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
