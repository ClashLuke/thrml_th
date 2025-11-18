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

def build_program(length: int) -> tuple[IsingSamplingProgram, SamplingSchedule, list]:

    nodes = [SpinNode() for _ in range(length)]
    edges = [(nodes[i], nodes[i + 1]) for i in range(length - 1)]
    biases = jnp.zeros((length,), dtype=jnp.float32)
    weights = jnp.ones((len(edges),), dtype=jnp.float32) * 0.35
    beta = jnp.array(1.0, dtype=jnp.float32)
    model = IsingEBM(nodes, edges, biases, weights, beta)
    program = IsingSamplingProgram(model, free_blocks=[Block(nodes)], clamped_blocks=[])
    schedule = SamplingSchedule(n_warmup=8, n_samples=64, steps_per_sample=2)
    init_state = hinton_init(jax.random.key(0), model, [Block(nodes)], ())
    return program, schedule, init_state

def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description="Generate labeled Ising states with thrml.th and train a classifier."
    )
    parser.add_argument("--length", type=int, default=32, help="Ising chain length")
    parser.add_argument("--dataset-size", type=int, default=512, help="Total samples to draw")
    parser.add_argument("--batch-samples", type=int, default=64, help="Samples per sampler call")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for DataLoader")
    parser.add_argument("--lr", type=float, default=3e-3, help="Optimizer learning rate")
    parser.add_argument("--seed", type=int, default=0, help="Seed for sampler + dataset")
    parser.add_argument(
        "--suppress-warnings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Suppress torchax and numpy warnings",
    )
    return parser
def generate_dataset(
    sampler: thrml_th.ThrmlSampler,
    init_state_torch,
    total_samples: int,
    batch_samples: int,
    key_stream: thrml_th.KeyStream,
) -> torch.Tensor:
    batches: list[torch.Tensor] = []
    while sum(batch.shape[0] for batch in batches) < total_samples:
        draws = sampler(init_state_torch, key=key_stream())
        batches.append(to_host_torch(draws).to(torch.float32))
    stacked = torch.cat(batches, dim=0)
    return stacked[: total_samples]

def build_dataset(samples: torch.Tensor) -> torch.utils.data.TensorDataset:

    # Label each sample by the sign of its magnetization.
    magnetization = samples.mean(dim=1, keepdim=True)
    labels = (magnetization > 0).float()
    return torch.utils.data.TensorDataset(samples, labels)

def run(args: argparse.Namespace) -> None:

    configure_example_runtime(args.suppress_warnings)
    thrml_th.enable()

    program, schedule, init_state = build_program(args.length)
    schedule = SamplingSchedule(
        n_warmup=schedule.n_warmup, n_samples=args.batch_samples, steps_per_sample=schedule.steps_per_sample
    )
    sampler = thrml_th.compile(
        program, schedule, nodes_to_sample=program.gibbs_spec.free_blocks, seed=args.seed
    )
    init_state_torch = thrml_th.torch_view(tuple(init_state))

    key_stream = thrml_th.KeyStream(args.seed)
    samples = generate_dataset(
        sampler, init_state_torch, args.dataset_size, args.batch_samples, key_stream
    )
    dataset = build_dataset(samples)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = torch.nn.Sequential(
        torch.nn.Linear(args.length, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        running_loss = 0.0
        for batch_inputs, batch_labels in dataloader:
            logits = model(batch_inputs)
            loss = loss_fn(logits, batch_labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(dataloader))
        print(f"epoch {epoch + 1:02d} | loss = {avg_loss:.4f}")

    with torch.no_grad():
        logits = model(samples)
        preds = torch.sigmoid(logits).round()
        accuracy = (preds == dataset.tensors[1]).float().mean().item()
    print(f"Final accuracy on generated dataset: {accuracy * 100:.2f}%")


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point for the pytorch_training_loop example."""

    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
