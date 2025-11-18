# `thrml.th`

`thrml.th` exposes THRML’s JAX-based samplers through a PyTorch-friendly API. The module supplies:

- `enable()` – registers the `jax` device type in PyTorch through `torchax`.
- `compile(program, schedule, …)` – turns a THRML sampling program into a `torch.nn.Module`.
- `torch_view` / `jax_view` – zero-copy conversions between JAX arrays and Torch tensors.
- RNG helpers (`make_key`, `KeyStream`) so repeated sampler calls stay deterministic when needed.

Everything necessary to use the package lives in this README and the example scripts.

## Install

```bash
pip install thrml-th
```

## Usage

```python
import jax
import thrml
import thrml.th as thrml_th
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

thrml_th.enable()

nodes = [thrml.SpinNode() for _ in range(4)]
program = IsingSamplingProgram(...)
schedule = thrml.SamplingSchedule(n_warmup=10, n_samples=16, steps_per_sample=2)

sampler = thrml_th.compile(program, schedule)
init_state = thrml_th.torch_view(tuple(hinton_init(
    jax.random.key(0), program.model, program.gibbs_spec.free_blocks, ()
)))

samples = sampler(init_state)
```

## Examples

- `examples/ising_sampler.py` – builds an Ising chain, compiles it with `thrml.th`, prints magnetization stats.
- `examples/pytorch_training_loop.py` – drops a compiled sampler into a regular PyTorch training loop.
- `examples/blockwise_sampler.py` – compares contiguous vs. checkerboard `nodes_to_sample` partitions.

Run any script with `python examples/<name>.py` after installing the package and `torchax` dependency.
