from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import torch

import thrml.th as thrml_th


class _DuplicateOpFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Duplicate op registration" not in record.getMessage()


_DUPLICATE_FILTER = _DuplicateOpFilter()


def configure_example_runtime(suppress: bool = True) -> None:
    if not suppress:
        return

    # Suppress torchax duplicate op registration spam
    root_logger = logging.getLogger()
    if _DUPLICATE_FILTER not in root_logger.filters:
        root_logger.addFilter(_DUPLICATE_FILTER)

    # Suppress numpy and CUDA warnings when running on CPU-only hosts
    warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
    warnings.filterwarnings("ignore", message="CUDA initialization: Unexpected error")


def to_host_torch(array: Any) -> torch.Tensor | Any:
    if isinstance(array, (list, tuple)):
        converted = [to_host_torch(item) for item in array]
        return type(array)(converted)

    # Convert the thrml_th tensor view back to JAX, then to numpy, then to torch
    jax_array = thrml_th.jax_view(array)
    numpy_array = np.asarray(jax_array)
    return torch.from_numpy(numpy_array)
