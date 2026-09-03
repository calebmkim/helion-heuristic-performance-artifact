"""Small environment helpers shared by the benchmark starter scripts."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def checkout_path(value: Path | None, environment_name: str) -> Path | None:
    """Resolve an optional checkout from a CLI value or environment variable."""
    configured = value or (
        Path(os.environ[environment_name])
        if os.environ.get(environment_name)
        else None
    )
    return configured.expanduser().resolve() if configured else None


def add_checkout(path: Path | None) -> None:
    """Make a source checkout importable without requiring an editable install."""
    if path is None:
        return
    value = str(path)
    if value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)


def cuda_arch() -> str:
    """Return the active CUDA architecture in Helion's ``smXY`` spelling."""
    import torch

    major, minor = torch.cuda.get_device_capability()
    return f"sm{major}{minor}"


def linear_aot_module(explicit: str | None = None) -> str:
    """Return the checked-in AOT selector module for the active architecture."""
    configured = explicit or os.environ.get("HELION_LINEAR_AOT_MODULE")
    if configured:
        return configured
    return f"examples.linear._helion_aot_linear_attention_engine_cuda_{cuda_arch()}"
