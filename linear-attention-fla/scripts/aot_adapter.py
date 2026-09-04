"""Narrow compatibility adapter for the pinned linear-attention AOT table."""

from __future__ import annotations

import functools
import importlib
from pathlib import Path
from typing import Any

from benchmark_paths import linear_aot_module


# Helion 6ae47cc9 fused q's L2 normalization into the varlen output kernel and
# updated its H100 AOT rows, but retained one zero-valued pipeline-stage entry
# for a range that no longer accepts that setting. Removing the stale [0] is
# semantic normalization: there is no corresponding staged range in the
# current kernel, and every other field in the checked-in AOT row is retained.
AOT_CONFIG_REPAIRS: dict[str, dict[str, object]] = {
    "chunk_fwd_o_diag_anchored_varlen_helion": {
        "field": "range_num_stages",
        "from": [0],
        "to": [],
        "reason": (
            "the pinned main AOT table retained a zero-valued entry for a "
            "range that no longer accepts range_num_stages"
        ),
    }
}


def repair_metadata() -> list[dict[str, object]]:
    return [
        {"kernel": kernel, **repair}
        for kernel, repair in AOT_CONFIG_REPAIRS.items()
    ]


def install_linear_aot_module(explicit: str | None = None) -> Any:
    """Load, narrowly repair, and register the pinned architecture AOT module."""
    module = importlib.import_module(linear_aot_module(explicit))
    for kernel_name, repair in AOT_CONFIG_REPAIRS.items():
        function_name = f"autotune_{kernel_name}"
        original = getattr(module, function_name)
        if getattr(original, "_artifact_aot_compat_repair", False):
            continue

        @functools.wraps(original)
        def repaired(
            *args: object,
            _original: Any = original,
            _kernel_name: str = kernel_name,
            _repair: dict[str, object] = repair,
        ) -> dict[str, object]:
            config = dict(_original(*args))
            field = str(_repair["field"])
            observed = config.get(field)
            expected = _repair["from"]
            if observed != expected:
                raise RuntimeError(
                    f"refusing unexpected AOT repair for {_kernel_name}: "
                    f"{field} is {observed!r}, expected {expected!r}"
                )
            replacement = _repair["to"]
            config[field] = (
                list(replacement)
                if isinstance(replacement, list)
                else replacement
            )
            return config

        repaired._artifact_aot_compat_repair = True  # type: ignore[attr-defined]
        setattr(module, function_name, repaired)

    from helion.autotuner.aot_cache import AOTAutotuneCache

    module_path = Path(module.__file__)
    AOTAutotuneCache._heuristic_modules[module_path] = module
    AOTAutotuneCache._heuristic_modules[module_path.resolve()] = module
    return module
