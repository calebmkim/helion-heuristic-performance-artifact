#!/usr/bin/env python3
"""Discover exact AOT-backed vLLM reduction cells from a Helion checkout."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Callable


Key = tuple[int, ...]
Shape = tuple[int, ...]


@dataclass(frozen=True)
class KernelSpec:
    name: str
    key_fields: tuple[str, ...]
    to_bench_shape: Callable[[Key], Shape]
    to_aot_key: Callable[[Shape], Key]
    core_prefixes: frozenset[tuple[int, ...]]


KERNEL_SPECS = (
    KernelSpec(
        "dynamic_per_token_scaled_fp8_quant",
        ("hidden_size", "num_tokens"),
        lambda key: (key[1], key[0]),
        lambda shape: (shape[1], shape[0]),
        frozenset({(2048,), (4096,), (5120,)}),
    ),
    KernelSpec(
        "per_token_group_fp8_quant",
        ("hidden_size", "group_size", "num_tokens"),
        lambda key: (key[2], key[0]),
        lambda shape: (shape[1], 128, shape[0]),
        frozenset({(2048, 128), (4096, 128), (5120, 128)}),
    ),
    KernelSpec(
        "rms_norm_dynamic_per_token_quant",
        ("hidden_size", "num_tokens"),
        lambda key: key,
        lambda shape: shape,
        frozenset({(2048,), (4096,), (5120,)}),
    ),
    KernelSpec(
        "rms_norm_per_block_quant",
        ("hidden_size", "group_size", "num_tokens"),
        lambda key: key,
        lambda shape: shape,
        frozenset({(2048, 128), (4096, 128), (5120, 128)}),
    ),
    KernelSpec(
        "silu_and_mul_per_block_quant",
        ("intermediate_size", "group_size", "num_tokens"),
        lambda key: (key[2], key[0]),
        lambda shape: (shape[1], 128, shape[0]),
        frozenset({(6144, 128), (12288, 128), (25600, 128)}),
    ),
    KernelSpec(
        "fused_qk_norm_rope",
        ("q_heads", "kv_heads", "num_tokens"),
        lambda key: (key[2], key[0], key[1]),
        lambda shape: (shape[1], shape[2], shape[0]),
        frozenset({(16, 8), (32, 8), (64, 8)}),
    ),
)

CORE_TOKEN_COUNTS = frozenset({1, 128, 8192})


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_metadata(root: Path) -> dict[str, object]:
    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()

    try:
        return {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "dirty": bool(git("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "dirty": None}


def _compute_capability(requested: str | None) -> str:
    if requested:
        value = requested.lower()
        if value.startswith("sm"):
            return value
        return f"sm{value}"

    import torch

    major, minor = torch.cuda.get_device_capability()
    return f"sm{major}{minor}"


def _assignment(tree: ast.Module, name: str) -> object:
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"could not find literal assignment {name}")


def _exact_keys(path: Path) -> list[Key]:
    value = _assignment(ast.parse(path.read_text()), "_KEYS")
    if not isinstance(value, list):
        raise TypeError(f"{path}: _KEYS is not a list")
    keys = [tuple(int(item) for item in key) for key in value]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path}: duplicate AOT keys")
    return keys


def _bench_shapes(path: Path) -> list[Shape]:
    """Execute only the source module's self-contained `_bench_shapes` function."""
    tree = ast.parse(path.read_text())
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_bench_shapes"
        ),
        None,
    )
    if not isinstance(function, ast.FunctionDef):
        raise ValueError(f"{path}: no synchronous _bench_shapes() function")

    isolated = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(isolated)
    namespace: dict[str, object] = {}
    exec(compile(isolated, str(path), "exec"), namespace)  # noqa: S102
    values = namespace["_bench_shapes"]()  # type: ignore[operator]
    return [tuple(int(item) for item in shape) for shape in values]


def _cell(spec: KernelSpec, key: Key, index: int, aot_file: Path) -> dict[str, object]:
    shape = dict(zip(spec.key_fields, key, strict=True))
    label = ",".join(f"{field}={value}" for field, value in shape.items())
    return {
        "cell_key": f"{spec.name}::{label}",
        "kernel": spec.name,
        "aot_key": list(key),
        "aot_index": index,
        "shape": shape,
        "bench_shape": list(spec.to_bench_shape(key)),
        "aot_file": str(aot_file),
    }


def discover(
    helion_root: Path,
    compute: str,
    profile: str,
    selected_names: set[str] | None,
) -> dict[str, object]:
    pretuned = helion_root / "pretuned_kernels"
    cells: list[dict[str, object]] = []
    inventory: dict[str, object] = {}

    for spec in KERNEL_SPECS:
        if selected_names is not None and spec.name not in selected_names:
            continue

        kernel_dir = pretuned / spec.name
        source_file = kernel_dir / f"{spec.name}.py"
        aot_file = kernel_dir / f"_helion_aot_{spec.name}_cuda_{compute}.py"
        if not source_file.is_file():
            raise FileNotFoundError(f"missing kernel source: {source_file}")
        if not aot_file.is_file():
            raise FileNotFoundError(f"missing {compute} AOT selector: {aot_file}")

        exact_keys = _exact_keys(aot_file)
        exact_set = set(exact_keys)
        curated_shapes = _bench_shapes(source_file)
        curated_keys = [spec.to_aot_key(shape) for shape in curated_shapes]
        curated_exact = set(curated_keys) & exact_set
        curated_nonexact = [list(key) for key in curated_keys if key not in exact_set]
        core_exact = {
            key
            for key in exact_keys
            if key[:-1] in spec.core_prefixes and key[-1] in CORE_TOKEN_COUNTS
        }

        if profile == "all":
            chosen = exact_set
        elif profile == "core":
            chosen = core_exact
        else:
            chosen = curated_exact

        selected = [
            _cell(spec, key, index, aot_file)
            for index, key in enumerate(exact_keys)
            if key in chosen
        ]
        cells.extend(selected)
        inventory[spec.name] = {
            "source_file": str(source_file),
            "aot_file": str(aot_file),
            "aot_file_sha256": _sha256(aot_file),
            "key_fields": list(spec.key_fields),
            "exact_key_count": len(exact_keys),
            "exact_keys": [list(key) for key in exact_keys],
            "curated_shape_count": len(curated_shapes),
            "curated_exact_count": len(curated_exact),
            "curated_nonexact_keys": curated_nonexact,
            "core_exact_count": len(core_exact),
            "selected_count": len(selected),
        }

    if selected_names is not None:
        unknown = selected_names - {spec.name for spec in KERNEL_SPECS}
        if unknown:
            raise ValueError(f"unknown kernels: {sorted(unknown)}")
    if not cells:
        raise ValueError("selected profile contains no exact AOT cells")

    return {
        "schema_version": 1,
        "profile": profile,
        "compute_capability": compute,
        "helion_root": str(helion_root),
        "helion_git": _git_metadata(helion_root),
        "selected_cell_count": len(cells),
        "inventory": inventory,
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--helion-root", type=Path, required=True)
    parser.add_argument(
        "--compute",
        help="AOT suffix such as sm90; default derives it from the visible GPU",
    )
    parser.add_argument(
        "--profile",
        choices=("curated", "core", "all"),
        default="curated",
    )
    parser.add_argument(
        "--kernels",
        help="optional comma-separated kernel subset",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.helion_root.expanduser().resolve()
    names = (
        {value.strip() for value in args.kernels.split(",") if value.strip()}
        if args.kernels
        else None
    )
    manifest = discover(root, _compute_capability(args.compute), args.profile, names)

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")

    counts = {
        name: details["selected_count"]
        for name, details in manifest["inventory"].items()  # type: ignore[union-attr]
    }
    print(
        f"Wrote {manifest['selected_cell_count']} {manifest['profile']} cells "
        f"for {manifest['compute_capability']} to {output}: {counts}"
    )


if __name__ == "__main__":
    main()
