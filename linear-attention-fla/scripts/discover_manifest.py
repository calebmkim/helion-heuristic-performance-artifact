"""Discover the current linear-attention population from a Helion checkout."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
from pathlib import Path
from typing import Any

from benchmark_paths import add_checkout
from benchmark_paths import checkout_path

FIELDS = (
    "cell_key",
    "variant",
    "mode",
    "shape_name",
    "shape_json",
    "lengths_json",
    "heads",
    "dim",
)


def _cell(
    variant: str,
    mode: str,
    shape_name: str,
    *,
    shape: list[int] | None = None,
    lengths: list[int] | None = None,
    heads: int | None = None,
    dim: int | None = None,
) -> dict[str, object]:
    return {
        "cell_key": f"{variant}::{mode}::{shape_name}",
        "variant": variant,
        "mode": mode,
        "shape_name": shape_name,
        "shape_json": json.dumps(shape),
        "lengths_json": json.dumps(lengths),
        "heads": "" if heads is None else heads,
        "dim": "" if dim is None else dim,
    }


def discover(module: Any) -> list[dict[str, object]]:
    """Translate the current ``benchmarks.run_linattn`` registry into cells."""
    variants = [str(value) for value in module.VARIANTS]
    dense = {str(value) for value in module.DENSE_VARIANTS}
    varlen = {str(value) for value in module.VARLEN_VARIANTS}
    rows: list[dict[str, object]] = []

    for item in module.SHAPES:
        if len(item) != 5:
            raise ValueError(
                "SHAPES no longer contains (name, B, T, H, D) entries; "
                "adapt discover_manifest.py to the current registry"
            )
        name, batch, tokens, heads, dim = item
        shape = [int(batch), int(heads), int(tokens), int(dim), int(dim)]
        for variant in variants:
            if variant in varlen:
                continue
            rows.append(_cell(variant, "forward", str(name), shape=shape))
            if variant in dense:
                rows.append(
                    _cell(variant, "forward_backward", str(name), shape=shape)
                )

    for item in module.VARLEN_SHAPES:
        if len(item) != 4:
            raise ValueError(
                "VARLEN_SHAPES no longer contains (name, lengths, H, D) "
                "entries; adapt discover_manifest.py to the current registry"
            )
        name, lengths, heads, dim = item
        for variant in variants:
            if variant in varlen:
                rows.append(
                    _cell(
                        variant,
                        "forward",
                        str(name),
                        lengths=[int(value) for value in lengths],
                        heads=int(heads),
                        dim=int(dim),
                    )
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--helion-root",
        type=Path,
        help="Helion checkout (or set HELION_ROOT); omit for an installed package",
    )
    parser.add_argument(
        "--benchmark-module",
        default="benchmarks.run_linattn",
        help="module that owns the current linear-attention registry",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    add_checkout(checkout_path(args.helion_root, "HELION_ROOT"))
    module = importlib.import_module(args.benchmark_module)
    rows = discover(module)
    if not rows:
        raise SystemExit(f"no cells discovered from {args.benchmark_module}")

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    mode_names = {str(row["mode"]) for row in rows}
    modes = {
        mode: sum(row["mode"] == mode for row in rows) for mode in mode_names
    }
    print(f"Wrote {len(rows)} cells to {output}: {modes}")


if __name__ == "__main__":
    main()
