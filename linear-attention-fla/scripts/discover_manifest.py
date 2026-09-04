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


def _selection(value: str | None) -> set[str] | None:
    if value is None:
        return None
    selected = {item.strip() for item in value.split(",") if item.strip()}
    return selected or None


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
    parser.add_argument(
        "--variants",
        help="optional comma-separated variant names",
    )
    parser.add_argument(
        "--shape-names",
        help="optional comma-separated dense or variable-length shape names",
    )
    parser.add_argument(
        "--modes",
        help="optional comma-separated modes: forward, forward_backward",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="optional final row limit for a small compatibility run",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    add_checkout(checkout_path(args.helion_root, "HELION_ROOT"))
    module = importlib.import_module(args.benchmark_module)
    rows = discover(module)
    variants = _selection(args.variants)
    shape_names = _selection(args.shape_names)
    modes = _selection(args.modes)
    available_variants = {str(row["variant"]) for row in rows}
    available_shapes = {str(row["shape_name"]) for row in rows}
    available_modes = {str(row["mode"]) for row in rows}
    for label, requested, available in (
        ("variants", variants, available_variants),
        ("shape names", shape_names, available_shapes),
        ("modes", modes, available_modes),
    ):
        unknown = sorted((requested or set()) - available)
        if unknown:
            parser.error(
                f"unknown {label}: {', '.join(unknown)}; "
                f"choose from {', '.join(sorted(available))}"
            )
    if variants is not None:
        rows = [row for row in rows if row["variant"] in variants]
    if shape_names is not None:
        rows = [row for row in rows if row["shape_name"] in shape_names]
    if modes is not None:
        rows = [row for row in rows if row["mode"] in modes]
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be at least 1")
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(
            f"no cells selected from {args.benchmark_module}; "
            "check the manifest filters"
        )

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
