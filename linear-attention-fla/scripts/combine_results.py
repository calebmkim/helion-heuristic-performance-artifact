"""Combine isolated benchmark-arm outputs into one analysis-friendly result."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from datetime import timezone
import gzip
import json
import math
from pathlib import Path
from typing import Any

ARM_ALIASES = {
    "default": "default",
    "raw": "default",
    "seed": "seed",
    "heuristic": "seed",
    "compiler": "seed",
    "aot_tuned": "aot_tuned",
    "pre_tuned": "aot_tuned",
    "tuned": "aot_tuned",
}
HELION_ARMS = ("default", "seed", "aot_tuned")


def _input(value: str) -> tuple[str, Path]:
    try:
        name, raw_path = value.split("=", 1)
        return ARM_ALIASES[name], Path(raw_path).expanduser().resolve()
    except (KeyError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "inputs must be ARM=PATH, where ARM is default, seed, or aot_tuned"
        ) from error


def _read_records(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text())
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"{path} must contain a cell record or list of records")
    return value


def _positive_number(value: object) -> float | None:
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and value > 0
    ):
        return float(value)
    return None


def _errors(record: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "relative_error_vs_fla",
        "relative_gradient_error_vs_reference",
        "reference_error",
        "relative_gradient_error_vs_fla",
        "fla_error",
    )
    return {
        field: record[field]
        for field in fields
        if record.get(field) not in (None, {}, [])
    }


def _helion_record(record: dict[str, Any] | None, arm: str) -> dict[str, Any]:
    if record is None:
        return {"status": "missing", "correct": False}
    if record.get("error"):
        return {
            "status": "error",
            "correct": False,
            "error": record["error"],
        }
    latency = _positive_number(record.get("helion_ms"))
    if latency is None:
        return {
            "status": "unavailable",
            "correct": False,
            "reason": "no reportable Helion latency",
        }
    samples = record.get("samples_ms")
    result: dict[str, Any] = {
        "status": "ok",
        "correct": record.get("accuracy_ok") is True,
        "latency_ms": latency,
        "samples_ms": samples.get("helion", []) if isinstance(samples, dict) else [],
        "selection": {
            "invoked_kernels": record.get("invoked_kernels", []),
            "label": record.get("label"),
        },
    }
    if record.get("aot_audit") is not None:
        result["selection"]["aot_audit"] = record["aot_audit"]
    if record.get("fla_error") is None:
        paired_fla = _positive_number(record.get("fla_ms"))
        if paired_fla is not None:
            result["paired_fla_latency_ms"] = paired_fla
            result["paired_fla_samples_ms"] = (
                samples.get("fla", []) if isinstance(samples, dict) else []
            )
    errors = _errors(record)
    if errors:
        result["errors"] = errors
    return result


def _fla_record(record: dict[str, Any] | None, source_arm: str) -> dict[str, Any]:
    if record is None:
        return {"status": "missing", "correct": False}
    if record.get("fla_error"):
        return {
            "status": "error",
            "correct": False,
            "error": record["fla_error"],
        }
    latency = _positive_number(record.get("fla_ms"))
    if latency is None:
        return {
            "status": "unavailable",
            "correct": False,
            "reason": "FLA was not timed for this cell",
        }
    samples = record.get("samples_ms")
    return {
        "status": "ok",
        "correct": True,
        "latency_ms": latency,
        "samples_ms": samples.get("fla", []) if isinstance(samples, dict) else [],
        "selection": {"measured_with": source_arm},
    }


def _shape(row: dict[str, str]) -> dict[str, Any]:
    name = row["shape_name"]
    shape = json.loads(row.get("shape_json") or "null")
    if isinstance(shape, list) and len(shape) == 5:
        batch, heads, tokens, dim, value_dim = shape
        return {
            "name": name,
            "B": batch,
            "H": heads,
            "T": tokens,
            "D": dim,
            "DV": value_dim,
        }
    lengths = json.loads(row.get("lengths_json") or "null")
    if isinstance(lengths, list):
        dim = int(row["dim"])
        return {
            "name": name,
            "B": len(lengths),
            "H": int(row["heads"]),
            "T": sum(lengths),
            "D": dim,
            "DV": dim,
            "lengths": lengths,
        }
    return {"name": name}


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt") as handle:
        json.dump(value, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--input",
        action="append",
        type=_input,
        default=[],
        metavar="ARM=PATH",
        help="raw runner output; repeat for forward and forward+backward files",
    )
    parser.add_argument(
        "--fla-from",
        choices=HELION_ARMS,
        default="seed",
        help="co-benchmarked arm whose FLA measurements should be retained",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    by_arm: dict[str, dict[str, dict[str, Any]]] = {
        arm: {} for arm in HELION_ARMS
    }
    input_paths: dict[str, list[str]] = {arm: [] for arm in HELION_ARMS}
    environments: dict[str, Any] = {}
    for arm, path in args.input:
        input_paths[arm].append(str(path))
        for record in _read_records(path):
            key = record.get("cell_key")
            if not isinstance(key, str):
                raise ValueError(f"record in {path} has no string cell_key")
            reported_arm = record.get("arm")
            if (
                isinstance(reported_arm, str)
                and ARM_ALIASES.get(reported_arm) != arm
            ):
                raise ValueError(
                    f"{path} reports arm {reported_arm!r}, not requested arm {arm!r}"
                )
            if key in by_arm[arm]:
                raise ValueError(f"duplicate {arm} result for {key}")
            by_arm[arm][key] = record
            if arm not in environments and isinstance(record.get("environment"), dict):
                environments[arm] = record["environment"]

    manifest = args.manifest.expanduser().resolve()
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    cells: list[dict[str, Any]] = []
    for row in rows:
        key = row["cell_key"]
        arms = {
            arm: _helion_record(by_arm[arm].get(key), arm) for arm in HELION_ARMS
        }
        arms["fla_triton"] = _fla_record(
            by_arm[args.fla_from].get(key), args.fla_from
        )
        cells.append(
            {
                "id": key,
                "variant": row["variant"],
                "mode": row["mode"],
                "shape": _shape(row),
                "arms": arms,
            }
        )

    result = {
        "schema_version": 1,
        "benchmark": "linear-attention-fla",
        "cells": cells,
        "provenance": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "manifest": str(manifest),
            "inputs": input_paths,
            "fla_from_arm": args.fla_from,
            "environments": environments,
        },
    }
    _write(args.output.expanduser().resolve(), result)
    print(f"Wrote {len(cells)} combined cells to {args.output}")


if __name__ == "__main__":
    main()
