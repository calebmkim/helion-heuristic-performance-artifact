"""Materialize default, seed, and existing AOT configs for each workload cell."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from datetime import timezone
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any

from benchmark_paths import add_checkout
from benchmark_paths import checkout_path
from benchmark_paths import linear_aot_module

ARMS = ("default", "seed", "aot_tuned")


def _rows(path: Path, mode: str) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows if mode == "all" else [row for row in rows if row["mode"] == mode]


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str))


def _set_arm_environment(arm: str) -> None:
    os.environ["HELION_AUTOTUNE_EFFORT"] = "none"
    if arm == "aot_tuned":
        os.environ["HELION_AUTOTUNE_EFFORT"] = "full"
        os.environ["HELION_AUTOTUNE_CACHE"] = "AOTAutotuneCache"
        os.environ["HELION_AOT_MODE"] = "evaluate"
        os.environ.pop("HELION_SKIP_CACHE", None)
        os.environ["HELION_DISABLE_AUTOTUNER_HEURISTICS"] = "0"
        return
    os.environ["HELION_SKIP_CACHE"] = "1"
    os.environ.pop("HELION_AUTOTUNE_CACHE", None)
    os.environ.pop("HELION_AOT_MODE", None)
    os.environ["HELION_DISABLE_AUTOTUNER_HEURISTICS"] = (
        "1" if arm == "default" else "0"
    )


def _install_aot_module(explicit: str | None) -> None:
    module = importlib.import_module(linear_aot_module(explicit))
    from helion.autotuner.aot_cache import AOTAutotuneCache

    module_path = Path(module.__file__)
    AOTAutotuneCache._heuristic_modules[module_path] = module
    AOTAutotuneCache._heuristic_modules[module_path.resolve()] = module


def _configure_arm(arm: str, aot_module: str | None) -> None:
    if arm == "aot_tuned":
        _install_aot_module(aot_module)
        return
    import helion.autotuner.aot_cache as aot_cache

    aot_cache.find_heuristic_file = lambda *_args, **_kwargs: None
    if arm == "default":
        from helion.autotuner.config_spec import ConfigSpec

        ConfigSpec.default_config = ConfigSpec._base_default_config


def _child(args: argparse.Namespace) -> None:
    _set_arm_environment(args.arm)
    add_checkout(args.fla_root)
    add_checkout(args.helion_root)
    import torch

    from replay import ConfigRecorder
    from workload import build_workload

    _configure_arm(args.arm, args.aot_module)
    rows = _rows(args.manifest, args.mode)
    prior: dict[str, dict[str, Any]] = {}
    if args.resume and args.out.exists():
        prior = {
            record["cell_key"]: record
            for record in json.loads(args.out.read_text())
        }

    recorder = ConfigRecorder(args.arm)
    with recorder.patched():
        for index, row in enumerate(rows):
            key = row["cell_key"]
            if key in prior:
                print(f"[{index + 1}/{len(rows)}] cached {key}", flush=True)
                continue
            recorder.reset()
            try:
                workload = build_workload(row)
                workload.clear_helion_grads()
                workload.helion()
                torch.cuda.synchronize()
                record = {
                    "cell_key": key,
                    "variant": row["variant"],
                    "mode": row["mode"],
                    "shape_name": row["shape_name"],
                    "arm": args.arm,
                    "plan": recorder.plan(),
                }
            except Exception as error:
                record = {
                    "cell_key": key,
                    "variant": row["variant"],
                    "mode": row["mode"],
                    "shape_name": row["shape_name"],
                    "arm": args.arm,
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc()[-4000:],
                }
            prior[key] = record
            ordered = [prior[row["cell_key"]] for row in rows if row["cell_key"] in prior]
            _write(args.out, ordered)
            print(f"[{index + 1}/{len(rows)}] {key}", flush=True)


def _parent(args: argparse.Namespace) -> None:
    work_dir = args.work_dir or args.out.parent / "config-discovery"
    work_dir.mkdir(parents=True, exist_ok=True)
    arm_outputs: dict[str, Path] = {}
    for arm in ARMS:
        output = work_dir / f"{arm}.json"
        arm_outputs[arm] = output
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--action",
            "child",
            "--manifest",
            str(args.manifest),
            "--mode",
            args.mode,
            "--arm",
            arm,
            "--out",
            str(output),
        ]
        if args.helion_root:
            command.extend(["--helion-root", str(args.helion_root)])
        if args.fla_root:
            command.extend(["--fla-root", str(args.fla_root)])
        if args.aot_module:
            command.extend(["--aot-module", args.aot_module])
        if args.resume:
            command.append("--resume")
        subprocess.run(command, check=True)

    by_arm: dict[str, dict[str, dict[str, Any]]] = {}
    for arm, path in arm_outputs.items():
        records = json.loads(path.read_text())
        by_arm[arm] = {record["cell_key"]: record for record in records}

    cells = []
    errors = []
    for row in _rows(args.manifest, args.mode):
        key = row["cell_key"]
        plans = {}
        for arm in ARMS:
            record = by_arm[arm].get(key)
            if record is None or "error" in record:
                errors.append(
                    f"{key} {arm}: "
                    + ("missing" if record is None else str(record["error"]))
                )
                continue
            plans[arm] = record["plan"]
        cells.append(
            {
                "cell_key": key,
                "variant": row["variant"],
                "mode": row["mode"],
                "shape_name": row["shape_name"],
                "arms": plans,
            }
        )

    result = {
        "schema_version": 1,
        "benchmark": "linear-attention-config-replay",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "discovery_outputs": {
            arm: str(path) for arm, path in arm_outputs.items()
        },
        "cells": cells,
        "errors": errors,
    }
    _write(args.out, result)
    if errors:
        raise SystemExit(
            f"materialization had {len(errors)} failed arm/cells; see {args.out}"
        )
    print(f"Wrote configs for {len(cells)} cells to {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("all", "child"), default="all")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--helion-root", type=Path)
    parser.add_argument("--fla-root", type=Path)
    parser.add_argument("--aot-module")
    parser.add_argument(
        "--mode",
        choices=("all", "forward", "forward_backward"),
        default="all",
    )
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.manifest = args.manifest.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    args.work_dir = args.work_dir.expanduser().resolve() if args.work_dir else None
    args.helion_root = checkout_path(args.helion_root, "HELION_ROOT")
    args.fla_root = checkout_path(args.fla_root, "FLA_ROOT")
    args.aot_module = args.aot_module or os.environ.get("HELION_LINEAR_AOT_MODULE")
    if args.action == "child":
        if args.arm is None:
            parser.error("--arm is required with --action child")
        _child(args)
    else:
        _parent(args)


if __name__ == "__main__":
    main()
