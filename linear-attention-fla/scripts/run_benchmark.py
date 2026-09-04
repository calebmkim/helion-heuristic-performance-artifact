"""Co-measure explicit default, seed, AOT, and FLA arms per workload cell."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from datetime import timezone
from functools import cache
import gzip
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import traceback
from typing import Any

from benchmark_paths import add_checkout
from benchmark_paths import checkout_path

HELION_ARMS = ("default", "seed", "aot_tuned")
ALL_ARMS = (*HELION_ARMS, "fla_triton")
torch: Any = None


def _rows(path: Path, mode: str) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows if mode == "all" else [row for row in rows if row["mode"] == mode]


def _read(path: Path) -> Any:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return json.load(handle)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt") as handle:
        json.dump(value, handle, indent=2, default=str)


def _set_replay_environment() -> None:
    os.environ["HELION_AUTOTUNE_EFFORT"] = "none"
    os.environ["HELION_SKIP_CACHE"] = "1"
    os.environ["HELION_DISABLE_AUTOTUNER_HEURISTICS"] = "1"
    os.environ.pop("HELION_AUTOTUNE_CACHE", None)
    os.environ.pop("HELION_AOT_MODE", None)


def _load_runtime(helion_root: Path | None, fla_root: Path | None) -> None:
    global torch
    add_checkout(fla_root)
    add_checkout(helion_root)
    import importlib

    torch = importlib.import_module("torch")


def _git_revision(module_file: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(Path(module_file).parent), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


@cache
def _environment_metadata() -> dict[str, object]:
    import fla
    import helion
    import triton

    result: dict[str, object] = {
        "helion_file": helion.__file__,
        "fla_file": fla.__file__,
        "fla_version": fla.__version__,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": torch.cuda.get_device_capability(),
    }
    if revision := _git_revision(helion.__file__):
        result["helion_revision"] = revision
    if revision := _git_revision(fla.__file__):
        result["fla_revision"] = revision
    return result


def _bench_all(
    functions: dict[str, Any],
    gradients: dict[str, list[Any]],
    rounds: int,
    row_index: int,
) -> tuple[dict[str, float], dict[str, list[float]], int]:
    """Cold-L2 CUDA-event timing, interleaved at every retained sample.

    Helion's generic interleaved helper uses a fixed arm order and records its
    start event before calling the arm, so it cannot clear backward gradients
    outside the timed region. This adapter changes only those semantics.
    """
    from triton import runtime

    labels = list(functions)
    if not labels:
        return {}, {}, 0
    driver = runtime.driver.active
    device = driver.get_device_interface()
    cache = driver.get_empty_cache_for_benchmark()

    def clear_grads(label: str) -> None:
        for leaf in gradients[label]:
            leaf.grad = None

    # Every arm is already compiled by the correctness pass. Warm its ordinary
    # dispatch path once before sizing the shared sample count.
    for label in labels:
        clear_grads(label)
        functions[label]()
    device.synchronize()

    probes = {label: [] for label in labels}
    for probe_index in range(3):
        offset = (row_index + probe_index) % len(labels)
        order = labels[offset:] + labels[:offset]
        for label in order:
            clear_grads(label)
            driver.clear_cache(cache)
            start = device.Event(enable_timing=True)
            end = device.Event(enable_timing=True)
            start.record()
            functions[label]()
            end.record()
            probes[label].append((start, end))
    device.synchronize()
    probe_ms = {
        label: statistics.median(
            start.elapsed_time(end) for start, end in events
        )
        for label, events in probes.items()
    }
    if any(value <= 0 for value in probe_ms.values()):
        raise RuntimeError(f"invalid interleaved timing probe: {probe_ms}")

    # Match the harness's roughly 100 ms timing target for the fastest arm,
    # while bounding a round when another arm is much slower.
    repeat = max(10, min(1000, int(100.0 / min(probe_ms.values()))))
    repeat = min(repeat, max(10, int(1000.0 / sum(probe_ms.values()))))
    repeat -= repeat % 2

    samples = {label: [] for label in labels}
    for round_index in range(rounds):
        events = {label: [] for label in labels}
        for sample_index in range(repeat):
            offset = (
                row_index + round_index + sample_index // 2
            ) % len(labels)
            rotated = labels[offset:] + labels[:offset]
            order = (
                rotated
                if sample_index % 2 == 0
                else list(reversed(rotated))
            )
            for label in order:
                clear_grads(label)
                driver.clear_cache(cache)
                start = device.Event(enable_timing=True)
                end = device.Event(enable_timing=True)
                start.record()
                functions[label]()
                end.record()
                events[label].append((start, end))
        device.synchronize()
        for label in labels:
            samples[label].append(
                statistics.mean(
                    start.elapsed_time(end) for start, end in events[label]
                )
            )
    return (
        {label: statistics.mean(values) for label, values in samples.items()},
        samples,
        repeat,
    )


def _reference_gradients(workload: Any) -> list[Any]:
    from examples.linear.linear_attention_harness import _grad_leaves

    inputs, leaves = _grad_leaves(workload.harness, workload.inputs)
    workload.harness.chunked_reference(inputs, 64).backward(workload.grad_out)
    torch.cuda.synchronize()
    return [leaf.grad.detach().clone() for leaf in leaves]


def _reported_calls(plan: dict[str, object]) -> list[dict[str, object]]:
    """Keep config evidence in results; full dispatch keys live in the manifest."""
    return [
        {key: value for key, value in call.items() if key != "call_key"}
        for call in plan["calls"]  # type: ignore[index]
    ]


def _cell(
    row: dict[str, str],
    arm_plans: dict[str, dict[str, object]],
    rounds: int,
    row_index: int,
) -> dict[str, object]:
    from examples.linear.linear_attention_harness import ACC_BWD_TOL
    from examples.linear.linear_attention_harness import ACC_FWD_TOL
    from examples.linear.linear_attention_utils import rel_error
    from replay import ConfigReplay
    from workload import build_workload
    from workload import shape_record

    workload = build_workload(row)
    replay = ConfigReplay(arm_plans)
    arm_errors: dict[str, str] = {}
    accuracy: dict[str, dict[str, object]] = {}

    fla_reference: Any = None
    fla_error: str | None = None
    try:
        workload.clear_fla_grads()
        fla_output = workload.fla()
        torch.cuda.synchronize()
        fla_reference = (
            workload.fla_gradient_values()
            if workload.is_backward
            else (
                fla_output
                if workload.varlen
                else fla_output.transpose(1, 2).contiguous()
            )
        )
    except Exception as error:
        fla_error = f"{type(error).__name__}: {error}"
        torch.cuda.empty_cache()

    if fla_reference is None and workload.is_backward:
        try:
            fla_reference = _reference_gradients(workload)
        except Exception as error:
            fla_error = (
                f"{fla_error}; reference fallback: {type(error).__name__}: {error}"
            )

    with replay.patched():
        for arm in HELION_ARMS:
            try:
                workload.clear_helion_grads()
                with replay.use(arm, audit=True):
                    output = workload.helion()
                torch.cuda.synchronize()
                actual = (
                    workload.helion_gradient_values()
                    if workload.is_backward
                    else output
                )
                if fla_reference is None:
                    raise RuntimeError("no correctness reference is available")
                if workload.is_backward:
                    errors = [
                        float(rel_error(value, expected))
                        for value, expected in zip(
                            actual, fla_reference, strict=True
                        )
                    ]
                    finite = all(
                        bool(torch.isfinite(value).all()) for value in actual
                    )
                    maximum = max(errors)
                    accuracy[arm] = {
                        "correct": finite and maximum < ACC_BWD_TOL,
                        "relative_gradient_errors": errors,
                    }
                else:
                    error = float(rel_error(actual, fla_reference))
                    finite = bool(torch.isfinite(actual).all())
                    accuracy[arm] = {
                        "correct": finite and error < ACC_FWD_TOL,
                        "relative_error": error,
                    }
            except Exception as error:
                arm_errors[arm] = f"{type(error).__name__}: {error}"
                accuracy[arm] = {
                    "correct": False,
                    "traceback": traceback.format_exc()[-3000:],
                }
                torch.cuda.empty_cache()

        functions: dict[str, Any] = {}
        gradients: dict[str, list[Any]] = {}
        for arm in HELION_ARMS:
            if arm in arm_errors:
                continue

            def helion_fn(active_arm: str = arm) -> Any:
                with replay.use(active_arm):
                    return workload.helion()

            functions[arm] = helion_fn
            gradients[arm] = workload.helion_grads
        if fla_error is None:
            functions["fla_triton"] = workload.fla
            gradients["fla_triton"] = workload.fla_grads

        timings, samples, repetitions = _bench_all(
            functions, gradients, rounds, row_index
        )

    fla_latency = timings.get("fla_triton")
    arms: dict[str, dict[str, object]] = {}
    for arm in HELION_ARMS:
        latency = timings.get(arm)
        if arm in arm_errors or latency is None:
            arms[arm] = {
                "status": "error",
                "correct": False,
                "error": arm_errors.get(arm, "arm was not timed"),
                "errors": accuracy.get(arm, {}),
            }
            continue
        record: dict[str, object] = {
            "status": "ok",
            "correct": accuracy[arm]["correct"],
            "latency_ms": latency,
            "samples_ms": samples[arm],
            "interleaved_repetitions_per_round": repetitions,
            "selection": {
                "source": "explicit config replay",
                "invoked_kernels": _reported_calls(arm_plans[arm]),
            },
            "errors": {
                key: value
                for key, value in accuracy[arm].items()
                if key != "correct"
            },
        }
        if fla_latency is not None:
            record["paired_fla_latency_ms"] = fla_latency
            record["paired_fla_samples_ms"] = samples["fla_triton"]
        arms[arm] = record

    if fla_latency is None:
        arms["fla_triton"] = {
            "status": "error",
            "correct": False,
            "error": fla_error or "FLA was not timed",
        }
    else:
        arms["fla_triton"] = {
            "status": "ok",
            "correct": True,
            "latency_ms": fla_latency,
            "samples_ms": samples["fla_triton"],
            "interleaved_repetitions_per_round": repetitions,
            "selection": {"measured_with": list(HELION_ARMS)},
        }

    return {
        "id": row["cell_key"],
        "variant": row["variant"],
        "mode": row["mode"],
        "shape": shape_record(row),
        "arms": arms,
    }


def _config_cells(path: Path) -> dict[str, dict[str, Any]]:
    value = _read(path)
    if (
        not isinstance(value, dict)
        or value.get("benchmark") != "linear-attention-config-replay"
        or not isinstance(value.get("cells"), list)
    ):
        raise ValueError(f"{path} is not a config replay manifest")
    return {cell["cell_key"]: cell for cell in value["cells"]}


def _one(args: argparse.Namespace) -> None:
    rows = _rows(args.manifest, args.mode)
    row = rows[args.row]
    try:
        _set_replay_environment()
        _load_runtime(args.helion_root, args.fla_root)
        config_cell = _config_cells(args.configs)[row["cell_key"]]
        missing = [arm for arm in HELION_ARMS if arm not in config_cell["arms"]]
        if missing:
            raise RuntimeError(f"missing replay plans: {', '.join(missing)}")
        cell = _cell(row, config_cell["arms"], args.rounds, args.row)
        result = {"cell": cell, "environment": _environment_metadata()}
    except Exception as error:
        from workload import shape_record

        message = f"{type(error).__name__}: {error}"
        cell = {
            "id": row["cell_key"],
            "variant": row["variant"],
            "mode": row["mode"],
            "shape": shape_record(row),
            "arms": {
                arm: {"status": "error", "correct": False, "error": message}
                for arm in ALL_ARMS
            },
        }
        result = {
            "cell": cell,
            "error": message,
            "traceback": traceback.format_exc()[-4000:],
        }
    _write(args.out, result)
    print(f"Wrote {row['cell_key']} to {args.out}", flush=True)


def _result(
    cells: list[dict[str, Any]],
    args: argparse.Namespace,
    environment: dict[str, object] | None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "benchmark": "linear-attention-fla",
        "cells": cells,
        "provenance": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "manifest": str(args.manifest),
            "config_replay_manifest": str(args.configs),
            "measurement": (
                "one fresh process per cell; default, seed, AOT, and FLA "
                "co-measured with sample-level cold-L2 interleaving"
            ),
            "outer_rounds": args.rounds,
            "environment": environment,
        },
    }


def _all(args: argparse.Namespace) -> None:
    rows = _rows(args.manifest, args.mode)
    prior: dict[str, dict[str, Any]] = {}
    environment = None
    if args.resume and args.out.exists():
        existing = _read(args.out)
        prior = {cell["id"]: cell for cell in existing.get("cells", [])}
        environment = existing.get("provenance", {}).get("environment")

    cell_dir = args.out.parent / "cell-results"
    cell_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows):
        key = row["cell_key"]
        if key in prior:
            print(f"[{index + 1}/{len(rows)}] cached {key}", flush=True)
            continue
        cell_output = cell_dir / f"{index:03d}.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--action",
            "one",
            "--manifest",
            str(args.manifest),
            "--configs",
            str(args.configs),
            "--mode",
            args.mode,
            "--row",
            str(index),
            "--rounds",
            str(args.rounds),
            "--out",
            str(cell_output),
        ]
        if args.helion_root:
            command.extend(["--helion-root", str(args.helion_root)])
        if args.fla_root:
            command.extend(["--fla-root", str(args.fla_root)])
        subprocess.run(command, check=True)
        payload = _read(cell_output)
        prior[key] = payload["cell"]
        if environment is None and isinstance(payload.get("environment"), dict):
            environment = payload["environment"]
        ordered = [prior[row["cell_key"]] for row in rows if row["cell_key"] in prior]
        _write(args.out, _result(ordered, args, environment))
        print(f"[{index + 1}/{len(rows)}] {key}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("one", "all"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--helion-root", type=Path)
    parser.add_argument("--fla-root", type=Path)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=("all", "forward", "forward_backward"),
        default="all",
    )
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be at least 1")
    args.manifest = args.manifest.expanduser().resolve()
    args.configs = args.configs.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    args.helion_root = checkout_path(args.helion_root, "HELION_ROOT")
    args.fla_root = checkout_path(args.fla_root, "FLA_ROOT")
    if args.action == "one":
        _one(args)
    else:
        _all(args)


if __name__ == "__main__":
    main()
