#!/usr/bin/env python3
"""Run default, heuristic-seed, and Torch-compile reduction arms together."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import traceback
from typing import Any
from typing import Callable


ARMS = ("default", "seed", "torch_compile")


@dataclass
class CapturedArm:
    name: str
    graph: Any
    replay: Callable[[], object]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, default=str)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _git_metadata(root: Path) -> dict[str, object]:
    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(root), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

    try:
        return {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "dirty": bool(git("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "dirty": None}


def _jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "value"):
        return _jsonable(getattr(value, "value"))
    return repr(value)


def _config(value: object | None) -> dict[str, object] | None:
    if value is None:
        return None
    config = getattr(value, "config", value)
    return _jsonable(dict(config))  # type: ignore[arg-type,return-value]


def _canonical_config(value: object) -> str:
    return json.dumps(_config(value), sort_keys=True, separators=(",", ":"))


def _expand_manifest(document: dict[str, Any]) -> list[dict[str, Any]]:
    groups = document["shape_groups"]
    cells: list[dict[str, Any]] = []
    for kernel, kernel_spec in document["kernels"].items():
        group_name = kernel_spec["shape_group"]
        group = groups[group_name]
        seen: set[tuple[int, ...]] = set()
        for shape_spec in group["shapes"]:
            shape = tuple(int(value) for value in shape_spec["shape"])
            if shape in seen:
                raise ValueError(f"duplicate shape for {kernel}: {shape}")
            seen.add(shape)
            encoded = "x".join(str(value) for value in shape)
            cells.append(
                {
                    "cell_key": f"{kernel}:{encoded}",
                    "kernel": kernel,
                    "body": kernel_spec["body"],
                    "dtype": kernel_spec["dtype"],
                    "shape_group": group_name,
                    "shape": list(shape),
                    "shape_convention": group["shape_convention"],
                    "provenance": {
                        key: value
                        for key, value in shape_spec.items()
                        if key != "shape"
                    },
                }
            )
    keys = [cell["cell_key"] for cell in cells]
    if len(keys) != len(set(keys)):
        raise ValueError("manifest produced duplicate cell keys")
    return cells


def _configure_environment(helion_root: Path) -> None:
    os.environ["HELION_ROOT"] = str(helion_root)
    os.environ["HELION_AUTOTUNE_EFFORT"] = "none"
    os.environ["HELION_SKIP_CACHE"] = "1"
    os.environ["HELION_DISABLE_AUTOTUNER_HEURISTICS"] = "0"
    root = str(helion_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    pretuned = str(helion_root / "pretuned_kernels")
    if pretuned not in sys.path:
        sys.path.insert(0, pretuned)


def _unique_bounds(kernel: object) -> list[Any]:
    bounds: list[Any] = []
    seen: set[int] = set()
    for bound in getattr(kernel, "_bound_kernels", {}).values():
        if id(bound) not in seen:
            seen.add(id(bound))
            bounds.append(bound)
    return bounds


def _selected_configs(kernel: object) -> list[dict[str, object]]:
    return [
        config
        for bound in _unique_bounds(kernel)
        if (config := _config(getattr(bound, "_config", None))) is not None
    ]


def _extract_configs(workload: Any) -> tuple[object, object, dict[str, object]]:
    kernel = workload.kernel_fn
    kernel.reset()
    bound = kernel.bind(workload.args)
    config_spec = bound.env.config_spec
    seeds = list(config_spec.compiler_seed_configs)
    fired = [str(value) for value in config_spec.autotuner_heuristics]
    if not seeds:
        raise RuntimeError(
            f"{workload.kernel} produced no compiler heuristic seed; fired={fired}"
        )
    with bound.env:
        default = config_spec._base_default_config()
    return default, seeds[0], {
        "default_source": "ConfigSpec._base_default_config",
        "seed_source": "first compiler seed with tuned caches and search disabled",
        "compiler_seed_count": len(seeds),
        "compiler_seed_configs": [_config(seed) for seed in seeds],
        "autotuner_heuristics": fired,
    }


def _fixed_kernel(helion: Any, original: Any, config: object) -> Any:
    settings = original.settings
    return helion.kernel(
        original.fn,
        config=config,
        static_shapes=settings.static_shapes,
        ignore_warnings=list(settings.ignore_warnings or []),
    )


def _accuracy(
    actual: dict[str, Any],
    expected: dict[str, Any],
    tolerances: dict[str, Any],
) -> dict[str, object]:
    import torch

    if set(actual) != set(expected):
        return {
            "pass": False,
            "error": (
                f"output names differ: actual={sorted(actual)}, "
                f"expected={sorted(expected)}"
            ),
        }
    passed = True
    outputs: dict[str, object] = {}
    for name, expected_value in expected.items():
        actual_value = actual[name]
        tolerance = tolerances[name]
        shape_matches = tuple(actual_value.shape) == tuple(expected_value.shape)
        dtype_matches = actual_value.dtype == expected_value.dtype
        if tolerance.exact:
            values_match = bool(torch.equal(actual_value, expected_value))
        else:
            values_match = bool(
                torch.allclose(
                    actual_value.to(torch.float32),
                    expected_value.to(torch.float32),
                    rtol=tolerance.rtol,
                    atol=tolerance.atol,
                    equal_nan=False,
                )
            )
        output_passed = shape_matches and dtype_matches and values_match
        outputs[name] = {
            "pass": output_passed,
            "shape": list(actual_value.shape),
            "expected_shape": list(expected_value.shape),
            "dtype": str(actual_value.dtype).removeprefix("torch."),
            "expected_dtype": str(expected_value.dtype).removeprefix("torch."),
            "dtype_matches": dtype_matches,
            "rtol": tolerance.rtol,
            "atol": tolerance.atol,
            "exact_required": tolerance.exact,
        }
        passed = passed and output_passed
    return {"pass": passed, "outputs": outputs}


def _prepare_helion_arm(
    name: str,
    workload: Any,
    config: object,
    expected: dict[str, Any],
    helion: Any,
    torch: Any,
) -> tuple[CapturedArm, dict[str, object]]:
    implementation = _fixed_kernel(helion, workload.kernel_fn, config)
    output = implementation(*workload.args)
    torch.cuda.synchronize()
    accuracy = _accuracy(
        workload.observe(output),
        expected,
        workload.tolerances,
    )
    if not accuracy["pass"]:
        raise RuntimeError(f"{name} failed correctness: {accuracy}")
    selected = _selected_configs(implementation)
    expected_config = _canonical_config(config)
    selected_matches = bool(selected) and {
        json.dumps(value, sort_keys=True, separators=(",", ":"))
        for value in selected
    } == {expected_config}
    if not selected_matches:
        raise RuntimeError(
            f"{name} explicit config did not match compiled config: {selected}"
        )
    call = lambda: implementation(*workload.args)
    captured = _capture(name, call, torch)
    return captured, {
        "status": "captured",
        "correctness": accuracy,
        "selection": {
            "backend": "helion_explicit_config",
            "config": _config(config),
            "source": (
                "base compiler default"
                if name == "default"
                else "compiler heuristic seed"
            ),
            "selected_matches_requested": selected_matches,
        },
        "compiled_configs": selected,
    }


def _prepare_torch_arm(
    workload: Any,
    expected: dict[str, Any],
    torch: Any,
    mode: str,
) -> tuple[CapturedArm, dict[str, object]]:
    torch._dynamo.reset()
    compiled = torch.compile(workload.reference_fn, mode=mode)
    output = compiled(*workload.args)
    torch.cuda.synchronize()
    accuracy = _accuracy(
        workload.observe(output),
        expected,
        workload.tolerances,
    )
    if not accuracy["pass"]:
        raise RuntimeError(f"torch_compile failed correctness: {accuracy}")
    call = lambda: compiled(*workload.args)
    captured = _capture("torch_compile", call, torch)
    return captured, {
        "status": "captured",
        "correctness": accuracy,
        "selection": {
            "backend": "torch_compile",
            "mode": mode,
            "hot_execution": True,
            "compilation_and_warmup_excluded": True,
        },
    }


def _capture(
    name: str,
    call: Callable[[], object],
    torch: Any,
) -> CapturedArm:
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    torch.cuda.synchronize()
    return CapturedArm(name, graph, graph.replay)


def _environment_metadata(
    helion_root: Path,
    torch: Any,
    torch_compile_mode: str,
) -> dict[str, object]:
    import helion
    import triton

    metadata: dict[str, object] = {
        "python": sys.version,
        "helion_file": helion.__file__,
        "helion_git": _git_metadata(helion_root),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch_compile_mode": torch_compile_mode,
    }
    try:
        metadata["tritonbench"] = importlib_metadata.version("tritonbench")
    except importlib_metadata.PackageNotFoundError:
        metadata["tritonbench"] = None
    return metadata


def _run_one(args: argparse.Namespace) -> dict[str, object]:
    manifest = _read_json(args.manifest)
    cells = _expand_manifest(manifest)
    cell = cells[args.cell_index]
    record: dict[str, object] = {
        "schema_version": 1,
        "run_fingerprint": args.fingerprint,
        "manifest_sha256": _sha256(args.manifest),
        "cell_index": args.cell_index,
        **cell,
        "requested_arms": list(ARMS),
        "status": "starting",
    }

    _configure_environment(args.helion_root)
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one CUDA device must be visible")
    import helion
    import helion.autotuner.aot_cache as aot_cache

    helion_path = Path(helion.__file__).resolve()
    if args.helion_root not in helion_path.parents:
        raise RuntimeError(
            f"imported Helion from {helion_path}, expected {args.helion_root}"
        )
    aot_cache.find_heuristic_file = lambda *_args, **_kwargs: None

    from workloads import build_workload
    from workloads import describe_inputs

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    workload = build_workload(cell["kernel"], tuple(cell["shape"]))
    if workload.body != cell["body"]:
        raise RuntimeError(
            f"manifest body {cell['body']} != workload body {workload.body}"
        )
    expected_output = workload.reference_fn(*workload.args)
    torch.cuda.synchronize()
    expected = workload.observe(expected_output)
    record.update(
        dtype=workload.dtype,
        input_tensors=describe_inputs(workload),
        optional_args=workload.optional_args,
        tolerances={
            name: {
                "rtol": tolerance.rtol,
                "atol": tolerance.atol,
                "exact": tolerance.exact,
            }
            for name, tolerance in workload.tolerances.items()
        },
        output_tensors={
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype).removeprefix("torch."),
            }
            for name, value in expected.items()
        },
    )

    default, seed, discovery = _extract_configs(workload)
    configs = {"default": default, "seed": seed}
    record["configs"] = {name: _config(config) for name, config in configs.items()}
    record["config_discovery"] = discovery

    prepared: list[CapturedArm] = []
    arm_results: dict[str, dict[str, object]] = {}
    for name in ("default", "seed"):
        try:
            captured, result = _prepare_helion_arm(
                name,
                workload,
                configs[name],
                expected,
                helion,
                torch,
            )
            prepared.append(captured)
            arm_results[name] = result
        except Exception as error:
            arm_results[name] = {
                "status": "error",
                "correctness": {"pass": False},
                "error": f"{type(error).__name__}: {error}",
                "trace": traceback.format_exc(),
            }

    try:
        captured, result = _prepare_torch_arm(
            workload,
            expected,
            torch,
            args.torch_compile_mode,
        )
        prepared.append(captured)
        arm_results["torch_compile"] = result
    except Exception as error:
        arm_results["torch_compile"] = {
            "status": "error",
            "correctness": {"pass": False},
            "error": f"{type(error).__name__}: {error}",
            "trace": traceback.format_exc(),
        }

    if prepared:
        import _bench

        samples_us = {arm.name: [] for arm in prepared}
        calls = [arm.replay for arm in prepared]
        for _round in range(args.rounds):
            timings_ms = _bench.bench_pre_captured_cudagraphs(
                calls,
                rep=args.repetitions,
            )
            for arm, timing_ms in zip(prepared, timings_ms, strict=True):
                samples_us[arm.name].append(float(timing_ms) * 1000.0)
        for arm in prepared:
            samples = samples_us[arm.name]
            arm_results[arm.name].update(
                status="ok",
                latency_us=statistics.median(samples),
                samples_us=samples,
                timing={
                    "method": (
                        "shared balanced CUDA-event timing of pre-captured "
                        "CUDA Graph replays"
                    ),
                    "rounds": args.rounds,
                    "repetitions_per_round": args.repetitions,
                    "cold_l2": True,
                    "cuda_graph": True,
                    "cpu_launch_overhead_included": False,
                    "all_available_arms_timed_together": True,
                    "ordering": "rotating positions followed by reversed rotations",
                },
            )

    record["arms"] = arm_results
    record["environment"] = _environment_metadata(
        args.helion_root,
        torch,
        args.torch_compile_mode,
    )
    record["status"] = (
        "ok"
        if all(arm_results.get(name, {}).get("status") == "ok" for name in ARMS)
        else "partial"
    )
    return record


def _one(args: argparse.Namespace) -> None:
    try:
        record = _run_one(args)
    except Exception as error:
        manifest = _read_json(args.manifest)
        cell = _expand_manifest(manifest)[args.cell_index]
        record = {
            "schema_version": 1,
            "run_fingerprint": args.fingerprint,
            "manifest_sha256": _sha256(args.manifest),
            "cell_index": args.cell_index,
            **cell,
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
            "trace": traceback.format_exc(),
        }
    _atomic_json(args.output, record)
    latencies = {
        name: arm.get("latency_us")
        for name, arm in record.get("arms", {}).items()
    }
    print(
        f"[{args.cell_index + 1:03d}] {record['kernel']} {record['shape']}: "
        f"{record['status']} {latencies}",
        flush=True,
    )


def _fingerprint(args: argparse.Namespace) -> str:
    script_dir = Path(__file__).resolve().parent
    value = {
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "workloads_sha256": _sha256(script_dir / "workloads.py"),
        "manifest_sha256": _sha256(args.manifest),
        "rounds": args.rounds,
        "repetitions": args.repetitions,
        "torch_compile_mode": args.torch_compile_mode,
        "helion_git": _git_metadata(args.helion_root),
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _all(args: argparse.Namespace) -> None:
    manifest = _read_json(args.manifest)
    cells = _expand_manifest(manifest)
    selected = list(enumerate(cells))
    if args.kernels:
        names = {name.strip() for name in args.kernels.split(",") if name.strip()}
        unknown = names - set(manifest["kernels"])
        if unknown:
            raise ValueError(f"unknown kernels: {sorted(unknown)}")
        selected = [(index, cell) for index, cell in selected if cell["kernel"] in names]
    if args.limit is not None:
        selected = selected[: args.limit]

    output = args.output.expanduser().resolve()
    parts = output.parent / f"{output.stem}.parts"
    parts.mkdir(parents=True, exist_ok=True)
    fingerprint = _fingerprint(args)
    records: list[dict[str, object]] = []
    env = os.environ.copy()
    env["HELION_ROOT"] = str(args.helion_root)
    env["PYTHONPATH"] = str(args.helion_root) + os.pathsep + env.get(
        "PYTHONPATH", ""
    )

    for ordinal, (index, cell) in enumerate(selected, 1):
        part = parts / f"{index:04d}.json"
        if args.resume and part.is_file():
            previous = _read_json(part)
            if previous.get("run_fingerprint") == fingerprint:
                records.append(previous)
                print(
                    f"[skip {ordinal:03d}/{len(selected):03d}] {cell['cell_key']}",
                    flush=True,
                )
                continue
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--action",
            "one",
            "--manifest",
            str(args.manifest),
            "--helion-root",
            str(args.helion_root),
            "--cell-index",
            str(index),
            "--rounds",
            str(args.rounds),
            "--repetitions",
            str(args.repetitions),
            "--torch-compile-mode",
            args.torch_compile_mode,
            "--fingerprint",
            fingerprint,
            "--output",
            str(part),
        ]
        completed = subprocess.run(command, check=False, env=env)
        if completed.returncode != 0 and not part.exists():
            _atomic_json(
                part,
                {
                    "schema_version": 1,
                    "run_fingerprint": fingerprint,
                    "cell_index": index,
                    **cell,
                    "status": "child_error",
                    "returncode": completed.returncode,
                },
            )
        records.append(_read_json(part))

    document = {
        "schema_version": 1,
        "profile": manifest["profile"],
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": _sha256(args.manifest),
        "run_fingerprint": fingerprint,
        "helion_git": _git_metadata(args.helion_root),
        "arms": list(ARMS),
        "torch_compile_mode": args.torch_compile_mode,
        "rounds": args.rounds,
        "repetitions_per_round": args.repetitions,
        "process_isolation": "one child process per cell; all arms together",
        "record_count": len(records),
        "records": records,
    }
    _atomic_json(output, document)
    good = sum(record.get("status") == "ok" for record in records)
    print(f"Wrote {good}/{len(records)} complete cells to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("all", "one"), default="all")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--helion-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument(
        "--torch-compile-mode",
        default="max-autotune-no-cudagraphs",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--kernels")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cell-index", type=int)
    parser.add_argument("--fingerprint", default="")
    args = parser.parse_args()

    args.manifest = args.manifest.expanduser().resolve()
    args.helion_root = args.helion_root.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.rounds <= 0 or args.repetitions <= 0:
        parser.error("--rounds and --repetitions must be positive")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("CUDA_VISIBLE_DEVICES must expose exactly one requested GPU")
    if args.action == "one":
        if args.cell_index is None:
            parser.error("--cell-index is required for --action one")
        _one(args)
    else:
        _all(args)


if __name__ == "__main__":
    main()
