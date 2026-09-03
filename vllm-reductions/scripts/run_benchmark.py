#!/usr/bin/env python3
"""Benchmark every available arm together in one isolated process per cell."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import hashlib
import importlib
from importlib import metadata as importlib_metadata
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import traceback
import types
from typing import Any
from typing import Callable

from vllm_extension import find_vllm_extension
from vllm_extension import load_vllm_extension


HELION_ARMS = ("default", "seed", "aot_tuned")
VLLM_ARM = "vllm_cuda"


@dataclass
class CapturedArm:
    name: str
    graph: Any
    replay: Callable[[], object]
    call: Callable[[], object]


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


def _git_metadata(root: Path | None) -> dict[str, object] | None:
    if root is None:
        return None

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


def _add_checkout(path: Path | None) -> None:
    if path is None:
        return
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def _configure_helion_environment() -> None:
    # AOT lookup is intentionally bypassed while discovering default and seed.
    # The exact AOT config is read from the manifest's selector and forced later.
    os.environ["HELION_AUTOTUNE_EFFORT"] = "none"
    os.environ["HELION_SKIP_CACHE"] = "1"
    os.environ["HELION_DISABLE_AUTOTUNER_HEURISTICS"] = "0"


def _stub_vllm_import() -> None:
    """Avoid importing vLLM's unrelated serving stack."""
    if "vllm" in sys.modules:
        return
    module = types.ModuleType("vllm")
    module.__file__ = None
    try:
        module.__version__ = importlib_metadata.version("vllm")
    except importlib_metadata.PackageNotFoundError:
        module.__version__ = None
    sys.modules["vllm"] = module


def _import_kernel_module(helion_root: Path, kernel: str) -> Any:
    pretuned = helion_root / "pretuned_kernels"
    _add_checkout(pretuned)
    source = pretuned / kernel / f"{kernel}.py"
    module_name = f"_vllm_reduction_artifact_{kernel}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


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


def _unique_bounds(kernel: object) -> list[Any]:
    bounds: list[Any] = []
    seen: set[int] = set()
    for bound in getattr(kernel, "_bound_kernels", {}).values():
        if id(bound) not in seen:
            seen.add(id(bound))
            bounds.append(bound)
    return bounds


def _kernel_metadata(kernel: object) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for bound in _unique_bounds(kernel):
        spec = getattr(bound, "config_spec", None)
        if spec is None:
            spec = getattr(getattr(bound, "env", None), "config_spec", None)
        records.append(
            {
                "selected_config": _config(getattr(bound, "_config", None)),
                "compiler_default_config": _config(
                    getattr(spec, "compiler_default_config", None)
                ),
                "compiler_seed_configs": [
                    _config(config)
                    for config in getattr(spec, "compiler_seed_configs", ())
                ],
                "autotuner_heuristics": [
                    str(value)
                    for value in getattr(spec, "autotuner_heuristics", ())
                ],
            }
        )
    return records


def _capture_module_call(
    module: Any,
    bench_shape: tuple[int, ...],
    bench_module: Any,
) -> Callable[[], object]:
    captured: dict[str, object] = {}

    def capture(
        shapes: object,
        make_calls: object,
        **_kwargs: object,
    ) -> dict[str, object]:
        values = list(shapes)  # type: ignore[arg-type]
        if values != [bench_shape]:
            raise RuntimeError(
                f"module ignored patched shape: expected {[bench_shape]}, got {values}"
            )
        call, _baselines, _shape_cells = make_calls(bench_shape)  # type: ignore[operator]
        captured["call"] = call
        return {
            "helion_wins": 0,
            "total": 0,
            "geomean": 0.0,
            "best_speedup": 0.0,
            "baselines": {},
        }

    module._bench_shapes = lambda: [bench_shape]
    original_sweep = bench_module.run_sweep
    original_baselines = getattr(module, "_baselines", None)
    bench_module.run_sweep = capture
    if original_baselines is not None:
        module._baselines = lambda: []
    try:
        module.main(verbose=False)
    finally:
        bench_module.run_sweep = original_sweep
        if original_baselines is not None:
            module._baselines = original_baselines
    call = captured.get("call")
    if not callable(call):
        raise RuntimeError("module main() did not provide a benchmark call")
    return call


def _discover_default_and_seed(
    module: Any,
    kernel_name: str,
    bench_shape: tuple[int, ...],
    bench_module: Any,
    torch: Any,
) -> tuple[object, object, dict[str, object]]:
    kernel = getattr(module, kernel_name)
    call = _capture_module_call(module, bench_shape, bench_module)
    call()
    torch.cuda.synchronize()

    configured = [
        bound for bound in _unique_bounds(kernel) if getattr(bound, "_config", None)
    ]
    if not configured:
        raise RuntimeError("seed discovery produced no configured BoundKernel")
    selected = [bound._config for bound in configured]
    if len({_canonical_config(config) for config in selected}) != 1:
        raise RuntimeError("one cell selected multiple different seed configs")
    seed = selected[0]

    defaults = []
    for bound in configured:
        with bound.env:
            defaults.append(bound.config_spec._base_default_config())
    if len({_canonical_config(config) for config in defaults}) != 1:
        raise RuntimeError("one cell produced multiple different base defaults")
    default = defaults[0]
    return default, seed, {
        "default_source": "ConfigSpec._base_default_config",
        "seed_source": "selected config with compiler heuristics enabled and caches disabled",
        "kernel_records": _kernel_metadata(kernel),
    }


def _exact_aot_config(cell: dict[str, Any]) -> dict[str, object]:
    path = Path(cell["aot_file"])
    tree = ast.parse(path.read_text(), filename=str(path))
    index = int(cell["aot_index"])

    keys: object | None = None
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_KEYS"
            for target in statement.targets
        ):
            keys = ast.literal_eval(statement.value)
            break
    if not isinstance(keys, list) or index >= len(keys):
        raise RuntimeError(f"{path}: invalid _KEYS entry {index}")
    if tuple(keys[index]) != tuple(cell["aot_key"]):
        raise RuntimeError(
            f"{path}: manifest key {cell['aot_key']} does not match _KEYS[{index}]"
        )

    function = next(
        (
            statement
            for statement in tree.body
            if isinstance(statement, ast.FunctionDef)
            and statement.name == f"autotune_{cell['kernel']}"
        ),
        None,
    )
    if function is None:
        raise RuntimeError(f"{path}: missing autotune_{cell['kernel']}")
    for statement in function.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_C"
            for target in statement.targets
        ):
            configs = ast.literal_eval(statement.value)
            if isinstance(configs, list) and index < len(configs):
                config = configs[index]
                if isinstance(config, dict):
                    return _jsonable(config)  # type: ignore[return-value]
    raise RuntimeError(f"{path}: could not read exact AOT config {index}")


def _fixed_kernel(helion: Any, original: Any, config: object) -> Any:
    settings = original.settings
    return helion.kernel(
        original.fn,
        config=config,
        static_shapes=settings.static_shapes,
        ignore_warnings=list(settings.ignore_warnings or []),
    )


def _prepare_arm(
    name: str,
    module: Any,
    kernel_name: str,
    implementation: object,
    bench_shape: tuple[int, ...],
    bench_module: Any,
    torch: Any,
    selection: dict[str, object],
    expected_config: dict[str, object] | None = None,
) -> tuple[CapturedArm, dict[str, object]]:
    setattr(module, kernel_name, implementation)
    module._bench_shapes = lambda: [bench_shape]
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    module.correctness_check()
    torch.cuda.synchronize()

    records = [] if name == VLLM_ARM else _kernel_metadata(implementation)
    if expected_config is not None:
        selected = [
            record["selected_config"]
            for record in records
            if record["selected_config"] is not None
        ]
        expected = json.dumps(expected_config, sort_keys=True, separators=(",", ":"))
        actual = {
            json.dumps(config, sort_keys=True, separators=(",", ":"))
            for config in selected
        }
        selection["selected_matches_manifest"] = bool(selected) and actual == {expected}
        if not selection["selected_matches_manifest"]:
            raise RuntimeError(
                "forced AOT config did not match the manifest selector entry"
            )

    call = _capture_module_call(module, bench_shape, bench_module)
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    torch.cuda.synchronize()

    result: dict[str, object] = {
        "status": "captured",
        "correctness": {"pass": True},
        "selection": selection,
        "kernel_records": records,
    }
    return CapturedArm(name, graph, graph.replay, call), result


def _environment_metadata(
    helion_root: Path,
    vllm_root: Path | None,
    torch: Any,
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
        "vllm_git": _git_metadata(vllm_root),
    }
    try:
        distribution = importlib_metadata.distribution("vllm")
        metadata.update(
            vllm_version=distribution.version,
            vllm_distribution_path=str(distribution.locate_file("")),
        )
    except importlib_metadata.PackageNotFoundError:
        metadata["vllm_version"] = None
    return metadata


def _run_one(args: argparse.Namespace) -> dict[str, object]:
    manifest = _read_json(args.manifest)
    cell = manifest["cells"][args.cell_index]
    requested_arms = [*HELION_ARMS]
    if args.include_vllm_cuda:
        requested_arms.append(VLLM_ARM)
    record: dict[str, object] = {
        "schema_version": 2,
        "run_fingerprint": args.fingerprint,
        "manifest_sha256": _sha256(args.manifest),
        "cell_index": args.cell_index,
        "cell_key": cell["cell_key"],
        "kernel": cell["kernel"],
        "shape": cell["shape"],
        "bench_shape": cell["bench_shape"],
        "aot_key": cell["aot_key"],
        "aot_index": cell["aot_index"],
        "requested_arms": requested_arms,
        "status": "starting",
    }

    _configure_helion_environment()
    _add_checkout(args.helion_root)
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one CUDA device must be visible")

    extension: dict[str, object] | None = None
    if args.include_vllm_cuda:
        extension = load_vllm_extension(
            torch,
            args.vllm_extension,
            args.vllm_root,
            include_sha256=False,
        )
    _stub_vllm_import()

    import helion
    import helion.autotuner.aot_cache as aot_cache

    # Neither default nor seed discovery may silently select a checked-in AOT config.
    aot_cache.find_heuristic_file = lambda *_args, **_kwargs: None

    module = _import_kernel_module(args.helion_root, cell["kernel"])
    original_kernel = getattr(module, cell["kernel"])
    bench_module = importlib.import_module("_bench")
    bench_shape = tuple(int(value) for value in cell["bench_shape"])

    default, seed, discovery = _discover_default_and_seed(
        module,
        cell["kernel"],
        bench_shape,
        bench_module,
        torch,
    )
    exact_aot = _exact_aot_config(cell)
    configs = {
        "default": default,
        "seed": seed,
        "aot_tuned": helion.Config(**exact_aot),
    }
    record["configs"] = {name: _config(config) for name, config in configs.items()}
    record["config_discovery"] = discovery

    prepared: list[CapturedArm] = []
    arm_results: dict[str, dict[str, object]] = {}
    for name in requested_arms:
        try:
            if name == VLLM_ARM:
                implementation = getattr(module, f"_{cell['kernel']}_vllm", None)
                if implementation is None:
                    raise RuntimeError(f"{cell['kernel']} has no vLLM adapter")
                selection: dict[str, object] = {
                    "backend": "cuda_cpp_extension",
                    "entry_point": f"torch.ops._C.{cell['kernel']}",
                    "extension": extension,
                }
                expected = None
            else:
                implementation = _fixed_kernel(helion, original_kernel, configs[name])
                selection = {
                    "backend": "helion_explicit_config",
                    "config": _config(configs[name]),
                    "source": (
                        "base compiler default"
                        if name == "default"
                        else "compiler heuristic seed"
                        if name == "seed"
                        else "exact architecture-specific AOT selector entry"
                    ),
                }
                if name == "aot_tuned":
                    selection.update(
                        exact_aot_key=True,
                        aot_key=cell["aot_key"],
                        aot_index=cell["aot_index"],
                        aot_file=cell["aot_file"],
                    )
                expected = exact_aot if name == "aot_tuned" else None
            captured, result = _prepare_arm(
                name,
                module,
                cell["kernel"],
                implementation,
                bench_shape,
                bench_module,
                torch,
                selection,
                expected,
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

    if prepared:
        samples_us = {arm.name: [] for arm in prepared}
        calls = [arm.replay for arm in prepared]
        for _round in range(args.rounds):
            timings_ms = bench_module.bench_pre_captured_cudagraphs(
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
        args.vllm_root,
        torch,
    )
    record["status"] = (
        "ok"
        if all(arm_results.get(name, {}).get("status") == "ok" for name in requested_arms)
        else "partial"
    )
    return record


def _one(args: argparse.Namespace) -> None:
    try:
        record = _run_one(args)
    except Exception as error:
        manifest = _read_json(args.manifest)
        cell = manifest["cells"][args.cell_index]
        record = {
            "schema_version": 2,
            "run_fingerprint": args.fingerprint,
            "manifest_sha256": _sha256(args.manifest),
            "cell_index": args.cell_index,
            "cell_key": cell["cell_key"],
            "kernel": cell["kernel"],
            "shape": cell["shape"],
            "bench_shape": cell["bench_shape"],
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
    value: dict[str, object] = {
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "manifest_sha256": _sha256(args.manifest),
        "rounds": args.rounds,
        "repetitions": args.repetitions,
        "include_vllm_cuda": args.include_vllm_cuda,
        "helion_git": _git_metadata(args.helion_root),
        "vllm_git": _git_metadata(args.vllm_root),
    }
    if args.include_vllm_cuda:
        extension = find_vllm_extension(args.vllm_extension, args.vllm_root)
        value["vllm_extension"] = {
            "path": str(extension),
            "sha256": _sha256(extension),
        }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _all(args: argparse.Namespace) -> None:
    manifest = _read_json(args.manifest)
    selected = list(enumerate(manifest["cells"]))
    if args.kernels:
        names = {name.strip() for name in args.kernels.split(",") if name.strip()}
        selected = [(index, cell) for index, cell in selected if cell["kernel"] in names]
    if args.limit is not None:
        selected = selected[: args.limit]

    output = args.output.expanduser().resolve()
    parts = output.parent / f"{output.stem}.parts"
    parts.mkdir(parents=True, exist_ok=True)
    fingerprint = _fingerprint(args)
    records: list[dict[str, object]] = []

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
            "--fingerprint",
            fingerprint,
            "--output",
            str(part),
        ]
        if args.vllm_root is not None:
            command.extend(["--vllm-root", str(args.vllm_root)])
        if args.vllm_extension is not None:
            command.extend(["--vllm-extension", str(args.vllm_extension)])
        if args.include_vllm_cuda:
            command.append("--include-vllm-cuda")
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0 and not part.exists():
            _atomic_json(
                part,
                {
                    "schema_version": 2,
                    "run_fingerprint": fingerprint,
                    "cell_index": index,
                    "cell_key": cell["cell_key"],
                    "kernel": cell["kernel"],
                    "shape": cell["shape"],
                    "status": "child_error",
                    "returncode": completed.returncode,
                },
            )
        records.append(_read_json(part))

    requested_arms = [*HELION_ARMS]
    if args.include_vllm_cuda:
        requested_arms.append(VLLM_ARM)
    document = {
        "schema_version": 2,
        "profile": manifest["profile"],
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": _sha256(args.manifest),
        "run_fingerprint": fingerprint,
        "helion_git": _git_metadata(args.helion_root),
        "vllm_git": _git_metadata(args.vllm_root),
        "arms": requested_arms,
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
    parser.add_argument("--vllm-root", type=Path)
    parser.add_argument("--vllm-extension", type=Path)
    parser.add_argument("--include-vllm-cuda", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--kernels")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cell-index", type=int)
    parser.add_argument("--fingerprint", default="")
    args = parser.parse_args()

    args.manifest = args.manifest.expanduser().resolve()
    args.helion_root = args.helion_root.expanduser().resolve()
    args.vllm_root = (
        args.vllm_root.expanduser().resolve() if args.vllm_root else None
    )
    args.vllm_extension = (
        args.vllm_extension.expanduser().resolve()
        if args.vllm_extension
        else None
    )
    args.output = args.output.expanduser().resolve()
    if args.rounds <= 0 or args.repetitions <= 0:
        parser.error("--rounds and --repetitions must be positive")
    if args.include_vllm_cuda and args.vllm_root is None:
        parser.error("--vllm-root is required with --include-vllm-cuda")

    if args.action == "one":
        if args.cell_index is None:
            parser.error("--cell-index is required for --action one")
        _one(args)
    else:
        _all(args)


if __name__ == "__main__":
    main()
