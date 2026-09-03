#!/usr/bin/env python3
"""Benchmark Helion's raw default and promoted compiler seed together."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HELION_ARMS = ("default", "seed")
TORCH_COMPILE_ARM = "torch_compile"
INDUCTOR_OPTIONS = {
    "max_autotune_gemm_backends": "TRITON",
    "max_autotune_conv_backends": "TRITON",
    "triton.autotune_cublasLt": False,
    "fx_graph_cache": False,
    "force_disable_caches": True,
}
EXTERNAL_CALL = re.compile(
    r"(?:extern_kernels|torch\.ops\.(?:aten|cuda|cudnn))\.[A-Za-z_][\w.]*\s*\("
)
EXTERNAL_LIBRARY = re.compile(
    r"\b(?:cublas|cudnn|cutlass|flash[_ ]?attention|"
    r"efficient_attention|nvgemm)\b",
    re.IGNORECASE,
)


def _arms(include_torch_compile: bool) -> tuple[str, ...]:
    return (
        *HELION_ARMS,
        *((TORCH_COMPILE_ARM,) if include_torch_compile else ()),
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        json.dump(value, handle, indent=2, default=str)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


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


def _expand_manifest(document: dict[str, Any]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for family, family_spec in document["families"].items():
        for shape in family_spec["shapes"]:
            cells.append(
                {
                    "cell_key": f"{family}:{shape['id']}",
                    "family": family,
                    "family_label": family_spec["label"],
                    "category": family_spec["category"],
                    "expected_heuristic": family_spec["expected_heuristic"],
                    "shape_id": shape["id"],
                    "shape_convention": family_spec["shape_convention"],
                    "values": shape["values"],
                    "dtype": shape.get("dtype", "bfloat16"),
                    "source": shape["source"],
                    "shape_spec": shape,
                }
            )
    keys = [cell["cell_key"] for cell in cells]
    if len(keys) != len(set(keys)):
        raise ValueError("manifest contains duplicate cell keys")
    return cells


def _jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "value"):
        return _jsonable(value.value)
    return repr(value)


def _config(value: object | None) -> dict[str, object] | None:
    if value is None:
        return None
    config = getattr(value, "config", value)
    return _jsonable(dict(config))  # type: ignore[arg-type,return-value]


def _canonical_config(value: object) -> str:
    return json.dumps(_config(value), sort_keys=True, separators=(",", ":"))


def _tensor_outputs(value: object) -> list[Any]:
    import torch

    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        return [
            tensor
            for key in sorted(value)
            for tensor in _tensor_outputs(value[key])
        ]
    if isinstance(value, (tuple, list)):
        return [tensor for item in value for tensor in _tensor_outputs(item)]
    return []


def _finite_outputs(value: object) -> dict[str, object]:
    import torch

    outputs = _tensor_outputs(value)
    if not outputs:
        return {"pass": False, "error": "kernel returned no tensor outputs"}
    details = []
    passed = True
    for output in outputs:
        finite = bool(torch.isfinite(output.float()).all())
        passed = passed and finite
        details.append(
            {
                "shape": list(output.shape),
                "dtype": str(output.dtype).removeprefix("torch."),
                "finite": finite,
            }
        )
    return {"pass": passed, "outputs": details}


def _accuracy(actual: object, expected: object) -> dict[str, object]:
    import torch

    actual_outputs = _tensor_outputs(actual)
    expected_outputs = _tensor_outputs(expected)
    if len(actual_outputs) != len(expected_outputs):
        return {
            "pass": False,
            "error": (
                f"output count differs: {len(actual_outputs)} != "
                f"{len(expected_outputs)}"
            ),
        }
    passed = True
    details = []
    for actual_tensor, expected_tensor in zip(
        actual_outputs, expected_outputs, strict=True
    ):
        shape_ok = actual_tensor.shape == expected_tensor.shape
        dtype_ok = actual_tensor.dtype == expected_tensor.dtype
        actual32 = actual_tensor.float()
        expected32 = expected_tensor.float()
        finite = bool(torch.isfinite(actual32).all()) and bool(
            torch.isfinite(expected32).all()
        )
        if shape_ok and finite:
            difference = (actual32 - expected32).abs()
            max_abs = float(difference.max()) if difference.numel() else 0.0
            scale = max(float(expected32.abs().max()), 1e-6)
            max_rel = max_abs / scale
            values_ok = max_abs <= 0.1 or max_rel <= 0.05
        else:
            max_abs = float("inf")
            max_rel = float("inf")
            values_ok = False
        output_ok = shape_ok and dtype_ok and finite and values_ok
        passed = passed and output_ok
        details.append(
            {
                "pass": output_ok,
                "shape": list(actual_tensor.shape),
                "dtype": str(actual_tensor.dtype).removeprefix("torch."),
                "shape_matches": shape_ok,
                "dtype_matches": dtype_ok,
                "finite": finite,
                "max_abs": max_abs,
                "max_rel_to_output_max": max_rel,
            }
        )
    return {
        "pass": passed,
        "criterion": "max_abs <= 0.1 or max_abs/max(reference_abs_max,1e-6) <= 0.05",
        "outputs": details,
    }


def _build_fixed(helion: Any, original: Any, config: dict[str, object]) -> Any:
    return helion.kernel(
        original.fn,
        config=helion.Config.from_dict(config),
        autotune_effort="none",
    )


def _audit_inductor_sources(
    sources: list[str],
    options: dict[str, object],
) -> dict[str, object]:
    forbidden = []
    for source_index, source in enumerate(sources):
        for line_number, line in enumerate(source.splitlines(), 1):
            stripped = line.strip()
            if EXTERNAL_CALL.search(stripped) or EXTERNAL_LIBRARY.search(stripped):
                forbidden.append(
                    {
                        "source": source_index,
                        "line": line_number,
                        "text": stripped[:500],
                    }
                )
    triton_kernels = sum(source.count("async_compile.triton(") for source in sources)
    return {
        "pass": bool(sources) and triton_kernels > 0 and not forbidden,
        "policy": (
            "Require generated Triton kernels and reject generated calls through "
            "ATen/extern_kernels or references to cuBLAS, cuDNN, CUTLASS, "
            "Flash/Efficient Attention, or NVGEMM."
        ),
        "inductor_options": options,
        "generated_source_count": len(sources),
        "generated_source_sha256": [_sha256_text(source) for source in sources],
        "generated_source_bytes": [len(source.encode()) for source in sources],
        "triton_kernel_count": triton_kernels,
        "forbidden_matches": forbidden,
    }


def _compile_torch_reference(
    workload: Any,
    mode: str,
    torch: Any,
    options: dict[str, object],
) -> tuple[Any, object, dict[str, object]]:
    from torch._inductor import config as inductor_config
    from torch._inductor.utils import run_and_get_code

    torch._dynamo.reset()
    with inductor_config.patch(options):
        compiled = torch.compile(
            workload.reference_fn,
            mode=mode,
            fullgraph=True,
        )
        output, sources = run_and_get_code(compiled, *workload.args)
        torch.cuda.synchronize()
    return compiled, output, _audit_inductor_sources(sources, options)


def _matmul_facts(spec: Any) -> list[dict[str, object]]:
    keys = (
        "static_m",
        "static_n",
        "static_k",
        "m_block_id",
        "n_block_id",
        "k_block_id",
    )
    return [
        {key: _jsonable(getattr(fact, key, None)) for key in keys}
        for fact in spec.matmul_facts
    ]


def _environment(torch: Any, helion: Any, helion_root: Path) -> dict[str, object]:
    versions: dict[str, object] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "helion_file": helion.__file__,
    }
    try:
        import triton

        versions["triton"] = triton.__version__
    except (ImportError, AttributeError):
        versions["triton"] = None
    return {
        "host": platform.node(),
        "platform": platform.platform(),
        "versions": versions,
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "visible_device": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "helion_git": _git_metadata(helion_root),
    }


def _run_cell(
    cell: dict[str, Any],
    helion_root: Path,
    rounds: int,
    row_index: int,
    include_torch_compile: bool,
    torch_compile_mode: str,
) -> dict[str, object]:
    os.environ["HELION_ROOT"] = str(helion_root)
    os.environ["HELION_AUTOTUNE_EFFORT"] = "none"
    os.environ["HELION_SKIP_CACHE"] = "1"
    os.environ["HELION_DISABLE_AUTOTUNER_HEURISTICS"] = "0"
    sys.path.insert(0, str(helion_root))

    import helion
    import torch
    from helion._testing import do_bench
    from workloads import build_workload

    torch.cuda.set_device(0)
    workload = build_workload(cell)
    arms = _arms(include_torch_compile)
    kernel = workload.kernel_fn
    kernel.reset()
    bound = kernel.bind(workload.args)
    spec = bound.config_spec
    inductor_options = {
        **INDUCTOR_OPTIONS,
        **(workload.inductor_options or {}),
    }
    fired = [str(value) for value in spec.autotuner_heuristics]
    compiler_seeds = list(spec.compiler_seed_configs)
    default_config = dict(spec._base_default_config())
    seed_config = dict(spec.default_config()) if compiler_seeds else None
    result: dict[str, object] = {
        **{key: value for key, value in cell.items() if key != "shape_spec"},
        "body": workload.body,
        "reference": workload.reference_name,
        "environment": _environment(torch, helion, helion_root),
        "heuristics_fired": fired,
        "expected_heuristic_fired": cell["expected_heuristic"] in fired,
        "compiler_default_fragment": _config(spec.compiler_default_config),
        "compiler_seed_count": len(compiler_seeds),
        "compiler_seed_configs": [_config(config) for config in compiler_seeds],
        "primary_seed_is_promoted_default": (
            bool(compiler_seeds)
            and spec.compiler_default_config is not None
            and _canonical_config(compiler_seeds[0])
            == _canonical_config(spec.compiler_default_config)
        ),
        "matmul_facts": _matmul_facts(spec),
        "arms": {
            "default": {
                "config": _jsonable(default_config),
                "config_source": "ConfigSpec._base_default_config()",
            },
            "seed": {
                "config": _jsonable(seed_config),
                "config_source": (
                    "ConfigSpec.default_config(): promoted rank-0 compiler "
                    "seed layered over base defaults"
                ),
            },
        },
    }
    if include_torch_compile:
        result["arms"][TORCH_COMPILE_ARM] = {  # type: ignore[index]
            "config_source": (
                f"torch.compile(mode={torch_compile_mode!r}, fullgraph=True)"
            ),
            "reference": workload.reference_name,
            "inductor_options": inductor_options,
        }
    if not compiler_seeds:
        result["arms"]["seed"].update(  # type: ignore[union-attr]
            {"status": "unavailable", "error": "no compiler seed was generated"}
        )

    compiled: dict[str, Any] = {}
    candidates: dict[str, Any] = {}
    config_keys: dict[str, str] = {}
    for arm in HELION_ARMS:
        record = result["arms"][arm]  # type: ignore[index]
        config = record["config"]
        if config is None:
            continue
        key = json.dumps(config, sort_keys=True, default=str)
        config_keys[arm] = key
        if key in compiled:
            record["duplicate_config_of"] = next(
                prior
                for prior, prior_key in config_keys.items()
                if prior != arm and prior_key == key
            )
            record["status"] = "ok"
            candidates[arm] = compiled[key]
            continue
        try:
            candidate = _build_fixed(helion, kernel, config)
            candidate(*workload.args)
            torch.cuda.synchronize()
            compiled[key] = candidate
            candidates[arm] = candidate
            record["status"] = "ok"
        except Exception as error:  # noqa: BLE001 - record arbitrary compiler failures
            record.update(
                {
                    "status": "compile_error",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc()[-4000:],
                }
            )

    if include_torch_compile:
        record = result["arms"][TORCH_COMPILE_ARM]  # type: ignore[index]
        try:
            candidate, compile_output, audit = _compile_torch_reference(
                workload,
                torch_compile_mode,
                torch,
                inductor_options,
            )
            record["audit"] = audit
            if not audit["pass"]:
                record.update(
                    {
                        "status": "audit_rejected",
                        "error": (
                            "generated TorchInductor source did not satisfy the "
                            "Triton-only dispatch audit"
                        ),
                    }
                )
            else:
                candidates[TORCH_COMPILE_ARM] = candidate
                record["status"] = "ok"
            del compile_output
        except Exception as error:  # noqa: BLE001 - record compiler failures
            record.update(
                {
                    "status": "compile_error",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc()[-8000:],
                }
            )

    outputs: dict[str, object] = {}
    for arm in arms:
        record = result["arms"][arm]  # type: ignore[index]
        if record.get("status") != "ok":
            continue
        try:
            output = candidates[arm](*workload.args)
            torch.cuda.synchronize()
            outputs[arm] = output
            record["correctness"] = (
                _finite_outputs(output)
                if arm == "default"
                else _accuracy(output, outputs.get("default"))
                if "default" in outputs
                else {"pass": False, "error": "default output unavailable"}
            )
        except Exception as error:  # noqa: BLE001 - record arbitrary runtime failures
            record.update(
                {
                    "status": "runtime_error",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc()[-4000:],
                }
            )

    samples: dict[str, list[float]] = {arm: [] for arm in arms}
    workload_args = workload.args
    eligible = [
        arm
        for arm in arms
        if result["arms"][arm].get("status") == "ok"  # type: ignore[index,union-attr]
        and result["arms"][arm].get("correctness", {}).get("pass")  # type: ignore[index,union-attr]
    ]
    for outer_round in range(rounds):
        offset = (row_index + outer_round) % max(1, len(eligible))
        ordered = eligible[offset:] + eligible[:offset]
        if ((row_index + outer_round) // max(1, len(eligible))) % 2:
            ordered.reverse()
        for arm in ordered:
            candidate = candidates[arm]
            latency_ms = float(
                do_bench(
                    lambda candidate=candidate, args=workload_args: candidate(*args),
                    return_mode="median",
                )
            )
            samples[arm].append(latency_ms * 1000.0)
    for arm in eligible:
        result["arms"][arm]["latency_us"] = statistics.median(samples[arm])  # type: ignore[index]
        result["arms"][arm]["outer_round_samples_us"] = samples[arm]  # type: ignore[index]

    del outputs
    del workload
    torch.cuda.empty_cache()
    return result


def _document(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    manifest_path: Path,
    records: list[dict[str, object]],
    started_at: str,
) -> dict[str, object]:
    arms = _arms(args.torch_compile)
    environment = next(
        (
            record["environment"]
            for record in records
            if isinstance(record.get("environment"), dict)
        ),
        {},
    )
    return {
        "schema_version": 2,
        "profile": manifest["profile"],
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "run_fingerprint": args.run_fingerprint,
        "arms": list(arms),
        "timing": {
            "method": (
                "Helion do_bench CUDA device timing; median within do_bench, "
                "then median of balanced outer rounds"
            ),
            "outer_rounds": args.rounds,
            "arms_compiled_and_timed_together": True,
            "fresh_process_per_shape": True,
        },
        "selection": {
            "default": "ConfigSpec._base_default_config()",
            "seed": (
                "ConfigSpec.default_config(), the promoted rank-0 compiler "
                "seed merged with base defaults"
            ),
            "torch_compile": (
                f"natural PyTorch reference compiled with {args.torch_compile_mode}; "
                "TorchInductor GEMM/conv choices restricted to Triton and "
                "generated source audited"
                if args.torch_compile
                else "not requested"
            ),
            "autotuning": "disabled",
            "tuned_cache": "disabled",
        },
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_gpu": args.gpu,
        "helion_root": str(args.helion_root),
        "helion_git": _git_metadata(args.helion_root),
        "environment": environment,
        "records": records,
    }


def _child(args: argparse.Namespace) -> None:
    manifest = _read_json(args.manifest)
    cell = next(
        cell
        for cell in _expand_manifest(manifest)
        if cell["cell_key"] == args.cell_key
    )
    try:
        result = _run_cell(
            cell,
            args.helion_root,
            args.rounds,
            args.row_index,
            args.torch_compile,
            args.torch_compile_mode,
        )
        result["run_fingerprint"] = args.run_fingerprint
    except Exception as error:  # noqa: BLE001 - isolate every shape-level failure
        result = {
            **{key: value for key, value in cell.items() if key != "shape_spec"},
            "driver_error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc()[-8000:],
            "run_fingerprint": args.run_fingerprint,
        }
    _atomic_json(args.part_output, result)
    print(json.dumps(result, default=str), flush=True)


def _parent(args: argparse.Namespace) -> None:
    manifest = _read_json(args.manifest)
    cells = _expand_manifest(manifest)
    if args.families:
        selected = set(args.families.split(","))
        cells = [cell for cell in cells if cell["family"] in selected]
    if args.limit is not None:
        cells = cells[: args.limit]
    if not cells:
        raise ValueError("no cells selected")

    started_at = datetime.now(timezone.utc).isoformat()
    args.run_fingerprint = _run_fingerprint(args)
    requested_arms = set(_arms(args.torch_compile))

    def compatible(record: dict[str, object]) -> bool:
        return (
            "driver_error" not in record
            and record.get("run_fingerprint") == args.run_fingerprint
            and requested_arms.issubset(record.get("arms", {}))  # type: ignore[arg-type]
        )

    prior: dict[str, dict[str, object]] = {}
    if args.resume and args.output.exists():
        prior = {
            record["cell_key"]: record
            for record in _read_json(args.output).get("records", [])
            if compatible(record)
        }

    parts = args.output.parent / "benchmark.parts"
    parts.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    for index, cell in enumerate(cells):
        key = cell["cell_key"]
        part = parts / f"{key.replace(':', '__')}.json"
        if args.resume and key in prior:
            print(f"[{index + 1}/{len(cells)}] cached {key}", flush=True)
            continue
        if args.resume and part.exists():
            cached = _read_json(part)
            if compatible(cached):
                prior[key] = cached
                print(f"[{index + 1}/{len(cells)}] cached part {key}", flush=True)
                continue
        command = [
            sys.executable,
            str(script),
            "--helion-root",
            str(args.helion_root),
            "--manifest",
            str(args.manifest),
            "--output",
            str(args.output),
            "--rounds",
            str(args.rounds),
            "--gpu",
            str(args.gpu),
            "--cell-key",
            key,
            "--row-index",
            str(index),
            "--part-output",
            str(part),
            "--torch-compile-mode",
            args.torch_compile_mode,
            "--run-fingerprint",
            args.run_fingerprint,
        ]
        if args.torch_compile:
            command.append("--torch-compile")
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        pythonpath = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = os.pathsep.join(
            value
            for value in (str(args.helion_root), str(script.parent), pythonpath)
            if value
        )
        try:
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                env=environment,
                timeout=args.cell_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            process = None
            result = {
                **{key: value for key, value in cell.items() if key != "shape_spec"},
                "driver_error": f"cell timed out after {args.cell_timeout}s",
                "stdout": (error.stdout or "")[-2000:],
                "stderr": (error.stderr or "")[-4000:],
            }
            _atomic_json(part, result)
        if process is not None:
            try:
                result = _read_json(part)
            except (OSError, json.JSONDecodeError):
                result = {
                    **{
                        item_key: value
                        for item_key, value in cell.items()
                        if item_key != "shape_spec"
                    },
                    "driver_error": f"child exited {process.returncode}",
                    "stdout": process.stdout[-2000:],
                    "stderr": process.stderr[-4000:],
                }
                _atomic_json(part, result)
        prior[key] = result
        ordered = [prior[cell["cell_key"]] for cell in cells if cell["cell_key"] in prior]
        _atomic_json(
            args.output,
            _document(args, manifest, args.manifest, ordered, started_at),
        )
        status = "driver_error" if "driver_error" in result else "ok"
        print(f"[{index + 1}/{len(cells)}] {key}: {status}", flush=True)

    ordered = [prior[cell["cell_key"]] for cell in cells if cell["cell_key"] in prior]
    _atomic_json(
        args.output,
        _document(args, manifest, args.manifest, ordered, started_at),
    )
    print(f"Wrote {args.output} with {len(ordered)} records", flush=True)


def _run_fingerprint(args: argparse.Namespace) -> str:
    scripts = Path(__file__).resolve().parent
    value = {
        "runner": _sha256(Path(__file__).resolve()),
        "workloads": _sha256(scripts / "workloads.py"),
        "manifest": _sha256(args.manifest),
        "helion_git": _git_metadata(args.helion_root),
        "rounds": args.rounds,
        "torch_compile": args.torch_compile,
        "torch_compile_mode": args.torch_compile_mode,
    }
    return _sha256_text(json.dumps(value, sort_keys=True, default=str))


def main() -> None:
    default_manifest = Path(__file__).resolve().parents[1] / "shapes.json"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--helion-root",
        type=Path,
        default=os.environ.get("HELION_ROOT"),
        required=os.environ.get("HELION_ROOT") is None,
    )
    parser.add_argument("--manifest", type=Path, default=default_manifest)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--families", help="comma-separated family IDs")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cell-timeout", type=int, default=1800)
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="also benchmark the Triton-only torch.compile PyTorch reference",
    )
    parser.add_argument(
        "--torch-compile-mode",
        default="max-autotune-no-cudagraphs",
    )
    parser.add_argument("--cell-key", help=argparse.SUPPRESS)
    parser.add_argument("--row-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--part-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--run-fingerprint", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.helion_root = args.helion_root.expanduser().resolve()
    args.manifest = args.manifest.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.cell_key:
        if args.part_output is None:
            parser.error("--part-output is required with --cell-key")
        _child(args)
    else:
        _parent(args)


if __name__ == "__main__":
    main()
