"""Preflight a Helion linear-attention versus FLA benchmark environment."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def _command(*args: str, cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(
            args,
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_info(path: Path) -> dict[str, Any]:
    root = _command("git", "rev-parse", "--show-toplevel", cwd=path)
    if root is None:
        return {"root": str(path), "commit": None, "dirty": None}
    root_path = Path(root)
    status = _command("git", "status", "--porcelain", cwd=root_path)
    return {
        "root": str(root_path),
        "commit": _command("git", "rev-parse", "HEAD", cwd=root_path),
        "dirty": bool(status),
    }


def _package_version(*names: str) -> str | None:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return None


def _under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _add_import_roots(helion: Path, fla: Path | None) -> None:
    candidates = [helion]
    if fla is not None:
        candidates.extend((fla, fla / "src", fla / "python"))
    for candidate in reversed(candidates):
        if candidate.exists():
            sys.path.insert(0, str(candidate.resolve()))


def _workload_info(module: Any) -> dict[str, Any]:
    variants = list(module.VARIANTS)
    dense = set(module.DENSE_VARIANTS)
    fused = set(module.FUSED_PREAMBLE_VARIANTS)
    varlen = set(module.VARLEN_VARIANTS)
    cells = 0
    for variant in variants:
        if variant in varlen:
            cells += len(module.VARLEN_SHAPES)
        elif variant in fused:
            cells += len(module.SHAPES)
        elif variant in dense:
            cells += 2 * len(module.SHAPES)
    return {
        "variants": variants,
        "dense_shapes": len(module.SHAPES),
        "varlen_shapes": len(module.VARLEN_SHAPES),
        "operation_cells": cells,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--helion", type=Path, required=True)
    parser.add_argument(
        "--fla",
        default=None,
        help="FLA checkout root, or omit when FLA is already installed",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    helion_root = args.helion.expanduser().resolve()
    fla_root = (
        None
        if args.fla is None or args.fla.upper() == "NONE"
        else Path(args.fla).expanduser().resolve()
    )
    _add_import_roots(helion_root, fla_root)

    errors: list[str] = []
    warnings: list[str] = []
    required_files = (
        "benchmarks/run_linattn.py",
        "examples/linear/linear_attention_harness.py",
        "examples/linear/linear_attention_fla.py",
        "examples/linear/linear_attention_engine.py",
    )
    missing = [name for name in required_files if not (helion_root / name).is_file()]
    if missing:
        errors.append(f"missing Helion files: {', '.join(missing)}")

    report: dict[str, Any] = {
        "ok": False,
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "environment": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "PYTHONPATH",
                "HELION_AUTOTUNE_EFFORT",
                "HELION_AUTOTUNE_CACHE",
                "HELION_DISABLE_AUTOTUNER_HEURISTICS",
                "HELION_AOT_MODE",
                "HELION_HEURISTIC_DIR",
            )
            if os.environ.get(name) is not None
        },
        "helion_git": _git_info(helion_root),
        "fla_git": _git_info(fla_root) if fla_root is not None else None,
    }

    try:
        import fla
        import helion
        import torch
        import triton
        from benchmarks import run_linattn
        from examples.linear.linear_attention_engine import LinearAttentionVariant
        from examples.linear.linear_attention_fla import get_fla_fwd_kernel
    except Exception as error:  # noqa: BLE001
        errors.append(f"import failed: {type(error).__name__}: {error}")
    else:
        helion_file = Path(helion.__file__).resolve()
        if not _under(helion_file, helion_root):
            errors.append(
                f"imported Helion from {helion_file}, not requested root {helion_root}"
            )

        fla_file = Path(fla.__file__).resolve()
        if fla_root is not None and not _under(fla_file, fla_root):
            errors.append(
                f"imported FLA from {fla_file}, not requested root {fla_root}"
            )

        report["software"] = {
            "helion_file": str(helion_file),
            "fla_file": str(fla_file),
            "fla_version": getattr(fla, "__version__", None)
            or _package_version("flash-linear-attention", "fla-core"),
            "torch": torch.__version__,
            "triton": triton.__version__,
            "cuda": torch.version.cuda,
        }
        report["workload"] = _workload_info(run_linattn)

        fla_support = {
            variant.value: get_fla_fwd_kernel(variant) is not None
            for variant in LinearAttentionVariant
        }
        report["fla_forward_support"] = fla_support
        expected = {
            "kda"
            if name in run_linattn.FUSED_PREAMBLE_VARIANTS
            or name in run_linattn.VARLEN_VARIANTS
            else name
            for name in run_linattn.VARIANTS
        }
        unsupported = sorted(name for name in expected if not fla_support.get(name))
        if unsupported:
            errors.append(f"FLA wrappers unavailable for: {', '.join(unsupported)}")

        if not torch.cuda.is_available():
            errors.append("torch.cuda.is_available() is false")
        else:
            major, minor = torch.cuda.get_device_capability()
            architecture = f"sm{major}{minor}"
            aot_relative = (
                "examples/linear/"
                f"_helion_aot_linear_attention_engine_cuda_{architecture}.py"
            )
            aot_path = helion_root / aot_relative
            report["hardware"] = {
                "gpu": torch.cuda.get_device_name(),
                "compute_capability": [major, minor],
                "architecture": architecture,
                "aot_module": aot_relative if aot_path.is_file() else None,
                "nvidia_smi": _command(
                    "nvidia-smi",
                    "--query-gpu=index,name,clocks.current.sm,"
                    "clocks.current.memory,power.limit,power.draw,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ),
            }
            if not aot_path.is_file():
                warnings.append(
                    f"no generated AOT module for {architecture}; "
                    "the pre_tuned arm will be unavailable"
                )

    report["errors"] = errors
    report["warnings"] = warnings
    report["ok"] = not errors
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
