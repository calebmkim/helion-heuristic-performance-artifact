"""Measure discovered linear-attention cells against FLA in isolated processes.

This is the adapter used for the H100 PR #3546 measurements, generalized to
accept checkout and manifest paths. Helion APIs can move; treat it as a tested
starting point and adapt its imports while preserving the arm semantics.
"""

from __future__ import annotations

import argparse
import csv
from functools import cache
import importlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import traceback
from typing import Any

from benchmark_paths import add_checkout
from benchmark_paths import checkout_path
from benchmark_paths import linear_aot_module

_ARM_ALIASES = {
    "default": "default",
    "raw": "default",
    "seed": "seed",
    "compiler": "seed",
    "aot_tuned": "aot_tuned",
    "tuned": "aot_tuned",
}
torch: Any = None
do_bench: Any = None


def _rows(path: Path, mode: str) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows if mode == "all" else [row for row in rows if row["mode"] == mode]


def _bench_ms(
    fn: object,
    rounds: int,
    grad_to_none: list[Any] | None = None,
) -> tuple[float, list[float]]:
    """Use Helion's benchmark statistic; optional outer rounds are explicit."""
    samples = [
        float(do_bench(fn, grad_to_none=grad_to_none))  # type: ignore[arg-type]
        for _ in range(rounds)
    ]
    return statistics.median(samples), samples


def _install_aot_module(aot_module: str | None) -> None:
    """Load the architecture's checked-in AOT selector without changing it."""
    module = importlib.import_module(linear_aot_module(aot_module))

    # AOTAutotuneCache loads the generated module by path instead of package
    # name. Seed both path spellings so an explicit module is honored.
    from helion.autotuner.aot_cache import AOTAutotuneCache

    module_path = Path(module.__file__)
    AOTAutotuneCache._heuristic_modules[module_path] = module
    AOTAutotuneCache._heuristic_modules[module_path.resolve()] = module


def _set_arm_environment(arm: str) -> None:
    """Set selection controls before Helion is imported in the child."""
    os.environ["HELION_AUTOTUNE_EFFORT"] = "none"
    if arm == "aot_tuned":
        # "none" resolves the implicit compiler default before cache lookup.
        # Evaluate mode does not search; "full" only allows the AOT cache read.
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


def _load_runtime(helion_root: Path | None, fla_root: Path | None) -> None:
    global do_bench, torch
    # Insert Helion last so its checkout takes precedence over an installed copy.
    add_checkout(fla_root)
    add_checkout(helion_root)
    torch = importlib.import_module("torch")
    do_bench = importlib.import_module("helion._testing").do_bench


def _configure_arm(arm: str, aot_module: str | None) -> None:
    if arm == "aot_tuned":
        _install_aot_module(aot_module)
        return

    import helion.autotuner.aot_cache as aot_cache

    aot_cache.find_heuristic_file = lambda *_args, **_kwargs: None
    if arm == "default":
        from helion.autotuner.config_spec import ConfigSpec

        ConfigSpec.default_config = ConfigSpec._base_default_config


def _kernel_metadata() -> list[dict[str, object]]:
    """Record configs actually attached to operation kernels after execution."""
    from examples.linear import linear_attention_engine as engine

    from helion.autotuner.aot_cache import AOTAutotuneCache
    from helion.runtime.kernel import Kernel

    aot_configs: dict[str, list[dict[str, object]]] = {}
    for (_source, kernel_name, _shape), config in (
        AOTAutotuneCache._heuristic_results.items()
    ):
        aot_configs.setdefault(kernel_name, []).append(dict(config))

    records: list[dict[str, object]] = []
    seen: set[int] = set()
    for name, kernel in vars(engine).items():
        if not isinstance(kernel, Kernel):
            continue
        for bound in kernel._bound_kernels.values():
            if id(bound) in seen:
                continue
            seen.add(id(bound))
            spec = bound.config_spec
            config = dict(bound._config) if bound._config is not None else None
            records.append(
                {
                    "kernel": name,
                    "config": config,
                    "autotune_cache": bound.settings.autotune_cache,
                    "autotune_effort": bound.settings.autotune_effort,
                    "selected_by_aot_cache": config in aot_configs.get(name, []),
                    "heuristics_fired": list(spec.autotuner_heuristics),
                    "matmul_facts": [
                        [fact.static_m, fact.static_n, fact.static_k]
                        for fact in spec.matmul_facts
                    ],
                }
            )
    return records


def _aot_audit(kernels: list[dict[str, object]]) -> dict[str, object]:
    from helion.autotuner.aot_cache import AOTAutotuneCache

    unverified = [
        str(record["kernel"])
        for record in kernels
        if record["selected_by_aot_cache"] is not True
    ]
    if unverified:
        raise RuntimeError(
            "AOT-tuned arm has configs not selected by AOTAutotuneCache: "
            + ", ".join(sorted(set(unverified)))
        )
    selected = [str(record["kernel"]) for record in kernels]
    return {
        "selected_bound_configs": len(selected),
        "selected_kernel_names": sorted(set(selected)),
        "fallback_kernel_names": sorted(AOTAutotuneCache._no_heuristic_warned),
    }


def _git_revision(path: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(Path(path).parent), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


@cache
def _environment_metadata() -> dict[str, object]:
    import triton

    import fla
    import helion

    metadata: dict[str, object] = {
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
        metadata["helion_revision"] = revision
    if revision := _git_revision(fla.__file__):
        metadata["fla_revision"] = revision
    try:
        metadata["gpu_state"] = subprocess.check_output(
            [
                "nvidia-smi",
                (
                    "--query-gpu=clocks.current.sm,clocks.current.memory,"
                    "power.limit,power.draw,temperature.gpu"
                ),
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
    except Exception as error:
        metadata["gpu_state_error"] = f"{type(error).__name__}: {error}"
    return metadata


def _forward_cell(
    row: dict[str, str],
    rounds: int,
    row_index: int,
    arm: str,
    label: str,
) -> dict[str, object]:
    from examples.linear.linear_attention_engine import LinearAttentionVariant
    from examples.linear.linear_attention_harness import _VARIANT_SPECS
    from examples.linear.linear_attention_harness import ACC_FWD_TOL
    from examples.linear.linear_attention_harness import DTYPE
    from examples.linear.linear_attention_harness import LinearAttentionExampleHarness
    from examples.linear.linear_attention_harness import _fla_inputs
    from examples.linear.linear_attention_utils import rel_error

    base_variant = row["variant"]
    fused = base_variant == "kda_fused"
    varlen = base_variant == "kda_varlen"
    variant_name = "kda" if fused or varlen else base_variant
    variant = next(
        item for item in LinearAttentionVariant if item.value == variant_name
    )
    harness = LinearAttentionExampleHarness(variant)
    _title, make_inputs, _grad_tensors = _VARIANT_SPECS[variant]

    torch.manual_seed(42)
    extra: dict[str, object] = {}
    if fused:
        extra["fused_preamble"] = True
    if varlen:
        lengths = json.loads(row["lengths_json"])
        shape = [len(lengths), int(row["heads"]), sum(lengths), int(row["dim"])]
        shape.append(shape[-1])
        extra.update(varlen=True, varlen_lengths=lengths)
    else:
        shape = json.loads(row["shape_json"])
    inputs = make_inputs(*shape, dtype=DTYPE, device="cuda", **extra)
    fla_inputs = _fla_inputs(inputs)

    def helion_fn() -> torch.Tensor:
        return harness.helion_fwd(inputs, 64)

    def fla_fn() -> torch.Tensor:
        return harness.fla_fwd(fla_inputs, inputs.scale)

    helion_out = helion_fn()
    fla_out = fla_fn()
    if not varlen:
        fla_out = fla_out.transpose(1, 2).contiguous()
    error = float(rel_error(helion_out, fla_out))
    finite = bool(torch.isfinite(helion_out).all())

    labels = ["helion", "fla"]
    if row_index % 2:
        labels.reverse()
    timings: dict[str, float] = {}
    samples: dict[str, list[float]] = {}
    for timing_label in labels:
        value, observed = _bench_ms(
            helion_fn if timing_label == "helion" else fla_fn, rounds
        )
        timings[timing_label] = value
        samples[timing_label] = observed

    return {
        "cell_key": row["cell_key"],
        "variant": row["variant"],
        "mode": row["mode"],
        "shape_name": row["shape_name"],
        "arm": arm,
        "label": label,
        "accuracy_ok": finite and error < ACC_FWD_TOL,
        "relative_error_vs_fla": error,
        "helion_ms": timings["helion"],
        "fla_ms": timings["fla"],
        "speedup_vs_fla": timings["fla"] / timings["helion"],
        "samples_ms": samples,
    }


def _backward_cell(
    row: dict[str, str],
    rounds: int,
    row_index: int,
    arm: str,
    label: str,
) -> dict[str, object]:
    from examples.linear.linear_attention_engine import LinearAttentionVariant
    from examples.linear.linear_attention_harness import _VARIANT_SPECS
    from examples.linear.linear_attention_harness import ACC_BWD_TOL
    from examples.linear.linear_attention_harness import DTYPE
    from examples.linear.linear_attention_harness import LinearAttentionExampleHarness
    from examples.linear.linear_attention_harness import _fla_benchmark_inputs
    from examples.linear.linear_attention_harness import _grad_leaves
    from examples.linear.linear_attention_utils import rel_error

    variant = next(
        item for item in LinearAttentionVariant if item.value == row["variant"]
    )
    harness = LinearAttentionExampleHarness(variant)
    _title, make_inputs, _grad_tensors = _VARIANT_SPECS[variant]

    torch.manual_seed(42)
    shape = json.loads(row["shape_json"])
    inputs = make_inputs(
        *shape,
        dtype=DTYPE,
        device="cuda",
        requires_grad=True,
    )
    h_inputs, h_grads = _grad_leaves(harness, inputs)
    fla_inputs, fla_grads = _fla_benchmark_inputs(harness, inputs)
    grad_out = torch.randn(
        shape[0],
        shape[1],
        shape[2],
        shape[4],
        device="cuda",
        dtype=DTYPE,
    )
    go_t = grad_out.transpose(1, 2).contiguous()

    def helion_fn() -> None:
        harness.helion_fb(h_inputs, grad_out, 64)

    def fla_fn() -> None:
        harness.fla_fb(fla_inputs, go_t, inputs.scale)

    helion_fn()
    torch.cuda.synchronize()
    helion_grad_values = [leaf.grad.detach().clone() for leaf in h_grads]

    fla_error: str | None = None
    fla_grad_errors: dict[str, float] = {}
    try:
        fla_fn()
        torch.cuda.synchronize()
        for name, actual, expected in zip(
            harness.grad_tensors,
            helion_grad_values,
            fla_grads,
            strict=True,
        ):
            fla_grad_errors[name] = float(
                rel_error(actual, expected.grad.transpose(1, 2).contiguous())
            )
    except Exception as error:
        fla_error = f"{type(error).__name__}: {error}"
        for leaf in fla_grads:
            leaf.grad = None
        torch.cuda.empty_cache()

    reference_error: str | None = None
    reference_grad_errors: dict[str, float] = {}
    try:
        r_inputs, r_grads = _grad_leaves(harness, inputs)
        harness.chunked_reference(r_inputs, 64).backward(grad_out)
        torch.cuda.synchronize()
        reference_grad_errors = {
            name: float(rel_error(actual, expected.grad))
            for name, actual, expected in zip(
                harness.grad_tensors,
                helion_grad_values,
                r_grads,
                strict=True,
            )
        }
        del r_inputs, r_grads
    except Exception as error:
        reference_error = f"{type(error).__name__}: {error}"
        torch.cuda.empty_cache()

    shared_errors = reference_grad_errors or fla_grad_errors
    finite = all(bool(torch.isfinite(grad).all()) for grad in helion_grad_values)
    accuracy_ok = (
        finite and bool(shared_errors) and max(shared_errors.values()) < ACC_BWD_TOL
    )

    timing_labels = ["helion"]
    if fla_error is None:
        timing_labels.append("fla")
    if row_index % 2:
        timing_labels.reverse()
    timings: dict[str, float] = {}
    samples: dict[str, list[float]] = {}
    for timing_label in timing_labels:
        value, observed = _bench_ms(
            helion_fn if timing_label == "helion" else fla_fn,
            rounds,
            h_grads if timing_label == "helion" else fla_grads,
        )
        timings[timing_label] = value
        samples[timing_label] = observed

    result: dict[str, object] = {
        "cell_key": row["cell_key"],
        "variant": row["variant"],
        "mode": row["mode"],
        "shape_name": row["shape_name"],
        "arm": arm,
        "label": label,
        "accuracy_ok": accuracy_ok,
        "finite_gradients": finite,
        "relative_gradient_error_vs_reference": reference_grad_errors,
        "reference_error": reference_error,
        "relative_gradient_error_vs_fla": fla_grad_errors,
        "fla_error": fla_error,
        "helion_ms": timings["helion"],
        "fla_ms": timings.get("fla"),
        "samples_ms": samples,
    }
    if "fla" in timings:
        result["speedup_vs_fla"] = timings["fla"] / timings["helion"]
    return result


def _cell(
    row: dict[str, str],
    rounds: int,
    row_index: int,
    arm: str,
    label: str,
) -> dict[str, object]:
    result = (
        _forward_cell(row, rounds, row_index, arm, label)
        if row["mode"] == "forward"
        else _backward_cell(row, rounds, row_index, arm, label)
    )
    kernels = _kernel_metadata()
    if arm == "aot_tuned":
        result["aot_audit"] = _aot_audit(kernels)
    result["environment"] = _environment_metadata()
    result["invoked_kernels"] = kernels
    return result


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str))


def _one(args: argparse.Namespace) -> None:
    row = _rows(args.manifest, args.mode)[args.row]
    try:
        _set_arm_environment(args.arm)
        _load_runtime(args.helion_root, args.fla_root)
        _configure_arm(args.arm, args.aot_module)
        result = _cell(row, args.rounds, args.row, args.arm, args.label)
    except Exception as error:
        result = {
            "cell_key": row["cell_key"],
            "variant": row["variant"],
            "mode": row["mode"],
            "shape_name": row["shape_name"],
            "arm": args.arm,
            "label": args.label,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc()[-2400:],
        }
    _write(args.out, result)
    print(json.dumps(result), flush=True)


def _all(args: argparse.Namespace) -> None:
    rows = _rows(args.manifest, args.mode)
    prior: dict[str, dict[str, object]] = {}
    if args.resume and args.out.exists():
        prior = {row["cell_key"]: row for row in json.loads(args.out.read_text())}

    _set_arm_environment(args.arm)
    _load_runtime(args.helion_root, args.fla_root)
    _configure_arm(args.arm, args.aot_module)

    for index, row in enumerate(rows):
        if row["cell_key"] in prior:
            print(f"[{index + 1}/{len(rows)}] cached {row['cell_key']}", flush=True)
            continue
        try:
            result = _cell(row, args.rounds, index, args.arm, args.label)
        except Exception as error:
            result = {
                "cell_key": row["cell_key"],
                "variant": row["variant"],
                "mode": row["mode"],
                "shape_name": row["shape_name"],
                "arm": args.arm,
                "label": args.label,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc()[-2400:],
            }
        prior[row["cell_key"]] = result
        ordered = [prior[row["cell_key"]] for row in rows if row["cell_key"] in prior]
        _write(args.out, ordered)
        print(f"[{index + 1}/{len(rows)}] {row['cell_key']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("one", "all"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--helion-root",
        type=Path,
        help="Helion checkout (or set HELION_ROOT); omit for an installed package",
    )
    parser.add_argument(
        "--fla-root",
        type=Path,
        help="FLA checkout (or set FLA_ROOT); omit for an installed package",
    )
    parser.add_argument(
        "--aot-module",
        help=(
            "generated Helion AOT module; defaults to the module for the active "
            "CUDA architecture"
        ),
    )
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=("all", "forward", "forward_backward"),
        default="all",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="outer repetitions of Helion's existing do_bench call (default: 1)",
    )
    parser.add_argument(
        "--arm",
        choices=tuple(_ARM_ALIASES),
        default="seed",
    )
    parser.add_argument("--label")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.arm = _ARM_ALIASES[args.arm]
    args.label = args.label or args.arm
    args.manifest = args.manifest.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    args.helion_root = checkout_path(args.helion_root, "HELION_ROOT")
    args.fla_root = checkout_path(args.fla_root, "FLA_ROOT")
    args.aot_module = args.aot_module or os.environ.get("HELION_LINEAR_AOT_MODULE")
    if args.action == "one":
        _one(args)
    else:
        _all(args)


if __name__ == "__main__":
    main()
