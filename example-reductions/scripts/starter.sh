#!/usr/bin/env bash
#
# STARTING POINT, NOT A STABLE BENCHMARK INTERFACE.
# Inspect and adapt it for the requested Helion revision.

set -euo pipefail

: "${HELION_ROOT:?set HELION_ROOT to the Helion checkout}"
: "${OUTPUT_DIR:?set OUTPUT_DIR for raw and rendered results}"
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to exactly one GPU}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MANIFEST="${MANIFEST:-$ARTIFACT_DIR/shapes.json}"
ROUNDS="${ROUNDS:-3}"
REPETITIONS="${REPETITIONS:-100}"
TORCH_COMPILE_MODE="${TORCH_COMPILE_MODE:-max-autotune-no-cudagraphs}"
REQUIRED_TORCH_VERSION="${REQUIRED_TORCH_VERSION:-2.13.0+cu132}"
REQUIRED_TRITON_VERSION="${REQUIRED_TRITON_VERSION:-3.7.1}"

"$PYTHON_BIN" - "$REQUIRED_TORCH_VERSION" "$REQUIRED_TRITON_VERSION" <<'PY'
import sys

import torch
import triton

required_torch, required_triton = sys.argv[1:]
observed = {
    "Torch": (torch.__version__, required_torch),
    "Triton": (triton.__version__, required_triton),
}
mismatches = [
    f"{name} {actual} (required {required})"
    for name, (actual, required) in observed.items()
    if actual != required
]
if mismatches:
    raise SystemExit(
        "Primary example-reduction stack mismatch: " + "; ".join(mismatches)
    )
print(
    f"Version check: Torch {torch.__version__}, Triton {triton.__version__}"
)
PY

mkdir -p "$OUTPUT_DIR"

benchmark_args=(
  --manifest "$MANIFEST"
  --helion-root "$HELION_ROOT"
  --rounds "$ROUNDS"
  --repetitions "$REPETITIONS"
  --torch-compile-mode "$TORCH_COMPILE_MODE"
  --output "$OUTPUT_DIR/benchmark.json"
  --resume
)
if [[ -n "${KERNELS:-}" ]]; then
  benchmark_args+=(--kernels "$KERNELS")
fi
if [[ -n "${LIMIT:-}" ]]; then
  benchmark_args+=(--limit "$LIMIT")
fi
"$PYTHON_BIN" "$SCRIPT_DIR/run_benchmark.py" "${benchmark_args[@]}"

"$PYTHON_BIN" "$SCRIPT_DIR/summarize_results.py" \
  --input "$OUTPUT_DIR/benchmark.json" \
  --json-output "$OUTPUT_DIR/summary.json" \
  --markdown-output "$OUTPUT_DIR/REPORT.md" \
  --csv-output "$OUTPUT_DIR/per_cell.csv"

"$PYTHON_BIN" "$SCRIPT_DIR/plot_results.py" \
  "$OUTPUT_DIR/summary.json" \
  --output "$OUTPUT_DIR/per-kernel-performance.png"

printf 'Results: %s\n' "$OUTPUT_DIR"
