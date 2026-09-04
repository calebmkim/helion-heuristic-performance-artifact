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
REPO_ROOT="$(cd -- "$ARTIFACT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MANIFEST="${MANIFEST:-$ARTIFACT_DIR/shapes.json}"
ROUNDS="${ROUNDS:-3}"
REPETITIONS="${REPETITIONS:-100}"
TORCH_COMPILE_MODE="${TORCH_COMPILE_MODE:-max-autotune-no-cudagraphs}"

"$PYTHON_BIN" "$REPO_ROOT/check_primary_stack.py" \
  --context "example reductions" \
  --helion-root "$HELION_ROOT"

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
