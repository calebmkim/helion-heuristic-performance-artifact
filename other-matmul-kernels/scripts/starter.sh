#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COMPARISON_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd -- "$COMPARISON_ROOT/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
HELION_ROOT="${HELION_ROOT:?Set HELION_ROOT to the Helion checkout to measure}"
GPU="${GPU:-0}"
GPU_LABEL="${GPU_LABEL:-H100}"
ROUNDS="${ROUNDS:-3}"
INCLUDE_TORCH_COMPILE="${INCLUDE_TORCH_COMPILE:-1}"
TORCH_COMPILE_MODE="${TORCH_COMPILE_MODE:-max-autotune-no-cudagraphs}"
CELL_TIMEOUT="${CELL_TIMEOUT:-1800}"
DEFER_CELLS="${DEFER_CELLS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$COMPARISON_ROOT/generated/$(date -u +%Y%m%d-%H%M%S)}"

"$PYTHON_BIN" "$REPO_ROOT/scripts/check_primary_stack.py" \
  --context "other matmul kernels" \
  --helion-root "$HELION_ROOT"

mkdir -p "$OUTPUT_DIR"

RUN_ARGS=(
  --helion-root "$HELION_ROOT" \
  --manifest "$COMPARISON_ROOT/shapes.json" \
  --output "$OUTPUT_DIR/raw-results.json" \
  --gpu "$GPU" \
  --rounds "$ROUNDS" \
  --torch-compile-mode "$TORCH_COMPILE_MODE" \
  --cell-timeout "$CELL_TIMEOUT" \
  --resume
)
if [[ "$INCLUDE_TORCH_COMPILE" == "1" ]]; then
  RUN_ARGS+=(--torch-compile)
fi
if [[ -n "$DEFER_CELLS" ]]; then
  RUN_ARGS+=(--defer-cells "$DEFER_CELLS")
fi

"$PYTHON_BIN" "$SCRIPT_DIR/run_benchmark.py" "${RUN_ARGS[@]}"

"$PYTHON_BIN" "$SCRIPT_DIR/summarize_results.py" \
  "$OUTPUT_DIR/raw-results.json" \
  --json-out "$OUTPUT_DIR/results.json" \
  --markdown-out "$OUTPUT_DIR/summary.md" \
  --csv-out "$OUTPUT_DIR/per-shape.csv"

"$PYTHON_BIN" "$SCRIPT_DIR/plot_results.py" \
  "$OUTPUT_DIR/results.json" \
  --output "$OUTPUT_DIR/per-family-bars.png" \
  --baseline default

"$PYTHON_BIN" "$SCRIPT_DIR/plot_blog_figures.py" \
  --summary "$OUTPUT_DIR/results.json" \
  --gpu "$GPU_LABEL" \
  --outdir "$OUTPUT_DIR/blog-figures" \
  --baseline default

if [[ "$INCLUDE_TORCH_COMPILE" == "1" ]]; then
  "$PYTHON_BIN" "$SCRIPT_DIR/plot_results.py" \
    "$OUTPUT_DIR/results.json" \
    --output "$OUTPUT_DIR/per-family-vs-torch-compile-bars.png" \
    --baseline torch_compile

  "$PYTHON_BIN" "$SCRIPT_DIR/plot_blog_figures.py" \
    --summary "$OUTPUT_DIR/results.json" \
    --gpu "$GPU_LABEL" \
    --outdir "$OUTPUT_DIR/blog-figures" \
    --baseline torch_compile
fi

printf 'Wrote %s\n' "$OUTPUT_DIR"
