#!/usr/bin/env bash
#
# NATIVE-HARNESS STARTING POINT, NOT A STABLE BENCHMARK INTERFACE.
#
# This preserves the native benchmark methodology and defaults to the checked-in
# AOT selector used by Helion's current linear-attention benchmark workflow.

set -euo pipefail

: "${HELION_ROOT:?set HELION_ROOT to the Helion checkout}"
: "${OUTPUT_DIR:?set OUTPUT_DIR for the native result}"
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to exactly one GPU}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
HELION_ROOT="$(cd -- "$HELION_ROOT" && pwd)"
if [[ -n "${FLA_ROOT:-}" ]]; then
  FLA_ROOT="$(cd -- "$FLA_ROOT" && pwd)"
fi
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd -- "$OUTPUT_DIR" && pwd)"

PYTHON_BIN="$(command -v "${PYTHON_BIN:-python}")"
"$PYTHON_BIN" "$REPO_ROOT/check_primary_stack.py" \
  --context "linear attention native harness" \
  --helion-root "$HELION_ROOT"

NATIVE_MODULE="${HELION_LINATTN_MODULE:-benchmarks.run_linattn}"
NATIVE_AUTOTUNE_CACHE="${HELION_AUTOTUNE_CACHE:-AOTAutotuneCache}"
NATIVE_CONFIG_LABEL="${NATIVE_CONFIG_LABEL:-H100 AOT (compat-repaired)}"
OUTPUT_FILE="${OUTPUT_FILE:-$OUTPUT_DIR/helionbench.json}"
if [[ "$OUTPUT_FILE" != /* ]]; then
  OUTPUT_FILE="$PWD/$OUTPUT_FILE"
fi

mkdir -p "$(dirname -- "$OUTPUT_FILE")"

python_path="$HELION_ROOT"
if [[ -n "${FLA_ROOT:-}" ]]; then
  python_path="$python_path:$FLA_ROOT"
fi
if [[ -n "${PYTHONPATH:-}" ]]; then
  python_path="$python_path:$PYTHONPATH"
fi

(
  cd -- "$HELION_ROOT"
  PYTHONPATH="$python_path" \
  HELION_AUTOTUNE_CACHE="$NATIVE_AUTOTUNE_CACHE" \
  "$PYTHON_BIN" "$SCRIPT_DIR/run_native_module.py" "$NATIVE_MODULE" \
    --output "$OUTPUT_FILE" \
    "$@"
)

if [[ -e "$OUTPUT_FILE" ]]; then
  render_args=(
    "$OUTPUT_FILE"
    --summary "$OUTPUT_DIR/summary.md"
    --plot "$OUTPUT_DIR/per-kernel-bars.png"
    --provenance "$OUTPUT_DIR/provenance.json"
    --helion-root "$HELION_ROOT"
    --config-label "$NATIVE_CONFIG_LABEL"
    --aot-compat-repair
  )
  if [[ -n "${FLA_ROOT:-}" ]]; then
    render_args+=(--fla-root "$FLA_ROOT")
  fi
  "$PYTHON_BIN" "$SCRIPT_DIR/render_native_results.py" "${render_args[@]}"
  printf 'Native Helion result: %s\n' "$OUTPUT_FILE"
  printf 'Native summary: %s\n' "$OUTPUT_DIR/summary.md"
  printf 'Native graph: %s\n' "$OUTPUT_DIR/per-kernel-bars.png"
fi
