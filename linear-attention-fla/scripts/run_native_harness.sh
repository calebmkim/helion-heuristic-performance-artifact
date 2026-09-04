#!/usr/bin/env bash
#
# THIN NATIVE-HARNESS STARTING POINT, NOT A STABLE BENCHMARK INTERFACE.
#
# This intentionally does not select a Helion config or alter the native
# benchmark methodology. Pin the target revisions and configuration environment
# before using its output to reproduce a claim.

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
"$PYTHON_BIN" "$REPO_ROOT/scripts/check_primary_stack.py" \
  --context "linear attention native harness" \
  --helion-root "$HELION_ROOT"

NATIVE_MODULE="${HELION_LINATTN_MODULE:-benchmarks.run_linattn}"
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
  PYTHONPATH="$python_path" "$PYTHON_BIN" -m "$NATIVE_MODULE" \
    --output "$OUTPUT_FILE" \
    "$@"
)

if [[ -e "$OUTPUT_FILE" ]]; then
  printf 'Native Helion result: %s\n' "$OUTPUT_FILE"
fi
