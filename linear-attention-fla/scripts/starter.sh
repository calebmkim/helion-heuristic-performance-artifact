#!/usr/bin/env bash
#
# STARTING POINT, NOT A STABLE BENCHMARK INTERFACE.
#
# Inspect the target Helion/FLA revisions first. Adapt these scripts when APIs
# or arm-selection semantics have changed, and retain that adapted copy with
# the results.

set -euo pipefail

: "${HELION_ROOT:?set HELION_ROOT to the Helion checkout}"
: "${OUTPUT_DIR:?set OUTPUT_DIR for raw and rendered results}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
ROUNDS="${ROUNDS:-1}"
RUN_AOT_TUNED="${RUN_AOT_TUNED:-1}"

mkdir -p "$OUTPUT_DIR"
MANIFEST="$OUTPUT_DIR/manifest.csv"

checkout_args=(--helion-root "$HELION_ROOT")
if [[ -n "${FLA_ROOT:-}" ]]; then
  checkout_args+=(--fla-root "$FLA_ROOT")
fi
aot_args=()
if [[ -n "${HELION_LINEAR_AOT_MODULE:-}" ]]; then
  aot_args+=(--aot-module "$HELION_LINEAR_AOT_MODULE")
fi

"$PYTHON_BIN" "$SCRIPT_DIR/discover_manifest.py" \
  --helion-root "$HELION_ROOT" \
  --output "$MANIFEST"

inputs=()
run_arm() {
  local arm="$1"
  local output="$OUTPUT_DIR/${arm}.json"
  "$PYTHON_BIN" "$SCRIPT_DIR/run_benchmark.py" \
    --action all \
    --manifest "$MANIFEST" \
    --arm "$arm" \
    --mode all \
    --rounds "$ROUNDS" \
    --out "$output" \
    --resume \
    "${checkout_args[@]}" \
    "${aot_args[@]}"
  inputs+=(--input "$arm=$output")
}

for arm in default seed; do
  run_arm "$arm"
done
if [[ "$RUN_AOT_TUNED" != "0" ]]; then
  run_arm aot_tuned
fi

COMBINED="$OUTPUT_DIR/results.json"
"$PYTHON_BIN" "$SCRIPT_DIR/combine_results.py" \
  --manifest "$MANIFEST" \
  "${inputs[@]}" \
  --output "$COMBINED"
"$PYTHON_BIN" "$SCRIPT_DIR/summarize_results.py" \
  "$COMBINED" \
  --output "$OUTPUT_DIR/summary.md"
"$PYTHON_BIN" "$SCRIPT_DIR/plot_results.py" \
  "$COMBINED" \
  --output "$OUTPUT_DIR/per-kernel-bars.png"

printf 'Results: %s\n' "$OUTPUT_DIR"
