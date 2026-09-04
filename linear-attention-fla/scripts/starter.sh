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

mkdir -p "$OUTPUT_DIR"
MANIFEST="$OUTPUT_DIR/manifest.csv"
CONFIGS="$OUTPUT_DIR/config-replay.json"

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

"$PYTHON_BIN" "$SCRIPT_DIR/materialize_configs.py" \
  --manifest "$MANIFEST" \
  --out "$CONFIGS" \
  --resume \
  "${checkout_args[@]}" \
  "${aot_args[@]}"

COMBINED="$OUTPUT_DIR/results.json"
"$PYTHON_BIN" "$SCRIPT_DIR/run_benchmark.py" \
  --action all \
  --manifest "$MANIFEST" \
  --configs "$CONFIGS" \
  --mode all \
  --rounds "$ROUNDS" \
  --out "$COMBINED" \
  --resume \
  "${checkout_args[@]}"
"$PYTHON_BIN" "$SCRIPT_DIR/summarize_results.py" \
  "$COMBINED" \
  --output "$OUTPUT_DIR/summary.md"
"$PYTHON_BIN" "$SCRIPT_DIR/plot_results.py" \
  "$COMBINED" \
  --output "$OUTPUT_DIR/per-kernel-bars.png"

printf 'Results: %s\n' "$OUTPUT_DIR"
