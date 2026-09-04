#!/usr/bin/env bash
#
# CONTROLLED FOUR-ARM STARTING POINT, NOT A STABLE BENCHMARK INTERFACE.
#
# Inspect the target Helion/FLA revisions first. Adapt these scripts when APIs
# or arm-selection semantics have changed, and retain that adapted copy with
# the results.

set -euo pipefail

: "${HELION_ROOT:?set HELION_ROOT to the Helion checkout}"
: "${OUTPUT_DIR:?set OUTPUT_DIR for raw and rendered results}"
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to exactly one GPU}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
ROUNDS="${ROUNDS:-1}"
RESUME="${RESUME:-0}"

"$PYTHON_BIN" "$REPO_ROOT/check_primary_stack.py" \
  --context "linear attention" \
  --helion-root "$HELION_ROOT"

mkdir -p "$OUTPUT_DIR"
MANIFEST="$OUTPUT_DIR/manifest.csv"
CONFIGS="$OUTPUT_DIR/config-replay.json"

discover_args=(
  --helion-root "$HELION_ROOT"
  --output "$MANIFEST"
)
if [[ -n "${VARIANTS:-}" ]]; then
  discover_args+=(--variants "$VARIANTS")
fi
if [[ -n "${SHAPE_NAMES:-}" ]]; then
  discover_args+=(--shape-names "$SHAPE_NAMES")
fi
if [[ -n "${MODES:-}" ]]; then
  discover_args+=(--modes "$MODES")
fi
if [[ -n "${LIMIT:-}" ]]; then
  discover_args+=(--limit "$LIMIT")
fi

checkout_args=(--helion-root "$HELION_ROOT")
if [[ -n "${FLA_ROOT:-}" ]]; then
  checkout_args+=(--fla-root "$FLA_ROOT")
fi
aot_args=()
if [[ -n "${HELION_LINEAR_AOT_MODULE:-}" ]]; then
  aot_args+=(--aot-module "$HELION_LINEAR_AOT_MODULE")
fi
resume_args=()
case "$RESUME" in
  0) ;;
  1) resume_args+=(--resume) ;;
  *) printf 'RESUME must be 0 or 1\n' >&2; exit 2 ;;
esac

"$PYTHON_BIN" "$SCRIPT_DIR/discover_manifest.py" "${discover_args[@]}"

"$PYTHON_BIN" "$SCRIPT_DIR/materialize_configs.py" \
  --manifest "$MANIFEST" \
  --out "$CONFIGS" \
  "${checkout_args[@]}" \
  "${aot_args[@]}" \
  "${resume_args[@]}"

COMBINED="$OUTPUT_DIR/results.json"
"$PYTHON_BIN" "$SCRIPT_DIR/run_benchmark.py" \
  --action all \
  --manifest "$MANIFEST" \
  --configs "$CONFIGS" \
  --mode all \
  --rounds "$ROUNDS" \
  --out "$COMBINED" \
  "${checkout_args[@]}" \
  "${resume_args[@]}"
"$PYTHON_BIN" "$SCRIPT_DIR/summarize_results.py" \
  "$COMBINED" \
  --output "$OUTPUT_DIR/summary.md"
"$PYTHON_BIN" "$SCRIPT_DIR/plot_results.py" \
  "$COMBINED" \
  --output "$OUTPUT_DIR/per-kernel-bars.png"
"$PYTHON_BIN" "$SCRIPT_DIR/plot_blog_figures.py" \
  --h100 "$COMBINED" \
  --outdir "$OUTPUT_DIR/blog-figures"

printf 'Results: %s\n' "$OUTPUT_DIR"
