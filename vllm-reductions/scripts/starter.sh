#!/usr/bin/env bash
#
# STARTING POINT, NOT A STABLE BENCHMARK INTERFACE.
# Inspect and adapt it for the requested Helion and vLLM revisions.

set -euo pipefail

: "${HELION_ROOT:?set HELION_ROOT to the Helion checkout}"
: "${VLLM_ROOT:?set VLLM_ROOT to the vLLM checkout}"
: "${OUTPUT_DIR:?set OUTPUT_DIR for raw and rendered results}"
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to exactly one GPU}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PROFILE="${PROFILE:-curated}"
ROUNDS="${ROUNDS:-3}"
REPETITIONS="${REPETITIONS:-100}"
REFERENCE="${REFERENCE:-auto}"
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
        "Primary vLLM-reduction stack mismatch: " + "; ".join(mismatches)
    )
print(
    f"Version check: Torch {torch.__version__}, Triton {triton.__version__}"
)
PY

mkdir -p "$OUTPUT_DIR"

discover_args=(
  --helion-root "$HELION_ROOT"
  --profile "$PROFILE"
  --output "$OUTPUT_DIR/manifest.json"
)
if [[ -n "${COMPUTE_CAPABILITY:-}" ]]; then
  discover_args+=(--compute "$COMPUTE_CAPABILITY")
fi
if [[ -n "${KERNELS:-}" ]]; then
  discover_args+=(--kernels "$KERNELS")
fi

"$PYTHON_BIN" "$SCRIPT_DIR/inspect_vllm.py" \
  --vllm-root "$VLLM_ROOT" \
  --output "$OUTPUT_DIR/vllm-implementations.json" \
  --markdown-output "$OUTPUT_DIR/vllm-implementations.md"

"$PYTHON_BIN" "$SCRIPT_DIR/discover_manifest.py" "${discover_args[@]}"

probe_args=(
  --vllm-root "$VLLM_ROOT"
  --output "$OUTPUT_DIR/vllm-extension.json"
)
if [[ -n "${VLLM_EXTENSION_PATH:-}" ]]; then
  probe_args+=(--vllm-extension "$VLLM_EXTENSION_PATH")
fi

cuda_available=0
if "$PYTHON_BIN" "$SCRIPT_DIR/probe_vllm_cuda.py" "${probe_args[@]}"; then
  cuda_available=1
fi

case "$REFERENCE" in
  auto)
    if ((cuda_available)); then
      reference_arm=vllm_cuda
    else
      reference_arm=aot_tuned
    fi
    ;;
  vllm_cuda)
    if ((!cuda_available)); then
      printf 'REFERENCE=vllm_cuda requested, but its extension is unavailable.\n' >&2
      exit 1
    fi
    reference_arm=vllm_cuda
    ;;
  aot_tuned)
    reference_arm=aot_tuned
    ;;
  *)
    printf 'REFERENCE must be auto, vllm_cuda, or aot_tuned; got %s\n' "$REFERENCE" >&2
    exit 1
    ;;
esac

benchmark_args=(
  --manifest "$OUTPUT_DIR/manifest.json"
  --helion-root "$HELION_ROOT"
  --vllm-root "$VLLM_ROOT"
  --rounds "$ROUNDS"
  --repetitions "$REPETITIONS"
  --output "$OUTPUT_DIR/benchmark.json"
  --resume
)
if [[ -n "${VLLM_EXTENSION_PATH:-}" ]]; then
  benchmark_args+=(--vllm-extension "$VLLM_EXTENSION_PATH")
fi
if [[ -n "${KERNELS:-}" ]]; then
  benchmark_args+=(--kernels "$KERNELS")
fi
if [[ -n "${LIMIT:-}" ]]; then
  benchmark_args+=(--limit "$LIMIT")
fi
if [[ "$reference_arm" == vllm_cuda ]]; then
  benchmark_args+=(--include-vllm-cuda)
fi
"$PYTHON_BIN" "$SCRIPT_DIR/run_benchmark.py" "${benchmark_args[@]}"

summary_args=(
  --input "$OUTPUT_DIR/benchmark.json" \
  --extension-metadata "$OUTPUT_DIR/vllm-extension.json" \
  --json-output "$OUTPUT_DIR/summary.json" \
  --markdown-output "$OUTPUT_DIR/REPORT.md" \
  --csv-output "$OUTPUT_DIR/per_cell.csv"
)
"$PYTHON_BIN" "$SCRIPT_DIR/summarize_results.py" "${summary_args[@]}"

"$PYTHON_BIN" "$SCRIPT_DIR/plot_results.py" \
  "$OUTPUT_DIR/summary.json" \
  --output "$OUTPUT_DIR/per-kernel-performance.png"

printf 'Results: %s\n' "$OUTPUT_DIR"
