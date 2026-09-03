#!/usr/bin/env python3
"""Inspect the vLLM source backing each external reduction baseline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess


@dataclass(frozen=True)
class Implementation:
    kernel: str
    op: str
    source: str
    symbol: str
    note: str = ""


IMPLEMENTATIONS = (
    Implementation(
        "dynamic_per_token_scaled_fp8_quant",
        "torch.ops._C.dynamic_per_token_scaled_fp8_quant",
        "csrc/libtorch_stable/quantization/w8a8/fp8/common.cu",
        "dynamic_per_token_scaled_fp8_quant",
    ),
    Implementation(
        "per_token_group_fp8_quant",
        "torch.ops._C.per_token_group_fp8_quant",
        "csrc/libtorch_stable/quantization/w8a8/fp8/per_token_group_quant.cu",
        "per_token_group_quant_fp8",
        "The public Python utility has a Triton fallback, but contiguous NVIDIA "
        "inputs prefer this compiled operator.",
    ),
    Implementation(
        "rms_norm_dynamic_per_token_quant",
        "torch.ops._C.rms_norm_dynamic_per_token_quant",
        "csrc/libtorch_stable/quantization/fused_kernels/"
        "fused_layernorm_dynamic_per_token_quant.cu",
        "rms_norm_dynamic_per_token_quant",
    ),
    Implementation(
        "rms_norm_per_block_quant",
        "torch.ops._C.rms_norm_per_block_quant",
        "csrc/libtorch_stable/quantization/fused_kernels/"
        "fused_layernorm_dynamic_per_token_quant.cu",
        "rms_norm_per_block_quant",
    ),
    Implementation(
        "silu_and_mul_per_block_quant",
        "torch.ops._C.silu_and_mul_per_block_quant",
        "csrc/libtorch_stable/quantization/fused_kernels/"
        "fused_silu_mul_block_quant.cu",
        "silu_and_mul_per_block_quant",
    ),
    Implementation(
        "fused_qk_norm_rope",
        "torch.ops._C.fused_qk_norm_rope",
        "csrc/libtorch_stable/fused_qknorm_rope_kernel.cu",
        "fused_qk_norm_rope",
        "The CUDA kernel is derived from TensorRT-LLM.",
    ),
)


def _git_metadata(root: Path) -> dict[str, object]:
    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()

    try:
        return {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "dirty": bool(git("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "dirty": None}


def inspect(root: Path) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for implementation in IMPLEMENTATIONS:
        source = root / implementation.source
        exists = source.is_file()
        contains_symbol = exists and implementation.symbol in source.read_text()
        helion_source = (
            root
            / "vllm"
            / "kernels"
            / "helion"
            / "ops"
            / f"{implementation.kernel}.py"
        )
        records.append(
            {
                "kernel": implementation.kernel,
                "entry_point": implementation.op,
                "backend": "cuda_cpp_extension",
                "source": str(source),
                "source_exists": exists,
                "symbol_found": contains_symbol,
                "optional_vllm_helion_override": (
                    str(helion_source) if helion_source.is_file() else None
                ),
                "note": implementation.note,
            }
        )

    return {
        "schema_version": 1,
        "vllm_root": str(root),
        "vllm_git": _git_metadata(root),
        "all_expected_sources_found": all(
            item["source_exists"] and item["symbol_found"] for item in records
        ),
        "implementations": records,
    }


def _markdown(data: dict[str, object]) -> str:
    git = data["vllm_git"]
    assert isinstance(git, dict)
    lines = [
        "# vLLM Reduction Implementation Inspection",
        "",
        f"- Revision: `{git.get('commit')}`",
        f"- Dirty checkout: `{git.get('dirty')}`",
        f"- All expected sources found: `{data['all_expected_sources_found']}`",
        "",
        "| Kernel | Entry point | Backend | Native source | Helion override |",
        "|---|---|---|---|---|",
    ]
    for item in data["implementations"]:  # type: ignore[union-attr]
        override = "yes" if item["optional_vllm_helion_override"] else "no"
        source = Path(str(item["source"])).name
        lines.append(
            f"| `{item['kernel']}` | `{item['entry_point']}` | CUDA C++ | "
            f"`{source}` | {override} |"
        )

    notes = [
        str(item["note"])
        for item in data["implementations"]  # type: ignore[union-attr]
        if item["note"]
    ]
    if notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in notes)
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()

    root = args.vllm_root.expanduser().resolve()
    data = inspect(root)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2) + "\n")
    if args.markdown_output:
        markdown = args.markdown_output.expanduser().resolve()
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(_markdown(data))
    print(
        f"Inspected {len(IMPLEMENTATIONS)} vLLM implementations at "
        f"{data['vllm_git']['commit']}: "
        f"all_sources_found={data['all_expected_sources_found']}"
    )


if __name__ == "__main__":
    main()
