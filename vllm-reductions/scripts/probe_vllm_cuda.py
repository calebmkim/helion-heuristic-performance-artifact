#!/usr/bin/env python3
"""Probe whether the optional vLLM CUDA comparison arm is available."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

from vllm_extension import extension_candidates
from vllm_extension import load_vllm_extension


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-root", type=Path)
    parser.add_argument("--vllm-extension", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.vllm_root.expanduser().resolve() if args.vllm_root else None
    explicit = (
        args.vllm_extension.expanduser().resolve()
        if args.vllm_extension
        else None
    )
    document: dict[str, object] = {
        "schema_version": 1,
        "available": False,
        "candidates": [
            str(path) for path in extension_candidates(explicit, root)
        ],
    }
    try:
        import torch

        extension = load_vllm_extension(torch, explicit, root)
        document.update(
            available=True,
            extension=extension,
            torch=torch.__version__,
            cuda_runtime=torch.version.cuda,
            cuda_visible_devices=__import__("os").environ.get(
                "CUDA_VISIBLE_DEVICES"
            ),
        )
        if torch.cuda.is_available():
            document["gpu"] = torch.cuda.get_device_name(0)
    except Exception as error:
        document.update(
            error=f"{type(error).__name__}: {error}",
            trace=traceback.format_exc(),
        )

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + "\n")
    print(
        f"vLLM CUDA extension available={document['available']}; "
        f"details: {output}"
    )
    raise SystemExit(0 if document["available"] else 1)


if __name__ == "__main__":
    main()
