#!/usr/bin/env python3
"""Fail fast unless a benchmark uses the repository's primary software stack."""

from __future__ import annotations

import argparse
import os

PRIMARY_TORCH_VERSION = "2.13.0+cu132"
PRIMARY_CUDA_VERSION = "13.2"
PRIMARY_TRITON_VERSION = "3.7.1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", default="benchmark")
    parser.add_argument(
        "--torch-version",
        default=os.environ.get("REQUIRED_TORCH_VERSION", PRIMARY_TORCH_VERSION),
    )
    parser.add_argument(
        "--cuda-version",
        default=os.environ.get("REQUIRED_CUDA_VERSION", PRIMARY_CUDA_VERSION),
    )
    parser.add_argument(
        "--triton-version",
        default=os.environ.get(
            "REQUIRED_TRITON_VERSION", PRIMARY_TRITON_VERSION
        ),
    )
    args = parser.parse_args()

    import torch
    import triton

    observed = {
        "PyTorch": (torch.__version__, args.torch_version),
        "CUDA": (torch.version.cuda, args.cuda_version),
        "Triton": (triton.__version__, args.triton_version),
    }
    mismatches = [
        f"{name} {actual} (required {required})"
        for name, (actual, required) in observed.items()
        if actual != required
    ]
    if mismatches:
        raise SystemExit(
            f"Primary stack mismatch for {args.context}: "
            + "; ".join(mismatches)
        )
    print(
        f"Primary stack check ({args.context}): "
        f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}, "
        f"Triton {triton.__version__}"
    )


if __name__ == "__main__":
    main()
