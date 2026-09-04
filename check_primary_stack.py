#!/usr/bin/env python3
"""Fail fast unless a benchmark uses the repository's primary revisions."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PRIMARY_HELION_COMMIT = "fa2f62eb686ef846c76f8b9e18beec30fbc5bee1"
PRIMARY_TORCH_VERSION = "2.13.0+cu132"
PRIMARY_CUDA_VERSION = "13.2"
PRIMARY_TRITON_VERSION = "3.7.1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", default="benchmark")
    parser.add_argument(
        "--helion-root",
        type=Path,
        default=os.environ.get("HELION_ROOT"),
    )
    parser.add_argument(
        "--helion-commit",
        default=os.environ.get(
            "REQUIRED_HELION_COMMIT", PRIMARY_HELION_COMMIT
        ),
    )
    parser.add_argument(
        "--allow-dirty-helion",
        action="store_true",
        default=os.environ.get("ALLOW_DIRTY_HELION") == "1",
    )
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
    if args.helion_root is None:
        parser.error("--helion-root or HELION_ROOT is required")

    helion_root = args.helion_root.expanduser().resolve()
    try:
        helion_commit = subprocess.check_output(
            ["git", "-C", str(helion_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        helion_dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(helion_root), "status", "--porcelain"],
                text=True,
                stderr=subprocess.STDOUT,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(
            f"Cannot inspect Helion checkout {helion_root}: {error}"
        ) from error

    sys.path.insert(0, str(helion_root))

    import helion
    import torch
    import triton

    helion_file = Path(helion.__file__).resolve()

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
    if helion_commit != args.helion_commit:
        mismatches.append(
            f"Helion {helion_commit} (required {args.helion_commit})"
        )
    if not helion_file.is_relative_to(helion_root):
        mismatches.append(
            f"Helion imported from {helion_file} "
            f"(required checkout {helion_root})"
        )
    if helion_dirty and not args.allow_dirty_helion:
        mismatches.append(
            "Helion worktree has local changes "
            "(set ALLOW_DIRTY_HELION=1 only for a labeled non-primary run)"
        )
    if mismatches:
        raise SystemExit(
            f"Primary stack mismatch for {args.context}: "
            + "; ".join(mismatches)
        )
    print(
        f"Primary stack check ({args.context}): "
        f"Helion {helion_commit[:8]}, "
        f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}, "
        f"Triton {triton.__version__}"
    )


if __name__ == "__main__":
    main()
