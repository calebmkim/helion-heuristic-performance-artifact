#!/usr/bin/env python3
"""Locate and load vLLM's stable-libtorch CUDA operator extension."""

from __future__ import annotations

import hashlib
from importlib import metadata
from pathlib import Path
from typing import Any


REQUIRED_OPS = (
    "dynamic_per_token_scaled_fp8_quant",
    "per_token_group_fp8_quant",
    "rms_norm_dynamic_per_token_quant",
    "rms_norm_per_block_quant",
    "silu_and_mul_per_block_quant",
    "fused_qk_norm_rope",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution_provenance(path: Path) -> dict[str, str]:
    """Read wheel metadata beside an installed or fully extracted extension."""
    site_packages = path.parent.parent
    for metadata_path in sorted(site_packages.glob("vllm-*.dist-info/METADATA")):
        fields: dict[str, str] = {}
        try:
            for line in metadata_path.read_text(errors="replace").splitlines():
                if not line:
                    break
                name, separator, value = line.partition(":")
                if separator and name in {"Name", "Version"}:
                    fields[name] = value.strip()
        except OSError:
            continue
        if fields.get("Name", "").lower() == "vllm" and fields.get("Version"):
            return {
                "distribution": fields["Name"],
                "distribution_version": fields["Version"],
                "distribution_metadata": str(metadata_path.resolve()),
            }
    return {}


def extension_candidates(
    explicit: Path | None,
    vllm_root: Path | None,
) -> list[Path]:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit.expanduser().resolve())
    if vllm_root is not None:
        candidates.extend(
            sorted(
                (vllm_root.expanduser().resolve() / "vllm").glob(
                    "_C_stable_libtorch*.so"
                )
            )
        )

    try:
        distribution = metadata.distribution("vllm")
    except metadata.PackageNotFoundError:
        distribution = None
    if distribution is not None:
        for item in distribution.files or ():
            path = Path(str(item))
            if (
                path.parent.name == "vllm"
                and path.name.startswith("_C_stable_libtorch")
                and path.suffix == ".so"
            ):
                candidates.append(Path(distribution.locate_file(item)).resolve())

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def find_vllm_extension(
    explicit: Path | None,
    vllm_root: Path | None,
) -> Path:
    candidates = extension_candidates(explicit, vllm_root)
    for path in candidates:
        if path.is_file():
            return path
    rendered = ", ".join(str(path) for path in candidates) or "none"
    raise FileNotFoundError(
        "vLLM stable-libtorch extension not found; candidates: " + rendered
    )


def load_vllm_extension(
    torch: Any,
    explicit: Path | None,
    vllm_root: Path | None,
    *,
    include_sha256: bool = True,
) -> dict[str, object]:
    path = find_vllm_extension(explicit, vllm_root)
    torch.ops.load_library(str(path))
    missing = [name for name in REQUIRED_OPS if not hasattr(torch.ops._C, name)]
    if missing:
        raise RuntimeError(
            f"{path} did not register required torch.ops._C operators: {missing}"
        )
    result: dict[str, object] = {
        "path": str(path),
        "required_ops": list(REQUIRED_OPS),
    }
    result.update(_distribution_provenance(path))
    if include_sha256:
        result["sha256"] = _sha256(path)
    return result
