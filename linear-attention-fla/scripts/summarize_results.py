"""Create higher-is-better tables from a combined benchmark result."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from result_utils import display_name
from result_utils import load_result
from result_utils import relative_to_fla
from result_utils import seed_speedup
from result_utils import variants


def _fmt(value: float | None, count: int) -> str:
    return "NA (0)" if value is None else f"{value:.4f}x ({count})"


def _relative_row(label: str, cells: list[dict[str, Any]]) -> str:
    default, default_n = relative_to_fla(cells, "default")
    fla, fla_n = relative_to_fla(cells, "fla_triton")
    seed, seed_n = relative_to_fla(cells, "seed")
    tuned, tuned_n = relative_to_fla(cells, "aot_tuned")
    return (
        f"| {label} | {len(cells)} | {_fmt(default, default_n)} | "
        f"{_fmt(fla, fla_n)} | {_fmt(seed, seed_n)} | "
        f"{_fmt(tuned, tuned_n)} |"
    )


def _seed_row(label: str, cells: list[dict[str, Any]]) -> str:
    default, default_n = seed_speedup(cells, "default")
    fla, fla_n = seed_speedup(cells, "fla_triton")
    tuned, tuned_n = seed_speedup(cells, "aot_tuned")
    return (
        f"| {label} | {len(cells)} | {_fmt(default, default_n)} | "
        f"{_fmt(fla, fla_n)} | {_fmt(tuned, tuned_n)} |"
    )


def _run_metadata(data: dict[str, Any]) -> list[str]:
    cells = data["cells"]
    complete = sum(
        all(
            cell.get("arms", {}).get(arm, {}).get("status") == "ok"
            for arm in ("default", "seed", "aot_tuned", "fla_triton")
        )
        for cell in cells
    )
    helion_arms = ("default", "seed", "aot_tuned")
    correct = sum(
        cell.get("arms", {}).get(arm, {}).get("correct") is True
        for cell in cells
        for arm in helion_arms
    )
    lines = [
        (
            f"Validation: {complete}/{len(cells)} cells completed all four "
            f"arms; {correct}/{len(cells) * len(helion_arms)} Helion "
            "arm-cells passed correctness."
        )
    ]
    environment = data.get("provenance", {}).get("environment", {})
    if not isinstance(environment, dict):
        return lines
    details = []
    if gpu := environment.get("gpu"):
        details.append(str(gpu))
    if revision := environment.get("helion_revision"):
        details.append(f"Helion `{str(revision)[:12]}`")
    if version := environment.get("fla_version"):
        fla = f"FLA `{version}`"
        if revision := environment.get("fla_revision"):
            fla += f" (`{str(revision)[:12]}`)"
        details.append(fla)
    if details:
        lines.append("Environment: " + ", ".join(details) + ".")
    return lines


def render(data: dict[str, Any]) -> str:
    cells = data["cells"]
    lines = [
        "# Linear-Attention Performance",
        "",
        (
            "All arms in a cell share one process and one FLA timing. Cold-L2 "
            "CUDA-event samples are interleaved in rotated forward/reverse "
            "order with equal counts."
        ),
        "",
        *_run_metadata(data),
        "",
        (
            "Values are higher-is-better geometric means; counts are correct "
            "common cells."
        ),
        "",
        "## Performance vs FLA",
        "",
        "| Population | Discovered | Default | FLA Triton | Seed | AOT-tuned |",
        "|---|---:|---:|---:|---:|---:|",
        _relative_row("Overall", cells),
    ]
    for mode in ("forward", "forward_backward"):
        selected = [cell for cell in cells if cell.get("mode") == mode]
        if selected:
            lines.append(_relative_row(display_name(mode), selected))

    lines.extend(["", "### Per Kernel", ""])
    lines.extend(
        [
            "| Kernel | Mode | Shapes | Default | FLA Triton | Seed | AOT-tuned |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for variant in variants(cells):
        for mode in ("forward", "forward_backward"):
            selected = [
                cell
                for cell in cells
                if cell.get("variant") == variant and cell.get("mode") == mode
            ]
            if selected:
                lines.append(
                    _relative_row(
                        f"{display_name(variant)} | {display_name(mode)}",
                        selected,
                    )
                )

    lines.extend(
        [
            "",
            "## Seed Comparisons",
            "",
            "Values above `1.0x` favor the heuristic seed.",
            "",
            "| Population | Discovered | Default / seed | FLA Triton / seed | AOT-tuned / seed |",
            "|---|---:|---:|---:|---:|",
            _seed_row("Overall", cells),
        ]
    )
    for mode in ("forward", "forward_backward"):
        selected = [cell for cell in cells if cell.get("mode") == mode]
        if selected:
            lines.append(_seed_row(display_name(mode), selected))

    lines.extend(["", "### Per Kernel", ""])
    lines.extend(
        [
            "| Kernel | Mode | Shapes | Default / seed | FLA Triton / seed | AOT-tuned / seed |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for variant in variants(cells):
        for mode in ("forward", "forward_backward"):
            selected = [
                cell
                for cell in cells
                if cell.get("variant") == variant and cell.get("mode") == mode
            ]
            if not selected:
                continue
            default, default_n = seed_speedup(selected, "default")
            fla, fla_n = seed_speedup(selected, "fla_triton")
            tuned, tuned_n = seed_speedup(selected, "aot_tuned")
            lines.append(
                f"| {display_name(variant)} | {display_name(mode)} | "
                f"{len(selected)} | {_fmt(default, default_n)} | "
                f"{_fmt(fla, fla_n)} | {_fmt(tuned, tuned_n)} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rendered = render(load_result(args.input))
    if args.output is None:
        print(rendered, end="")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)


if __name__ == "__main__":
    main()
