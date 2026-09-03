"""Create higher-is-better tables from a combined benchmark result."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from result_utils import display_name
from result_utils import load_result
from result_utils import seed_speedup
from result_utils import variants


def _fmt(value: float | None, count: int) -> str:
    return "NA (0)" if value is None else f"{value:.4f}x ({count})"


def _row(label: str, cells: list[dict[str, Any]]) -> str:
    default, default_n = seed_speedup(cells, "default")
    fla, fla_n = seed_speedup(cells, "fla_triton")
    tuned, tuned_n = seed_speedup(cells, "aot_tuned")
    return (
        f"| {label} | {len(cells)} | {_fmt(default, default_n)} | "
        f"{_fmt(fla, fla_n)} | {_fmt(tuned, tuned_n)} |"
    )


def render(data: dict[str, Any]) -> str:
    cells = data["cells"]
    lines = [
        "# Linear-Attention Performance",
        "",
        (
            "Each Helion arm is normalized to FLA timed in the same process, "
            "then compared with seed; values above `1.0x` favor seed. Counts "
            "are correct common cells."
        ),
        "",
        "| Population | Discovered | Default / seed | FLA Triton / seed | AOT-tuned / seed |",
        "|---|---:|---:|---:|---:|",
        _row("Overall", cells),
    ]
    for mode in ("forward", "forward_backward"):
        selected = [cell for cell in cells if cell.get("mode") == mode]
        if selected:
            lines.append(_row(display_name(mode), selected))

    lines.extend(["", "## Per Kernel", ""])
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
