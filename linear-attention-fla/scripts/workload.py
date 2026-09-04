"""Build one workload cell using Helion's linear-attention example harness."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any


@dataclass
class CellWorkload:
    row: dict[str, str]
    harness: Any
    inputs: Any
    fla_inputs: Any
    helion_grads: list[Any]
    fla_grads: list[Any]
    grad_out: Any | None
    fla_grad_out: Any | None
    varlen: bool

    @property
    def is_backward(self) -> bool:
        return self.row["mode"] == "forward_backward"

    def clear_helion_grads(self) -> None:
        for leaf in self.helion_grads:
            leaf.grad = None

    def clear_fla_grads(self) -> None:
        for leaf in self.fla_grads:
            leaf.grad = None

    def helion(self) -> Any:
        if self.is_backward:
            self.harness.helion_fb(self.inputs, self.grad_out, 64)
            return None
        return self.harness.helion_fwd(self.inputs, 64)

    def fla(self) -> Any:
        if self.is_backward:
            self.harness.fla_fb(
                self.fla_inputs,
                self.fla_grad_out,
                self.inputs.scale,
            )
            return None
        return self.harness.fla_fwd(self.fla_inputs, self.inputs.scale)

    def helion_gradient_values(self) -> list[Any]:
        return [leaf.grad.detach().clone() for leaf in self.helion_grads]

    def fla_gradient_values(self) -> list[Any]:
        return [
            leaf.grad.transpose(1, 2).contiguous().detach().clone()
            for leaf in self.fla_grads
        ]


def build_workload(row: dict[str, str]) -> CellWorkload:
    """Create inputs and callables exactly through the example harness."""
    import torch

    from examples.linear.linear_attention_engine import LinearAttentionVariant
    from examples.linear.linear_attention_harness import DTYPE
    from examples.linear.linear_attention_harness import LinearAttentionExampleHarness
    from examples.linear.linear_attention_harness import _fla_benchmark_inputs
    from examples.linear.linear_attention_harness import _fla_inputs
    from examples.linear.linear_attention_harness import _grad_leaves

    base_variant = row["variant"]
    fused = base_variant == "kda_fused"
    varlen = base_variant == "kda_varlen"
    variant_name = "kda" if fused or varlen else base_variant
    variant = next(
        item for item in LinearAttentionVariant if item.value == variant_name
    )
    harness = LinearAttentionExampleHarness(variant)

    torch.manual_seed(42)
    extra: dict[str, object] = {}
    if fused:
        extra["fused_preamble"] = True
    if varlen:
        lengths = json.loads(row["lengths_json"])
        shape = [len(lengths), int(row["heads"]), sum(lengths), int(row["dim"])]
        shape.append(shape[-1])
        extra.update(varlen=True, varlen_lengths=lengths)
    else:
        shape = json.loads(row["shape_json"])

    requires_grad = row["mode"] == "forward_backward"
    base_inputs = harness.make_inputs(
        *shape,
        dtype=DTYPE,
        device="cuda",
        requires_grad=requires_grad,
        **extra,
    )
    if requires_grad:
        inputs, helion_grads = _grad_leaves(harness, base_inputs)
        fla_inputs, fla_grads = _fla_benchmark_inputs(harness, base_inputs)
        grad_out = torch.randn(
            shape[0],
            shape[1],
            shape[2],
            shape[4],
            device="cuda",
            dtype=DTYPE,
        )
        fla_grad_out = grad_out.transpose(1, 2).contiguous()
    else:
        inputs = base_inputs
        helion_grads = []
        fla_inputs = _fla_inputs(base_inputs)
        fla_grads = []
        grad_out = None
        fla_grad_out = None

    return CellWorkload(
        row=row,
        harness=harness,
        inputs=inputs,
        fla_inputs=fla_inputs,
        helion_grads=helion_grads,
        fla_grads=fla_grads,
        grad_out=grad_out,
        fla_grad_out=fla_grad_out,
        varlen=varlen,
    )


def shape_record(row: dict[str, str]) -> dict[str, Any]:
    shape = json.loads(row.get("shape_json") or "null")
    if isinstance(shape, list) and len(shape) == 5:
        batch, heads, tokens, dim, value_dim = shape
        return {
            "name": row["shape_name"],
            "B": batch,
            "H": heads,
            "T": tokens,
            "D": dim,
            "DV": value_dim,
        }
    lengths = json.loads(row.get("lengths_json") or "null")
    if isinstance(lengths, list):
        dim = int(row["dim"])
        return {
            "name": row["shape_name"],
            "B": len(lengths),
            "H": int(row["heads"]),
            "T": sum(lengths),
            "D": dim,
            "DV": dim,
            "lengths": lengths,
        }
    return {"name": row["shape_name"]}
