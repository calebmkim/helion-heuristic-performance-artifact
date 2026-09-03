from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any
from typing import Callable

import torch

DEVICE = torch.device("cuda")


@dataclass(frozen=True)
class Workload:
    kernel_fn: Any
    args: tuple[Any, ...]
    body: str
    reference_fn: Callable[..., Any]
    reference_name: str
    inductor_options: dict[str, object] | None = None


def _module(name: str) -> Any:
    return importlib.import_module(name)


def _dtype(name: str | None) -> torch.dtype:
    return {
        None: torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]


def _generator() -> torch.Generator:
    return torch.Generator(device=DEVICE).manual_seed(0)


def _rand(
    shape: tuple[int, ...],
    dtype: torch.dtype = torch.bfloat16,
    *,
    scale: float = 0.02,
) -> torch.Tensor:
    result = torch.empty(shape, device=DEVICE, dtype=dtype)
    return result.normal_(0.0, scale, generator=_generator())


def _positive(
    shape: tuple[int, ...],
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    result = torch.empty(shape, device=DEVICE, dtype=dtype)
    return result.uniform_(0.005, 0.02, generator=_generator())


def _decay(
    batch: int,
    heads: int,
    chunks: int,
    chunk_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    values = -0.01 * torch.arange(chunk_size, device=DEVICE, dtype=torch.float32)
    return (
        values.view(1, 1, 1, chunk_size)
        .expand(batch, heads, chunks, chunk_size)
        .to(dtype)
        .contiguous()
    )


def _gather_reference(
    w: torch.Tensor,
    idx: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    return torch.matmul(w[idx.to(torch.int64)].to(x.dtype), x)


def _attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.matmul(q, k.transpose(-2, -1)).float() / math.sqrt(q.size(-1))
    if bias is not None:
        scores = scores + bias
    if causal:
        mask = torch.ones(
            scores.size(-2),
            scores.size(-1),
            dtype=torch.bool,
            device=scores.device,
        ).tril()
        scores = scores.masked_fill(~mask, -torch.inf)
    probabilities = torch.softmax(scores, dim=-1).to(v.dtype)
    output = torch.matmul(probabilities, v)
    lse = torch.logsumexp(scores, dim=-1)
    if bias is None:
        lse = lse * math.log2(math.e)
    return output, lse


def _dense_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _attention_reference(q, k, v)


def _causal_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _attention_reference(q, k, v, causal=True)


def _biased_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _attention_reference(q, k, v, bias)


def _attention_backward_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    _o: torch.Tensor,
    lse: torch.Tensor,
    do: torch.Tensor,
    delta: torch.Tensor,
    sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (-1, q.size(-2), q.size(-1))
    q_view = q.reshape(shape)
    k_view = k.reshape(shape)
    v_view = v.reshape(shape)
    do_view = do.reshape(shape)
    lse_view = lse.reshape(-1, q.size(-2))
    delta_view = delta.reshape(-1, q.size(-2))

    scores_t = torch.bmm(k_view, q_view.transpose(1, 2)).float()
    p_t = torch.exp2(scores_t - lse_view[:, None, :])
    dp_t = torch.bmm(v_view, do_view.transpose(1, 2)).float()
    ds_t = (p_t * (dp_t - delta_view[:, None, :])).to(q.dtype)
    dq = torch.bmm(ds_t.transpose(1, 2), k_view).float() * math.log(2.0)
    dk = (torch.bmm(ds_t, q_view).float() * sm_scale).to(k.dtype)
    dv = torch.bmm(p_t.to(v.dtype), do_view).to(v.dtype)
    return dq.reshape(q.shape), dk.reshape(k.shape), dv.reshape(v.shape)


def _squeeze_excitation_reference(
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    c = torch.relu(x @ a)
    d = torch.sigmoid(c @ b)
    return x * d, c, d


def _gdn_reference(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    batch, seqlen, heads, dim = k.shape
    chunks = seqlen // chunk_size
    state_dim = u.size(-1)
    batch_heads = batch * heads
    dtype = k.dtype

    # Put batch and head next to each other once. The recurrent loop can then
    # use BF16 tensor-core BMMs and keep the surrounding state math in FP32.
    k_chunks = (
        k.permute(0, 2, 1, 3)
        .contiguous()
        .reshape(batch_heads, chunks, chunk_size, dim)
    )
    w_chunks = (
        w.permute(0, 2, 1, 3)
        .contiguous()
        .reshape(batch_heads, chunks, chunk_size, dim)
    )
    u_chunks = (
        u.permute(0, 2, 1, 3)
        .contiguous()
        .reshape(batch_heads, chunks, chunk_size, state_dim)
    )
    g_chunks = (
        g.permute(0, 2, 1)
        .contiguous()
        .reshape(batch_heads, chunks, chunk_size)
    )
    g_last = g_chunks[:, :, -1]
    chunk_decay = torch.exp(g_last)
    token_decay = torch.exp(g_last[:, :, None] - g_chunks)

    state = torch.zeros(
        batch_heads,
        dim,
        state_dim,
        dtype=torch.float32,
        device=k.device,
    )

    def step(
        carry: torch.Tensor,
        inputs: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k_chunk, w_chunk, u_chunk, token_decay_chunk, chunk_decay_chunk = inputs
        output = carry.to(dtype)
        projected = torch.bmm(w_chunk, output).float()
        value = (u_chunk.float() - projected) * token_decay_chunk[:, :, None]
        update = torch.bmm(k_chunk.transpose(1, 2), value.to(dtype)).float()
        next_carry = carry * chunk_decay_chunk[:, None, None] + update
        return next_carry, output

    # Scan preserves the recurrence without making Dynamo unroll every chunk.
    _, outputs = torch._higher_order_ops.scan(
        step,
        state,
        (k_chunks, w_chunks, u_chunks, token_decay, chunk_decay),
        dim=1,
    )
    return (
        outputs.permute(1, 0, 2, 3)
        .reshape(batch, heads, chunks, dim, state_dim)
        .permute(0, 2, 1, 3, 4)
    )


def _jagged_reference(lengths: tuple[int, ...]) -> Callable[..., torch.Tensor]:
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)

    def reference(
        max_seq_len: int,
        alpha: float,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        _seq_offsets: torch.Tensor,
    ) -> torch.Tensor:
        outputs = []
        for start, stop in zip(offsets[:-1], offsets[1:], strict=True):
            q_batch = q[start:stop].transpose(0, 1)
            k_batch = k[start:stop].permute(1, 2, 0)
            v_batch = v[start:stop].transpose(0, 1)
            scores = torch.nn.functional.silu(torch.bmm(q_batch, k_batch) * alpha)
            mask = torch.ones(
                stop - start,
                stop - start,
                dtype=torch.bool,
                device=q.device,
            ).tril(diagonal=-1)
            scores = torch.where(mask, scores / max_seq_len, 0.0)
            outputs.append(torch.bmm(scores.to(v.dtype), v_batch).transpose(0, 1))
        return torch.cat(outputs)

    return reference


def _plain_matmul(values: list[int], _shape: dict[str, Any]) -> Workload:
    m, k, n = values
    module = _module("examples.matmul")
    return Workload(
        module.matmul,
        (_rand((m, k)), _rand((k, n))),
        "examples/matmul.py:matmul",
        torch.matmul,
        "torch.matmul",
    )


def _bf16xint16(values: list[int], _shape: dict[str, Any]) -> Workload:
    m, k, n = values
    module = _module("examples.bf16xint16_gemm")
    weight = torch.randint(
        -8,
        9,
        (k, n),
        device=DEVICE,
        dtype=torch.int16,
        generator=_generator(),
    )
    return Workload(
        module._bf16xint16_gemm,
        (_rand((m, k)), weight),
        "examples/bf16xint16_gemm.py:_bf16xint16_gemm",
        module.reference_bf16xint16_pytorch,
        "examples.bf16xint16_gemm.reference_bf16xint16_pytorch",
    )


def _broadcast(values: list[int], _shape: dict[str, Any]) -> Workload:
    batch, m, k, n = values
    module = _module("examples.broadcast_matmul")
    return Workload(
        module.broadcast_matmul,
        (_rand((batch, m, k)), _rand((k, n))),
        "examples/broadcast_matmul.py:broadcast_matmul",
        torch.matmul,
        "torch.matmul",
    )


def _gather(values: list[int], _shape: dict[str, Any]) -> Workload:
    batch, size, count = values
    module = _module("examples.gather_gemv")
    indices = torch.arange(count, device=DEVICE, dtype=torch.int32) % batch
    return Workload(
        module.gather_gemv,
        (_rand((batch, size, size)), indices, _rand((size,))),
        "examples/gather_gemv.py:gather_gemv",
        _gather_reference,
        "vectorized PyTorch gather + matmul",
    )


def _chunk_state(values: list[int], _shape: dict[str, Any]) -> Workload:
    batch, heads, groups, seqlen, chunk_size, dim, state = values
    chunks = seqlen // chunk_size
    module = _module("examples.mamba2_chunk_state")
    return Workload(
        module.helion_mamba2_chunk_state_kernel,
        (
            _rand((batch, seqlen, groups, state)),
            _rand((batch, seqlen, heads, dim)),
            _positive((batch, heads, chunks, chunk_size)),
            _decay(batch, heads, chunks, chunk_size, torch.bfloat16),
        ),
        "examples/mamba2_chunk_state.py:helion_mamba2_chunk_state_kernel",
        module.ref_chunk_state,
        "examples.mamba2_chunk_state.ref_chunk_state",
    )


def _attention(values: list[int], shape: dict[str, Any]) -> Workload:
    batch, heads, query_len, key_len, dim = values
    dtype = _dtype(shape.get("dtype"))
    module = _module("examples.attention")
    return Workload(
        module.attention,
        (
            _rand((batch, heads, query_len, dim), dtype),
            _rand((batch, heads, key_len, dim), dtype),
            _rand((batch, heads, key_len, dim), dtype),
        ),
        "examples/attention.py:attention",
        _dense_attention_reference,
        "explicit PyTorch matmul + softmax + matmul",
    )


def _causal_attention(values: list[int], shape: dict[str, Any]) -> Workload:
    batch, heads, seqlen, dim = values
    dtype = _dtype(shape.get("dtype"))
    module = _module("examples.attention")
    args = tuple(_rand((batch, heads, seqlen, dim), dtype) for _ in range(3))
    return Workload(
        module.causal_attention,
        args,
        "examples/attention.py:causal_attention",
        _causal_attention_reference,
        "explicit causal PyTorch matmul + softmax + matmul",
    )


def _biased_attention(values: list[int], shape: dict[str, Any]) -> Workload:
    batch, heads, query_len, key_len, dim = values
    dtype = _dtype(shape.get("dtype"))
    module = _module("examples.attention")
    return Workload(
        module.biased_attention,
        (
            _rand((batch, heads, query_len, dim), dtype),
            _rand((batch, heads, key_len, dim), dtype),
            _rand((batch, heads, key_len, dim), dtype),
            _rand((batch, heads, query_len, key_len), torch.float32),
        ),
        "examples/attention.py:biased_attention",
        _biased_attention_reference,
        "explicit biased PyTorch matmul + softmax + matmul",
    )


def _attention_backward(values: list[int], shape: dict[str, Any]) -> Workload:
    batch, heads, seqlen, dim = values
    dtype = _dtype(shape.get("dtype"))
    module = _module("examples.attention")
    tensor_shape = (batch, heads, seqlen, dim)
    row_shape = (batch, heads, seqlen)
    return Workload(
        module.attention_backward,
        (
            _rand(tensor_shape, dtype),
            _rand(tensor_shape, dtype),
            _rand(tensor_shape, dtype),
            _rand(tensor_shape, dtype),
            torch.full(
                row_shape,
                math.log2(seqlen),
                device=DEVICE,
                dtype=torch.float32,
            ),
            _rand(tensor_shape, dtype),
            torch.zeros(row_shape, device=DEVICE, dtype=torch.float32),
            1.0 / math.sqrt(dim),
        ),
        "examples/attention.py:attention_backward",
        _attention_backward_reference,
        "explicit PyTorch attention-backward equations",
    )


def _jagged_hstu(values: list[int], shape: dict[str, Any]) -> Workload:
    heads, dim, max_seq_len = values
    lengths = shape.get("lengths")
    if lengths is None:
        value, repeats = shape["lengths_repeat"]
        lengths = [value] * repeats
    total = sum(lengths)
    offsets = torch.zeros(len(lengths) + 1, device=DEVICE, dtype=torch.int32)
    offsets[1:] = torch.tensor(lengths, device=DEVICE, dtype=torch.int32).cumsum(0)
    module = _module("examples.jagged_hstu_attn")
    return Workload(
        module._helion_jagged_attention_kernel,
        (
            max_seq_len,
            1.0 / dim**2,
            _rand((total, heads, dim)),
            _rand((total, heads, dim)),
            _rand((total, heads, dim)),
            offsets,
        ),
        "examples/jagged_hstu_attn.py:_helion_jagged_attention_kernel",
        _jagged_reference(tuple(lengths)),
        "static-segment PyTorch HSTU reference",
    )


def _squeeze_excitation(values: list[int], _shape: dict[str, Any]) -> Workload:
    m, n, k = values
    module = _module("examples.squeeze_and_excitation_net")
    return Workload(
        module.squeeze_and_excitation_net_fwd,
        (_rand((m, n)), _rand((n, k)), _rand((k, n))),
        "examples/squeeze_and_excitation_net.py:squeeze_and_excitation_net_fwd",
        _squeeze_excitation_reference,
        "examples.squeeze_and_excitation_net reference with intermediates",
    )


def _gdn(values: list[int], _shape: dict[str, Any]) -> Workload:
    batch, seqlen, heads, chunk_size, dim, state = values
    module = _module("examples.gdn_fwd_h")
    positions = torch.arange(seqlen, device=DEVICE, dtype=torch.float32) % chunk_size
    decay = (
        (-0.01 * positions)
        .view(1, seqlen, 1)
        .expand(batch, seqlen, heads)
        .contiguous()
    )
    return Workload(
        module.helion_gdn_fwd_h,
        (
            _rand((batch, seqlen, heads, dim)),
            _rand((batch, seqlen, heads, dim)),
            _rand((batch, seqlen, heads, state)),
            decay,
            chunk_size,
        ),
        "examples/gdn_fwd_h.py:helion_gdn_fwd_h",
        _gdn_reference,
        "optimized PyTorch GDN scan with BF16 BMM and FP32 state arithmetic",
        inductor_options={"epilogue_fusion": False},
    )


def _chunk_scan(values: list[int], _shape: dict[str, Any]) -> Workload:
    batch, heads, groups, seqlen, chunk_size, dim, state = values
    chunks = seqlen // chunk_size
    module = _module("examples.mamba2_chunk_scan")
    return Workload(
        module.helion_mamba2_chunk_scan_kernel,
        (
            _rand((batch, chunks, groups, chunk_size, chunk_size)),
            _rand((batch, seqlen, heads, dim)),
            _positive((batch, heads, chunks, chunk_size)),
            _decay(batch, heads, chunks, chunk_size, torch.bfloat16),
            _rand((batch, seqlen, groups, state)),
            _rand((batch, chunks, heads, dim, state)),
            _rand((heads,)),
        ),
        "examples/mamba2_chunk_scan.py:helion_mamba2_chunk_scan_kernel",
        module.ref_chunk_scan,
        "examples.mamba2_chunk_scan.ref_chunk_scan",
    )


BUILDERS = {
    "plain_matmul": _plain_matmul,
    "bf16xint16_gemm": _bf16xint16,
    "broadcast_matmul": _broadcast,
    "gather_gemv": _gather,
    "mamba2_chunk_state": _chunk_state,
    "dense_attention": _attention,
    "causal_attention": _causal_attention,
    "biased_attention": _biased_attention,
    "attention_backward": _attention_backward,
    "jagged_hstu": _jagged_hstu,
    "squeeze_excitation": _squeeze_excitation,
    "gdn_forward_h": _gdn,
    "mamba2_chunk_scan": _chunk_scan,
}


def build_workload(cell: dict[str, Any]) -> Workload:
    return BUILDERS[cell["family"]](cell["values"], cell["shape_spec"])
