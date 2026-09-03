from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Any
from typing import Callable

import torch
import torch.nn.functional as F


HELION_ROOT = Path(os.environ["HELION_ROOT"]).expanduser().resolve()
DEVICE = torch.device("cuda")
EPS = 1e-5


@dataclass(frozen=True)
class Tolerance:
    rtol: float
    atol: float
    exact: bool = False


@dataclass
class Workload:
    kernel: str
    body: str
    shape: tuple[int, ...]
    dtype: str
    kernel_fn: Any
    reference_fn: Callable[..., Any]
    args: tuple[Any, ...]
    observe: Callable[[Any], dict[str, torch.Tensor]]
    tolerances: dict[str, Tolerance]
    optional_args: dict[str, Any]


_MODULES: dict[str, Any] = {}
_MISSING = object()


def _testing_stub() -> ModuleType:
    module = ModuleType("helion._testing")
    module.DEVICE = DEVICE
    module.HALF_DTYPE = torch.float16

    def run_example(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("run_example is unavailable in the artifact import shim")

    module.run_example = run_example
    return module


def _load_module(name: str, relative_path: str) -> Any:
    cached = _MODULES.get(name)
    if cached is not None:
        return cached
    path = HELION_ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(
        f"_example_reduction_artifact_{name}", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    previous_testing: object = _MISSING
    if relative_path.startswith("examples/"):
        previous_testing = sys.modules.get("helion._testing", _MISSING)
        sys.modules["helion._testing"] = _testing_stub()
    try:
        spec.loader.exec_module(module)
    finally:
        if previous_testing is _MISSING:
            sys.modules.pop("helion._testing", None)
        else:
            sys.modules["helion._testing"] = previous_testing
    _MODULES[name] = module
    return module


def _returned(*names: str) -> Callable[[Any], dict[str, torch.Tensor]]:
    def observe(output: Any) -> dict[str, torch.Tensor]:
        values = output if isinstance(output, (tuple, list)) else (output,)
        return dict(zip(names, values, strict=True))

    return observe


def _build_rms_norm(shape: tuple[int, ...]) -> Workload:
    module = _load_module("rms_norm", "pretuned_kernels/rms_norm/rms_norm.py")
    rows, hidden = shape
    x = torch.randn(rows, hidden, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(hidden, device=DEVICE, dtype=torch.bfloat16)
    return Workload(
        "rms_norm",
        "pretuned_kernels/rms_norm/rms_norm.py:rms_norm",
        shape,
        "bfloat16",
        module.rms_norm,
        module._rms_norm_torch,
        (x, weight),
        _returned("output"),
        {"output": Tolerance(1e-2, 1e-2)},
        {"eps": 1e-5},
    )


def _build_layer_norm(shape: tuple[int, ...]) -> Workload:
    module = _load_module(
        "layer_norm", "pretuned_kernels/layer_norm/layer_norm.py"
    )
    rows, hidden = shape
    x = torch.randn(rows, hidden, device=DEVICE, dtype=torch.float16)
    weight = torch.randn(hidden, device=DEVICE, dtype=torch.float16)
    bias = torch.randn(hidden, device=DEVICE, dtype=torch.float16)
    return Workload(
        "layer_norm",
        "pretuned_kernels/layer_norm/layer_norm.py:layer_norm",
        shape,
        "float16",
        module.layer_norm,
        module._layer_norm_torch,
        (x, weight, bias),
        _returned("output"),
        {"output": Tolerance(1e-2, 1e-2)},
        {},
    )


def _build_softmax(shape: tuple[int, ...]) -> Workload:
    module = _load_module("softmax", "pretuned_kernels/softmax/softmax.py")
    rows, width = shape
    x = torch.randn(rows, width, device=DEVICE, dtype=torch.float16)
    return Workload(
        "softmax",
        "pretuned_kernels/softmax/softmax.py:softmax",
        shape,
        "float16",
        module.softmax,
        module._softmax_torch,
        (x,),
        _returned("output"),
        {"output": Tolerance(1e-3, 1e-3)},
        {},
    )


def _build_cross_entropy(shape: tuple[int, ...]) -> Workload:
    module = _load_module(
        "cross_entropy", "pretuned_kernels/cross_entropy/cross_entropy.py"
    )
    tokens, vocab = shape
    logits = torch.randn(tokens, vocab, device=DEVICE, dtype=torch.bfloat16)
    labels = torch.randint(0, vocab, (tokens,), device=DEVICE, dtype=torch.int64)
    return Workload(
        "cross_entropy",
        "pretuned_kernels/cross_entropy/cross_entropy.py:cross_entropy",
        shape,
        "bfloat16",
        module.cross_entropy,
        module._cross_entropy_torch,
        (logits, labels),
        _returned("loss"),
        {"loss": Tolerance(1e-2, 1e-2)},
        {},
    )


def _kl_reference(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    log_target: bool,
    reduction: str,
    _eps: float,
) -> torch.Tensor:
    return F.kl_div(
        y_pred,
        y_true,
        reduction=reduction,
        log_target=log_target,
    ).to(torch.float32)


def _build_kl_div(shape: tuple[int, ...]) -> Workload:
    module = _load_module("kl_div", "examples/kl_div.py")
    tokens, vocab = shape
    y_pred = torch.randn(
        tokens, vocab, device=DEVICE, dtype=torch.bfloat16
    ).log_softmax(-1)
    y_true = torch.randn(
        tokens, vocab, device=DEVICE, dtype=torch.bfloat16
    ).softmax(-1)
    return Workload(
        "kl_div",
        "examples/kl_div.py:kl_div_forward",
        shape,
        "bfloat16",
        module.kl_div_forward,
        _kl_reference,
        (y_pred, y_true, False, "batchmean", 1e-10),
        _returned("loss"),
        {"loss": Tolerance(2e-2, 2e-2)},
        {"log_target": False, "reduction": "batchmean", "eps": 1e-10},
    )


def _jsd_reference(
    log_q: torch.Tensor,
    log_p: torch.Tensor,
    shift_labels: torch.Tensor | None,
    beta: float,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = torch.exp(log_q.to(torch.float32))
    p = torch.exp(log_p.to(torch.float32))
    mixture = beta * p + (1.0 - beta) * q
    log_mixture = torch.log(mixture)
    per_value_loss = beta * p * (log_p - log_mixture) + (1.0 - beta) * q * (
        log_q - log_mixture
    )
    per_value_dx = (1.0 - beta) * q * (log_q - log_mixture)
    if shift_labels is None:
        scale = 1.0 / log_q.shape[0]
        mask = None
    else:
        mask = shift_labels != ignore_index
        count = torch.clamp(mask.sum(), min=1)
        scale = 1.0 / count
    per_row_loss = torch.sum(per_value_loss * scale, dim=-1)
    per_row_dx = torch.sum(per_value_dx * scale, dim=-1)
    if mask is not None:
        per_row_loss = torch.where(mask, per_row_loss, 0.0)
        per_row_dx = torch.where(mask, per_row_dx, 0.0)
    return torch.sum(per_row_loss), per_row_dx


def _build_jsd(shape: tuple[int, ...]) -> Workload:
    module = _load_module("jsd", "examples/jsd.py")
    tokens, vocab = shape
    log_q = torch.randn(
        tokens, vocab, device=DEVICE, dtype=torch.bfloat16
    ).log_softmax(-1)
    log_p = torch.randn(
        tokens, vocab, device=DEVICE, dtype=torch.bfloat16
    ).log_softmax(-1)
    return Workload(
        "jsd",
        "examples/jsd.py:jsd_forward",
        shape,
        "bfloat16",
        module.jsd_forward,
        _jsd_reference,
        (log_q, log_p, None, 0.5, -100),
        _returned("loss", "dX"),
        {
            "loss": Tolerance(3e-2, 3e-2),
            "dX": Tolerance(3e-2, 3e-2),
        },
        {"shift_labels": None, "beta": 0.5, "ignore_index": -100},
    )


def _fused_linear_jsd_reference(
    beta: float,
    _ignore_index: int,
    temperature: float,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    student_scaled = student_logits.to(torch.float32) / temperature
    teacher_scaled = teacher_logits.to(torch.float32) / temperature
    student_prob = torch.softmax(student_scaled, dim=-1)
    teacher_prob = torch.softmax(teacher_scaled, dim=-1)
    student_log_prob = torch.log_softmax(student_scaled, dim=-1)
    teacher_log_prob = torch.log_softmax(teacher_scaled, dim=-1)
    mixture = (1.0 - beta) * student_prob + beta * teacher_prob
    log_mixture = torch.log(mixture)
    student_kl = torch.sum(
        student_prob * (student_log_prob - log_mixture), dim=-1
    )
    teacher_kl = torch.sum(
        teacher_prob * (teacher_log_prob - log_mixture), dim=-1
    )
    loss = (1.0 - beta) * student_kl + beta * teacher_kl
    grad = ((1.0 - beta) / temperature) * (student_prob - mixture)
    return loss, grad


def _build_fused_linear_jsd(shape: tuple[int, ...]) -> Workload:
    module = _load_module("fused_linear_jsd", "examples/fused_linear_jsd.py")
    tokens, vocab = shape
    student = torch.randn(tokens, vocab, device=DEVICE, dtype=torch.bfloat16)
    teacher = torch.randn(tokens, vocab, device=DEVICE, dtype=torch.bfloat16)
    return Workload(
        "fused_linear_jsd",
        "examples/fused_linear_jsd.py:jsd_kernel",
        shape,
        "bfloat16",
        module.jsd_kernel,
        _fused_linear_jsd_reference,
        (0.5, -100, 1.0, student, teacher),
        _returned("loss", "grad_student_logits"),
        {
            "loss": Tolerance(3e-2, 3e-2),
            "grad_student_logits": Tolerance(3e-2, 3e-2),
        },
        {"beta": 0.5, "ignore_index": -100, "temperature": 1.0},
    )


def _grpo_reference(
    logits: torch.Tensor,
    selected_logits: torch.Tensor,
    old_logp: torch.Tensor | None,
    ref_logp: torch.Tensor | None,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor | None,
    temperature: float,
    beta: float,
    eps_low: float,
    eps_high: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    scaled = logits[:, :-1, :].to(torch.float32) / temperature
    lse = torch.logsumexp(scaled, dim=-1)
    logp = selected_logits - lse
    old = logp if old_logp is None else old_logp
    coefficient = torch.exp(logp - old)
    clipped_coefficient = torch.clamp(
        coefficient, 1.0 - eps_low, 1.0 + eps_high
    )
    loss_1 = coefficient * advantages[:, None]
    loss_2 = clipped_coefficient * advantages[:, None]
    loss = -torch.minimum(loss_1, loss_2)
    if completion_mask is not None:
        loss = loss * completion_mask
    kl = torch.zeros_like(loss)
    if beta != 0.0 and ref_logp is not None:
        kl = torch.exp(ref_logp - logp) - (ref_logp - logp) - 1.0
        if completion_mask is not None:
            kl = kl * completion_mask
        loss = loss + beta * kl
    is_clipped = (loss_1 < loss_2).to(torch.float32)
    return loss, kl, is_clipped, lse


def _build_grpo(shape: tuple[int, ...]) -> Workload:
    module = _load_module("grpo", "examples/grpo_loss.py")
    batch, sequence, vocab = shape
    temperature, beta, eps_low, eps_high = 0.9, 0.2, 0.2, 0.4
    logits = torch.randn(
        batch,
        sequence + 1,
        vocab,
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    completion_ids = torch.randint(
        0,
        vocab,
        (batch, sequence),
        device=DEVICE,
        dtype=torch.int64,
    )
    selected = module.extract_selected_logits_pytorch(
        logits[:, :-1, :], completion_ids, temperature
    )
    old_logp = torch.randn(
        batch, sequence, device=DEVICE, dtype=torch.float32
    )
    ref_logp = torch.randn(
        batch, sequence, device=DEVICE, dtype=torch.float32
    )
    advantages = torch.randn(batch, device=DEVICE, dtype=torch.float32)
    completion_mask = torch.ones(
        batch, sequence, device=DEVICE, dtype=torch.float32
    )
    args = (
        logits,
        selected,
        old_logp,
        ref_logp,
        advantages,
        completion_mask,
        temperature,
        beta,
        eps_low,
        eps_high,
    )
    return Workload(
        "grpo",
        "examples/grpo_loss.py:grpo_loss_forward",
        shape,
        "bfloat16",
        module.grpo_loss_forward,
        _grpo_reference,
        args,
        _returned("loss", "kl_loss", "is_clipped", "lse"),
        {
            "loss": Tolerance(3e-2, 3e-2),
            "kl_loss": Tolerance(3e-2, 3e-2),
            "is_clipped": Tolerance(0.0, 0.0, exact=True),
            "lse": Tolerance(3e-2, 3e-2),
        },
        {
            "temperature": temperature,
            "beta": beta,
            "eps_low": eps_low,
            "eps_high": eps_high,
            "old_logp": True,
            "ref_logp": True,
            "completion_mask": True,
        },
    )


def _rms_norm_bwd_reference(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    rsqrt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_float = x.to(torch.float32)
    grad_float = grad_out.to(torch.float32)
    rsqrt_float = rsqrt.to(torch.float32)
    weight_float = weight.to(torch.float32)[None, :]
    grad_weight = torch.sum(x_float * grad_float * rsqrt_float, dim=0)
    grad_x = weight_float * grad_float * rsqrt_float - (
        x_float
        * rsqrt_float**3
        * torch.mean(weight_float * grad_float * x_float, dim=-1, keepdim=True)
    )
    return grad_x.to(x.dtype), grad_weight.to(weight.dtype)


def _build_rms_norm_bwd(shape: tuple[int, ...]) -> Workload:
    module = _load_module("rms_norm_bwd", "examples/rms_norm.py")
    rows, hidden = shape
    x = torch.randn(rows, hidden, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(hidden, device=DEVICE, dtype=torch.bfloat16)
    grad_out = torch.randn(rows, hidden, device=DEVICE, dtype=torch.bfloat16)
    rsqrt = torch.rsqrt(
        torch.mean(x.to(torch.float32) ** 2, dim=-1, keepdim=True) + EPS
    ).to(torch.bfloat16)
    return Workload(
        "rms_norm_bwd",
        "examples/rms_norm.py:rms_norm_bwd",
        shape,
        "bfloat16",
        module.rms_norm_bwd,
        _rms_norm_bwd_reference,
        (grad_out, x, weight, rsqrt),
        _returned("grad_x", "grad_weight"),
        {
            "grad_x": Tolerance(3e-2, 8e-2),
            "grad_weight": Tolerance(3e-2, 3e-2),
        },
        {"eps": EPS, "rsqrt_dtype": "bfloat16"},
    )


def _layer_norm_bwd_reference(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_float = x.to(torch.float32)
    grad_float = grad_out.to(torch.float32)
    weight_float = weight.to(torch.float32)[None, :]
    x_hat = (x_float - mean.to(torch.float32)[:, None]) * rstd.to(
        torch.float32
    )[:, None]
    grad_weight = torch.sum(grad_float * x_hat, dim=0)
    grad_bias = torch.sum(grad_float, dim=0)
    weighted_grad = weight_float * grad_float
    c1 = torch.mean(x_hat * weighted_grad, dim=-1, keepdim=True)
    c2 = torch.mean(weighted_grad, dim=-1, keepdim=True)
    grad_x = (weighted_grad - (x_hat * c1 + c2)) * rstd.to(torch.float32)[
        :, None
    ]
    return (
        grad_x.to(x.dtype),
        grad_weight.to(weight.dtype),
        grad_bias.to(weight.dtype),
    )


def _build_layer_norm_bwd(shape: tuple[int, ...]) -> Workload:
    module = _load_module("layer_norm_bwd", "examples/layer_norm.py")
    rows, hidden = shape
    x = torch.randn(rows, hidden, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(hidden, device=DEVICE, dtype=torch.bfloat16)
    grad_out = torch.randn(rows, hidden, device=DEVICE, dtype=torch.bfloat16)
    mean = torch.mean(x.to(torch.float32), dim=-1)
    rstd = torch.rsqrt(
        torch.var(x.to(torch.float32), dim=-1, unbiased=False) + EPS
    )
    return Workload(
        "layer_norm_bwd",
        "examples/layer_norm.py:layer_norm_bwd",
        shape,
        "bfloat16",
        module.layer_norm_bwd,
        _layer_norm_bwd_reference,
        (grad_out, x, mean, rstd, weight),
        _returned("grad_x", "grad_weight", "grad_bias"),
        {
            "grad_x": Tolerance(3e-2, 3e-2),
            "grad_weight": Tolerance(3e-2, 3e-2),
            "grad_bias": Tolerance(3e-2, 3e-2),
        },
        {"compute_bias_grad": True, "eps": EPS},
    )


BUILDERS: dict[str, Callable[[tuple[int, ...]], Workload]] = {
    "rms_norm": _build_rms_norm,
    "layer_norm": _build_layer_norm,
    "softmax": _build_softmax,
    "cross_entropy": _build_cross_entropy,
    "kl_div": _build_kl_div,
    "jsd": _build_jsd,
    "fused_linear_jsd": _build_fused_linear_jsd,
    "grpo": _build_grpo,
    "rms_norm_bwd": _build_rms_norm_bwd,
    "layer_norm_bwd": _build_layer_norm_bwd,
}


def build_workload(kernel: str, shape: tuple[int, ...]) -> Workload:
    try:
        builder = BUILDERS[kernel]
    except KeyError as error:
        raise ValueError(f"unknown kernel: {kernel}") from error
    return builder(shape)


def describe_inputs(workload: Workload) -> list[dict[str, object]]:
    return [
        {
            "arg": index,
            "shape": list(value.shape),
            "dtype": str(value.dtype).removeprefix("torch."),
        }
        for index, value in enumerate(workload.args)
        if torch.is_tensor(value)
    ]
