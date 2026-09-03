# vLLM Reduction Implementation Inspection

- Revision: `fc7fc421e98863c4ffb1aa02d46bd6e4d0202c26`
- Dirty checkout: `False`
- All expected sources found: `True`

| Kernel | Entry point | Backend | Native source | Helion override |
|---|---|---|---|---|
| `dynamic_per_token_scaled_fp8_quant` | `torch.ops._C.dynamic_per_token_scaled_fp8_quant` | CUDA C++ | `common.cu` | yes |
| `per_token_group_fp8_quant` | `torch.ops._C.per_token_group_fp8_quant` | CUDA C++ | `per_token_group_quant.cu` | yes |
| `rms_norm_dynamic_per_token_quant` | `torch.ops._C.rms_norm_dynamic_per_token_quant` | CUDA C++ | `fused_layernorm_dynamic_per_token_quant.cu` | yes |
| `rms_norm_per_block_quant` | `torch.ops._C.rms_norm_per_block_quant` | CUDA C++ | `fused_layernorm_dynamic_per_token_quant.cu` | yes |
| `silu_and_mul_per_block_quant` | `torch.ops._C.silu_and_mul_per_block_quant` | CUDA C++ | `fused_silu_mul_block_quant.cu` | no |
| `fused_qk_norm_rope` | `torch.ops._C.fused_qk_norm_rope` | CUDA C++ | `fused_qknorm_rope_kernel.cu` | no |

## Notes

- The public Python utility has a Triton fallback, but contiguous NVIDIA inputs prefer this compiled operator.
- The CUDA kernel is derived from TensorRT-LLM.
