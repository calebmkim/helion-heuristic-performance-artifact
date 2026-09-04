# Generated H100 Results

## Primary

[`h100-main-torch213-triton371-liger-mixed`](h100-main-torch213-triton371-liger-mixed/)
is the default reproduction target:

- Helion main `fa2f62eb686ef846c76f8b9e18beec30fbc5bee1`
- PyTorch `2.13.0+cu132`
- Triton `3.7.1`
- H100, CUDA runtime 13.2
- 80/80 correct common cells
- heuristic seed `1.088x`; base default `0.268x`

## Historical

[`historical/torch212-triton370-h100-pr3551-liger-mixed`](historical/torch212-triton370-h100-pr3551-liger-mixed/)
is retained for historical comparison:

- Helion `6ca445ca0605f703d44967dbedd153a0a89a5e00`
- PyTorch `2.12.0+cu132`
- Triton `3.7.0`
- H100, CUDA runtime 13.2
- 80/80 correct common cells
- heuristic seed `1.073x`; base default `0.253x`

The Helion seed's absolute latency is nearly unchanged between these runs.
Some normalized per-kernel ratios moved because `torch.compile` changed,
especially for fused JSD and LayerNorm backward. Always report the software
pair with normalized results.
