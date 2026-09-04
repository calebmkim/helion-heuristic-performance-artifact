# Generated H100 Results

## Primary

[`h100-main-torch213-triton371-vllm024-curated`](h100-main-torch213-triton371-vllm024-curated/)
is the default reproduction target:

- Helion main `fa2f62eb686ef846c76f8b9e18beec30fbc5bee1`
- PyTorch `2.13.0+cu132`
- Triton `3.7.1`
- vLLM `0.24.0` at `ee0da84ab9e04ac7610e28580af62c365e898389`
- vLLM stable-extension SHA-256
  `686ece5394839c1eb214ba91fdf0219e47fab4ccce0a69a10d3fba39db9a47fb`
- H100, CUDA runtime 13.2
- 153/153 correct four-arm cells
- heuristic seed `1.269x`; exact AOT `1.310x`; base default `0.578x`

## Historical

[`historical/torch213dev-triton370-h100-pr3551-curated`](historical/torch213dev-triton370-h100-pr3551-curated/)
is retained for historical comparison:

- Helion `746ee7c8a94fcc5fe5eab18356bde1aab69f9c43`
- PyTorch `2.13.0.dev20260520+cu130`
- Triton `3.7.0`
- vLLM development revision
  `fc7fc421e98863c4ffb1aa02d46bd6e4d0202c26`
- H100, CUDA runtime 13.0
- 153/153 correct four-arm cells
- heuristic seed `1.267x`; exact AOT `1.308x`; base default `0.577x`

The new and historical normalized geomeans agree to within 0.8% for every
kernel and arm. The primary result exists to make the software contract exact
and reproducible, not because the prior good result changed materially.
