# PyTorch 2.13 Compatibility Smoke

This bounded H100 run checks the repository's unified primary stack without
repeating the complete 96-cell linear-attention campaign:

- PyTorch `2.13.0+cu132`
- CUDA `13.2`
- Triton `3.7.1`
- Helion `eacfee67c0fdbc5a1c068f16a3b2f9f15ce23eb7`
- FLA `0.5.2` at `6bd90692588c81fe102ee6e12ac70686359658a2`

The Helion and FLA revisions, GPU, shape, and timing protocol match the
existing PyTorch 2.12 H100 result. The benchmark-relevant package-version
change under test is PyTorch 2.12 to 2.13.

## Population

The controlled run uses `B1_T8192_H96_D128` for four representative dense
variants, with both forward and forward-plus-backward:

- vanilla linear attention;
- full GLA;
- gated delta rule;
- KDA.

All 8 four-arm cells completed. All 24 replayed Helion arm-cells passed
correctness, and all default, seed, and AOT config selections were identical
to the existing result.

## Comparison with the Existing Result

Values are geometric-mean performance relative to the same cell's FLA timing.
Higher is better.

| Arm | PyTorch 2.12 | PyTorch 2.13 | Change |
|---|---:|---:|---:|
| Default | `0.4797x` | `0.4799x` | `+0.05%` |
| Heuristic seed | `1.1169x` | `1.1172x` | `+0.03%` |
| AOT tuned | `1.2915x` | `1.2916x` | `+0.01%` |

Every individual absolute latency changed by at most `0.55%`; every
performance-versus-FLA ratio changed by at most `0.41%`. This supports treating
the Torch update as performance-neutral for the sampled cells. It is a
compatibility check, not a replacement for the complete 96-cell result.

## Timing Workflows

The controlled result is in [results.json](results.json), with its
[manifest](manifest.csv), [config replay](config-replay.json), and generated
[summary](summary.md).

The native Helion harness was also run for the same four variants and one
shape. Its [dashboard output](native-helionbench.json) contains all eight
forward/backward model rows, every accuracy value is `1.0`, and every latency
is positive. Native Helion latencies were within `0.71%` of the controlled
seed timings, but the two workflows remain methodologically distinct and
their results must not be merged.
