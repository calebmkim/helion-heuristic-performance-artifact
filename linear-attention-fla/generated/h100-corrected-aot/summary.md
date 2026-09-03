# Linear-Attention Performance

Each Helion arm is normalized to FLA timed in the same process, then compared with seed; values above `1.0x` favor seed. Counts are correct common cells.

| Population | Discovered | Default / seed | FLA Triton / seed | AOT-tuned / seed |
|---|---:|---:|---:|---:|
| Overall | 96 | 2.0908x (96) | 1.1756x (96) | 0.9140x (96) |
| Forward | 54 | 2.2236x (54) | 1.1776x (54) | 0.9478x (54) |
| Forward + backward | 42 | 1.9317x (42) | 1.1730x (42) | 0.8724x (42) |

## Per Kernel

| Kernel | Mode | Shapes | Default / seed | FLA Triton / seed | AOT-tuned / seed |
|---|---|---:|---:|---:|---:|
| Vanilla linear attention | Forward | 6 | 3.2532x (6) | 1.0745x (6) | 0.9635x (6) |
| Vanilla linear attention | Forward + backward | 6 | 2.6883x (6) | 1.3942x (6) | 0.9989x (6) |
| Simple GLA | Forward | 6 | 3.0400x (6) | 1.0801x (6) | 0.9552x (6) |
| Simple GLA | Forward + backward | 6 | 2.9351x (6) | 1.6427x (6) | 0.9927x (6) |
| Retention | Forward | 6 | 3.0032x (6) | 1.4531x (6) | 0.9578x (6) |
| Retention | Forward + backward | 6 | 2.4712x (6) | 1.3703x (6) | 0.9563x (6) |
| Full GLA | Forward | 6 | 2.6721x (6) | 1.4042x (6) | 0.9513x (6) |
| Full GLA | Forward + backward | 6 | 1.7450x (6) | 1.3134x (6) | 0.6642x (6) |
| Delta Rule | Forward | 6 | 2.1437x (6) | 1.1734x (6) | 0.9317x (6) |
| Delta Rule | Forward + backward | 6 | 1.6261x (6) | 0.9101x (6) | 0.8961x (6) |
| Gated delta rule | Forward | 6 | 2.0846x (6) | 1.0283x (6) | 0.9124x (6) |
| Gated delta rule | Forward + backward | 6 | 1.2889x (6) | 0.7890x (6) | 0.7526x (6) |
| KDA | Forward | 6 | 2.0148x (6) | 1.1837x (6) | 0.9666x (6) |
| KDA | Forward + backward | 6 | 1.4072x (6) | 1.0325x (6) | 0.9051x (6) |
| KDA fused | Forward | 6 | 1.8512x (6) | 1.2110x (6) | 0.9522x (6) |
| KDA variable length | Forward | 6 | 1.0047x (6) | 1.0629x (6) | 0.9407x (6) |
