# Linear-Attention Performance

All arms in a cell share one process and one FLA timing. Cold-L2 CUDA-event samples are interleaved in rotated forward/reverse order with equal counts.

Validation: 8/8 cells completed all four arms; 24/24 Helion arm-cells passed correctness.
Environment: NVIDIA H100 80GB HBM3, Helion `eacfee67c0fd`, FLA `0.5.2` (`6bd90692588c`).

Values are higher-is-better geometric means; counts are correct common cells.

## Performance vs FLA

| Population | Discovered | Default | FLA Triton | Seed | AOT-tuned |
|---|---:|---:|---:|---:|---:|
| Overall | 8 | 0.4799x (8) | 1.0000x (8) | 1.1172x (8) | 1.2916x (8) |
| Forward | 4 | 0.4236x (4) | 1.0000x (4) | 1.2101x (4) | 1.2525x (4) |
| Forward + backward | 4 | 0.5437x (4) | 1.0000x (4) | 1.0314x (4) | 1.3319x (4) |

### Per Kernel

| Kernel | Mode | Shapes | Default | FLA Triton | Seed | AOT-tuned |
|---|---|---:|---:|---:|---:|---:|
| Vanilla linear attention | Forward | 1 | 0.2664x (1) | 1.0000x (1) | 1.0221x (1) | 1.0670x (1) |
| Vanilla linear attention | Forward + backward | 1 | 0.3632x (1) | 1.0000x (1) | 1.3117x (1) | 1.3615x (1) |
| Full GLA | Forward | 1 | 0.4952x (1) | 1.0000x (1) | 1.5360x (1) | 1.5423x (1) |
| Full GLA | Forward + backward | 1 | 0.7094x (1) | 1.0000x (1) | 1.2915x (1) | 2.1884x (1) |
| Gated delta rule | Forward | 1 | 0.4222x (1) | 1.0000x (1) | 1.0649x (1) | 1.1504x (1) |
| Gated delta rule | Forward + backward | 1 | 0.4801x (1) | 1.0000x (1) | 0.6643x (1) | 0.9366x (1) |
| KDA | Forward | 1 | 0.5783x (1) | 1.0000x (1) | 1.2829x (1) | 1.2998x (1) |
| KDA | Forward + backward | 1 | 0.7068x (1) | 1.0000x (1) | 1.0056x (1) | 1.1276x (1) |

## Seed Comparisons

Values above `1.0x` favor the heuristic seed.

| Population | Discovered | Default / seed | FLA Triton / seed | AOT-tuned / seed |
|---|---:|---:|---:|---:|
| Overall | 8 | 2.3278x (8) | 1.1172x (8) | 0.8650x (8) |
| Forward | 4 | 2.8566x (4) | 1.2101x (4) | 0.9662x (4) |
| Forward + backward | 4 | 1.8969x (4) | 1.0314x (4) | 0.7744x (4) |

### Per Kernel

| Kernel | Mode | Shapes | Default / seed | FLA Triton / seed | AOT-tuned / seed |
|---|---|---:|---:|---:|---:|
| Vanilla linear attention | Forward | 1 | 3.8369x (1) | 1.0221x (1) | 0.9579x (1) |
| Vanilla linear attention | Forward + backward | 1 | 3.6117x (1) | 1.3117x (1) | 0.9634x (1) |
| Full GLA | Forward | 1 | 3.1015x (1) | 1.5360x (1) | 0.9959x (1) |
| Full GLA | Forward + backward | 1 | 1.8207x (1) | 1.2915x (1) | 0.5902x (1) |
| Gated delta rule | Forward | 1 | 2.5224x (1) | 1.0649x (1) | 0.9257x (1) |
| Gated delta rule | Forward + backward | 1 | 1.3838x (1) | 0.6643x (1) | 0.7093x (1) |
| KDA | Forward | 1 | 2.2185x (1) | 1.2829x (1) | 0.9870x (1) |
| KDA | Forward + backward | 1 | 1.4228x (1) | 1.0056x (1) | 0.8918x (1) |
