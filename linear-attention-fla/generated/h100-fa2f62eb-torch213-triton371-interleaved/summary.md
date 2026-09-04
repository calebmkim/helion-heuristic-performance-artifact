# Linear-Attention Performance

All arms in a cell share one process and one FLA timing. Cold-L2 CUDA-event samples are interleaved in rotated forward/reverse order with equal counts.

Validation: 96/96 cells completed all four arms; 288/288 Helion arm-cells passed correctness.
Environment: NVIDIA H100 80GB HBM3, Helion `fa2f62eb686e`, FLA `0.5.2` (`6bd90692588c`).

Values are higher-is-better geometric means; counts are correct common cells.

## Performance vs FLA

| Population | Discovered | Default | FLA Triton | Seed | AOT-tuned |
|---|---:|---:|---:|---:|---:|
| Overall | 96 | 0.5445x (96) | 1.0000x (96) | 1.1573x (96) | 1.2888x (96) |
| Forward | 54 | 0.5235x (54) | 1.0000x (54) | 1.1843x (54) | 1.2612x (54) |
| Forward + backward | 42 | 0.5729x (42) | 1.0000x (42) | 1.1235x (42) | 1.3252x (42) |

### Per Kernel

| Kernel | Mode | Shapes | Default | FLA Triton | Seed | AOT-tuned |
|---|---|---:|---:|---:|---:|---:|
| Vanilla linear attention | Forward | 6 | 0.3316x (6) | 1.0000x (6) | 1.0835x (6) | 1.1642x (6) |
| Vanilla linear attention | Forward + backward | 6 | 0.4916x (6) | 1.0000x (6) | 1.3863x (6) | 1.4527x (6) |
| Simple GLA | Forward | 6 | 0.3324x (6) | 1.0000x (6) | 1.0680x (6) | 1.1238x (6) |
| Simple GLA | Forward + backward | 6 | 0.5179x (6) | 1.0000x (6) | 1.4885x (6) | 1.5797x (6) |
| Retention | Forward | 6 | 0.4947x (6) | 1.0000x (6) | 1.5738x (6) | 1.6137x (6) |
| Retention | Forward + backward | 6 | 0.5152x (6) | 1.0000x (6) | 1.4101x (6) | 1.4715x (6) |
| Full GLA | Forward | 6 | 0.5317x (6) | 1.0000x (6) | 1.4382x (6) | 1.5143x (6) |
| Full GLA | Forward + backward | 6 | 0.7198x (6) | 1.0000x (6) | 1.2526x (6) | 1.9579x (6) |
| Delta Rule | Forward | 6 | 0.5389x (6) | 1.0000x (6) | 1.1340x (6) | 1.2544x (6) |
| Delta Rule | Forward + backward | 6 | 0.5719x (6) | 1.0000x (6) | 0.8831x (6) | 1.0094x (6) |
| Gated delta rule | Forward | 6 | 0.4750x (6) | 1.0000x (6) | 1.0116x (6) | 1.1238x (6) |
| Gated delta rule | Forward + backward | 6 | 0.5242x (6) | 1.0000x (6) | 0.6932x (6) | 0.9650x (6) |
| KDA | Forward | 6 | 0.5965x (6) | 1.0000x (6) | 1.2007x (6) | 1.2476x (6) |
| KDA | Forward + backward | 6 | 0.7153x (6) | 1.0000x (6) | 1.0128x (6) | 1.1149x (6) |
| KDA fused | Forward | 6 | 0.6321x (6) | 1.0000x (6) | 1.1871x (6) | 1.2615x (6) |
| KDA variable length | Forward | 6 | 1.0548x (6) | 1.0000x (6) | 1.0704x (6) | 1.1381x (6) |

## Seed Comparisons

Values above `1.0x` favor the heuristic seed.

| Population | Discovered | Default / seed | FLA Triton / seed | AOT-tuned / seed |
|---|---:|---:|---:|---:|
| Overall | 96 | 2.1254x (96) | 1.1573x (96) | 0.8980x (96) |
| Forward | 54 | 2.2625x (54) | 1.1843x (54) | 0.9391x (54) |
| Forward + backward | 42 | 1.9612x (42) | 1.1235x (42) | 0.8478x (42) |

### Per Kernel

| Kernel | Mode | Shapes | Default / seed | FLA Triton / seed | AOT-tuned / seed |
|---|---|---:|---:|---:|---:|
| Vanilla linear attention | Forward | 6 | 3.2675x (6) | 1.0835x (6) | 0.9307x (6) |
| Vanilla linear attention | Forward + backward | 6 | 2.8197x (6) | 1.3863x (6) | 0.9543x (6) |
| Simple GLA | Forward | 6 | 3.2132x (6) | 1.0680x (6) | 0.9503x (6) |
| Simple GLA | Forward + backward | 6 | 2.8740x (6) | 1.4885x (6) | 0.9423x (6) |
| Retention | Forward | 6 | 3.1813x (6) | 1.5738x (6) | 0.9753x (6) |
| Retention | Forward + backward | 6 | 2.7370x (6) | 1.4101x (6) | 0.9583x (6) |
| Full GLA | Forward | 6 | 2.7048x (6) | 1.4382x (6) | 0.9498x (6) |
| Full GLA | Forward + backward | 6 | 1.7402x (6) | 1.2526x (6) | 0.6398x (6) |
| Delta Rule | Forward | 6 | 2.1042x (6) | 1.1340x (6) | 0.9040x (6) |
| Delta Rule | Forward + backward | 6 | 1.5443x (6) | 0.8831x (6) | 0.8749x (6) |
| Gated delta rule | Forward | 6 | 2.1296x (6) | 1.0116x (6) | 0.9002x (6) |
| Gated delta rule | Forward + backward | 6 | 1.3224x (6) | 0.6932x (6) | 0.7184x (6) |
| KDA | Forward | 6 | 2.0129x (6) | 1.2007x (6) | 0.9624x (6) |
| KDA | Forward + backward | 6 | 1.4160x (6) | 1.0128x (6) | 0.9085x (6) |
| KDA fused | Forward | 6 | 1.8781x (6) | 1.1871x (6) | 0.9410x (6) |
| KDA variable length | Forward | 6 | 1.0149x (6) | 1.0704x (6) | 0.9406x (6) |
