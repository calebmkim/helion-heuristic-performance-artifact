# Linear-Attention Performance

All arms in a cell share one process and one FLA timing. Cold-L2 CUDA-event samples are interleaved in rotated forward/reverse order with equal counts.

Validation: 96/96 cells completed all four arms; 288/288 Helion arm-cells passed correctness.
Environment: NVIDIA H100 80GB HBM3, Helion `eacfee67c0fd`, FLA `0.5.2` (`6bd90692588c`).

Values are higher-is-better geometric means; counts are correct common cells.

## Performance vs FLA

| Population | Discovered | Default | FLA Triton | Seed | AOT-tuned |
|---|---:|---:|---:|---:|---:|
| Overall | 96 | 0.5492x (96) | 1.0000x (96) | 1.1620x (96) | 1.2861x (96) |
| Forward | 54 | 0.5217x (54) | 1.0000x (54) | 1.1845x (54) | 1.2574x (54) |
| Forward + backward | 42 | 0.5868x (42) | 1.0000x (42) | 1.1338x (42) | 1.3240x (42) |

### Per Kernel

| Kernel | Mode | Shapes | Default | FLA Triton | Seed | AOT-tuned |
|---|---|---:|---:|---:|---:|---:|
| Vanilla linear attention | Forward | 6 | 0.3303x (6) | 1.0000x (6) | 1.0868x (6) | 1.1597x (6) |
| Vanilla linear attention | Forward + backward | 6 | 0.4939x (6) | 1.0000x (6) | 1.3566x (6) | 1.4473x (6) |
| Simple GLA | Forward | 6 | 0.3313x (6) | 1.0000x (6) | 1.0677x (6) | 1.1190x (6) |
| Simple GLA | Forward + backward | 6 | 0.5610x (6) | 1.0000x (6) | 1.5065x (6) | 1.5412x (6) |
| Retention | Forward | 6 | 0.4903x (6) | 1.0000x (6) | 1.5665x (6) | 1.6126x (6) |
| Retention | Forward + backward | 6 | 0.5234x (6) | 1.0000x (6) | 1.4142x (6) | 1.4331x (6) |
| Full GLA | Forward | 6 | 0.5297x (6) | 1.0000x (6) | 1.4354x (6) | 1.5082x (6) |
| Full GLA | Forward + backward | 6 | 0.7422x (6) | 1.0000x (6) | 1.2905x (6) | 2.0091x (6) |
| Delta Rule | Forward | 6 | 0.5381x (6) | 1.0000x (6) | 1.1503x (6) | 1.2533x (6) |
| Delta Rule | Forward + backward | 6 | 0.5482x (6) | 1.0000x (6) | 0.8618x (6) | 1.0066x (6) |
| Gated delta rule | Forward | 6 | 0.4705x (6) | 1.0000x (6) | 1.0020x (6) | 1.1134x (6) |
| Gated delta rule | Forward + backward | 6 | 0.5694x (6) | 1.0000x (6) | 0.7414x (6) | 0.9962x (6) |
| KDA | Forward | 6 | 0.5947x (6) | 1.0000x (6) | 1.2018x (6) | 1.2470x (6) |
| KDA | Forward + backward | 6 | 0.7132x (6) | 1.0000x (6) | 1.0104x (6) | 1.1076x (6) |
| KDA fused | Forward | 6 | 0.6308x (6) | 1.0000x (6) | 1.1859x (6) | 1.2608x (6) |
| KDA variable length | Forward | 6 | 1.0606x (6) | 1.0000x (6) | 1.0712x (6) | 1.1348x (6) |

## Seed Comparisons

Values above `1.0x` favor the heuristic seed.

| Population | Discovered | Default / seed | FLA Triton / seed | AOT-tuned / seed |
|---|---:|---:|---:|---:|
| Overall | 96 | 2.1157x (96) | 1.1620x (96) | 0.9035x (96) |
| Forward | 54 | 2.2706x (54) | 1.1845x (54) | 0.9420x (54) |
| Forward + backward | 42 | 1.9320x (42) | 1.1338x (42) | 0.8563x (42) |

### Per Kernel

| Kernel | Mode | Shapes | Default / seed | FLA Triton / seed | AOT-tuned / seed |
|---|---|---:|---:|---:|---:|
| Vanilla linear attention | Forward | 6 | 3.2906x (6) | 1.0868x (6) | 0.9371x (6) |
| Vanilla linear attention | Forward + backward | 6 | 2.7468x (6) | 1.3566x (6) | 0.9373x (6) |
| Simple GLA | Forward | 6 | 3.2227x (6) | 1.0677x (6) | 0.9542x (6) |
| Simple GLA | Forward + backward | 6 | 2.6854x (6) | 1.5065x (6) | 0.9774x (6) |
| Retention | Forward | 6 | 3.1952x (6) | 1.5665x (6) | 0.9714x (6) |
| Retention | Forward + backward | 6 | 2.7018x (6) | 1.4142x (6) | 0.9868x (6) |
| Full GLA | Forward | 6 | 2.7098x (6) | 1.4354x (6) | 0.9517x (6) |
| Full GLA | Forward + backward | 6 | 1.7387x (6) | 1.2905x (6) | 0.6423x (6) |
| Delta Rule | Forward | 6 | 2.1378x (6) | 1.1503x (6) | 0.9178x (6) |
| Delta Rule | Forward + backward | 6 | 1.5720x (6) | 0.8618x (6) | 0.8562x (6) |
| Gated delta rule | Forward | 6 | 2.1295x (6) | 1.0020x (6) | 0.8999x (6) |
| Gated delta rule | Forward + backward | 6 | 1.3019x (6) | 0.7414x (6) | 0.7442x (6) |
| KDA | Forward | 6 | 2.0209x (6) | 1.2018x (6) | 0.9637x (6) |
| KDA | Forward + backward | 6 | 1.4167x (6) | 1.0104x (6) | 0.9122x (6) |
| KDA fused | Forward | 6 | 1.8800x (6) | 1.1859x (6) | 0.9406x (6) |
| KDA variable length | Forward | 6 | 1.0100x (6) | 1.0712x (6) | 0.9439x (6) |
