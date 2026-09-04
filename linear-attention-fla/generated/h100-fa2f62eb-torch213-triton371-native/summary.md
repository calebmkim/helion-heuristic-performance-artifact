# Native Linear-Attention Performance

This is Helion's native block-ordered timing path, not the controlled sample-interleaved four-arm measurement. It compares FLA with the single active Helion configuration: **H100 AOT (compat-repaired)**.

Environment: NVIDIA H100 80GB HBM3, Helion `fa2f62eb686e`, FLA `0.5.2` (`6bd90692588c`).

AOT compatibility repair: `chunk_fwd_o_diag_anchored_varlen_helion` changes `range_num_stages: [0]` to `[]`. This removes a stale zero-valued setting for a range that no longer accepts it; all performance-bearing AOT fields remain unchanged.

Validation: 96/96 cells were accepted by the native accuracy output; 96/96 Helion latencies were positive.

Values are higher-is-better geometric means of the native `helion_speedup` metric (`FLA latency / Helion latency`). Counts are accepted common cells.

| Population | Cells | Accuracy accepted | Positive latency | Helion / FLA |
|---|---:|---:|---:|---:|
| Overall | 96 | 96 | 96 | 1.2860x (96) |
| Forward | 54 | 54 | 54 | 1.2409x (54) |
| Forward + backward | 42 | 42 | 42 | 1.3464x (42) |

## Per Kernel

| Kernel | Mode | Cells | Accuracy accepted | Positive latency | Helion / FLA |
|---|---|---:|---:|---:|---:|
| Vanilla linear attention | Forward | 6 | 6 | 6 | 1.1202x (6) |
| Vanilla linear attention | Forward + backward | 6 | 6 | 6 | 1.5181x (6) |
| Simple GLA | Forward | 6 | 6 | 6 | 1.1170x (6) |
| Simple GLA | Forward + backward | 6 | 6 | 6 | 1.6373x (6) |
| Retention | Forward | 6 | 6 | 6 | 1.5395x (6) |
| Retention | Forward + backward | 6 | 6 | 6 | 1.4525x (6) |
| Full GLA | Forward | 6 | 6 | 6 | 1.4593x (6) |
| Full GLA | Forward + backward | 6 | 6 | 6 | 1.8887x (6) |
| Delta Rule | Forward | 6 | 6 | 6 | 1.2453x (6) |
| Delta Rule | Forward + backward | 6 | 6 | 6 | 1.0239x (6) |
| Gated delta rule | Forward | 6 | 6 | 6 | 1.1211x (6) |
| Gated delta rule | Forward + backward | 6 | 6 | 6 | 1.0251x (6) |
| KDA | Forward | 6 | 6 | 6 | 1.2305x (6) |
| KDA | Forward + backward | 6 | 6 | 6 | 1.1208x (6) |
| KDA fused | Forward | 6 | 6 | 6 | 1.2711x (6) |
| KDA variable length | Forward | 6 | 6 | 6 | 1.1369x (6) |
