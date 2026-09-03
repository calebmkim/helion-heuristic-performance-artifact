# Linear-Attention Result Schema

The durable result is one JSON object. Paths, package versions, and optional
diagnostics may be added, but the fields below are the stable contract.
Large archived results may use gzip transport with a `.json.gz` suffix; their
uncompressed content follows the same contract.

## Top Level

```json
{
  "schema_version": 1,
  "benchmark": "linear-attention-fla",
  "provenance": {
    "helion": {"commit": "<sha>", "dirty": false},
    "fla": {"commit": "<sha-or-null>", "version": "<version>", "dirty": false},
    "python": "<version>",
    "torch": "<version>",
    "triton": "<version>",
    "cuda": "<version>",
    "gpu": "<name>",
    "compute_capability": [9, 0],
    "outer_rounds": 3
  },
  "cells": []
}
```

`provenance` is intentionally extensible. Include imported module paths, the
Python executable, GPU state, environment controls, adapter hash, and command
line when available.

## Cell

```json
{
  "id": "simple_gla::forward::B1_T8192_H96_D128",
  "variant": "simple_gla",
  "mode": "forward",
  "shape": {
    "name": "B1_T8192_H96_D128",
    "B": 1,
    "H": 96,
    "T": 8192,
    "D": 128,
    "DV": 128
  },
  "arms": {
    "default": {},
    "fla_triton": {},
    "heuristic": {},
    "pre_tuned": {}
  }
}
```

`mode` is `forward` or `forward_backward`. Additional modes are allowed in
future schema versions.

## Arm

Every arm has a status:

```json
{
  "status": "ok",
  "latency_ms": 0.7038,
  "samples_ms": [0.7051, 0.7029, 0.7038],
  "correct": true,
  "selection": {},
  "errors": {}
}
```

Allowed statuses:

| Status | Meaning |
|---|---|
| `ok` | Correct and timed |
| `unsupported` | The implementation intentionally does not support this cell |
| `missing_exact_config` | `pre_tuned` invoked at least one key absent from the generated module |
| `error` | Compilation, execution, reference, correctness, or timing failed |

For `status: "ok"`:

- `latency_ms` must be finite and positive.
- `samples_ms` must contain every positive outer-round sample.
- `correct` must be `true`.
- A `pre_tuned` arm must also contain `"exact": true`.

For any other status, omit `latency_ms` or set it to `null`. Preserve an
`error`, `reason`, or structured diagnostic. A fallback timing may be stored
under a clearly named diagnostic field, but not as the arm latency.

## Selection Evidence

Helion arms should retain one record per invoked kernel:

```json
{
  "selection": {
    "invoked_kernels": [
      {
        "kernel": "chunk_fwd_o_helion",
        "config": {"block_sizes": [1, 128, 32], "num_warps": 4},
        "compiler_default_config": {"block_sizes": [1, 128, 32]},
        "autotuner_heuristics": ["triton_h100_multi_matmul"],
        "aot_call_key": null,
        "aot_exact": null
      }
    ]
  }
}
```

The exact config shape may evolve with Helion. Preserve the full serializable
configuration rather than forcing it into a fixed list of knobs.

Expected arm evidence:

- `default`: no compiler heuristics and no AOT selection.
- `heuristic`: compiler default and heuristic names where they fired.
- `pre_tuned`: call key and exact-match decision for every invocation.
- `fla_triton`: FLA function/module identity and
  `"measured_with": "heuristic"`.

## Aggregation Rules

The summarizer includes only `status: "ok"` and `correct: true` rows.
`pre_tuned` additionally requires `exact: true`.

For each baseline, its pairwise population is the intersection with
`heuristic`. The four-way population is the intersection of all four arms.
Ratios are computed per cell from absolute latency and then geometrically
averaged.

Never encode a missing latency as zero. Never carry an FLA or pre-tuned value
from a different shape.
