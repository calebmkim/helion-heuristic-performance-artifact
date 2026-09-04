# Compile-Time Heuristics: blog post and supporting material

Draft of the PyTorch blog post on Helion's compile-time config heuristics, plus the
figures it uses and the notes the numbers were checked against.

| Path | What it is |
|---|---|
| `BLOG_POST.md` | The draft. An HTML comment at the end lists every number with its source and the open questions before publication. |
| `figures/` | The figures the draft references, with their generating scripts. |
| `data/` | Raw per-cell autotuner trajectories behind Result 2, so that result reproduces without the machine it ran on. |
| `research/` | Notes from a verification pass: every headline number re-derived from raw result files, the heuristic implementation read against the traces, and the voice/positioning of the earlier posts in the series. `research/00-sources.md` links the eight published Helion posts the voice notes are based on. |

## Regenerating the figures

The results comparisons are generated from this repo's own data by the scripts
that sit with each comparison:

    cd ../linear-attention-fla/scripts
    python plot_blog_figures.py \
        --h100 ../generated/h100-fa2f62eb-torch213-triton371-interleaved/results.json \
        --b200 <path>/linear_attention_e2e_per_cell.csv \
        --outdir ../../blog/figures

    cd ../vllm-reductions/scripts
    python plot_blog_figures.py \
        --summary ../generated/h100-fa2f62eb-torch213-triton371-vllm024-curated/summary.json \
        --gpu H100 --outdir ../../blog/figures

    cd ../example-reductions/scripts
    python plot_blog_figures.py \
        --summary ../generated/h100-fa2f62eb-torch213-triton371-liger-mixed/summary.json \
        --gpu H100 --outdir ../generated/blog-figures

    cd ../other-matmul-kernels/scripts
    python plot_blog_figures.py \
        --summary ../generated/h100-fa2f62eb-torch213-triton371/results.json \
        --gpu H100 --outdir ../generated/h100-fa2f62eb-torch213-triton371/blog-figures

The rest are self-contained:

    cd figures/diagrams && python mechanism.py && python framing.py   # diagrams A1, A2, B, C, D, E
    cd figures && python seed_trajectory.py                            # autotune trajectory

`seed_trajectory.py` reads the committed trajectories under `data/seeded-search/`, so it
runs anywhere.

## The seeded-search data

`data/seeded-search/` holds one gzipped CSV per cell per arm -- 38 cells in `no-seed/`
and the same 38 in `expanded/` -- plus `per-cell-summary.json`, the merged per-cell
record the published medians were computed from.

Each CSV is one autotuner run: a row per attempt, with `run_id`, `timestamp_s`,
`config_id`, `generation`, `status`, `perf_ms`, `compile_time_s`, and the full `config`.
`perf_ms` is empty when an attempt failed to compile or failed to produce a timing;
those rows still consumed search budget, which is why the medians in the post count
them. Best-so-far performance for a cell is the running minimum of `perf_ms` over the
rows in order, which is what the trajectory figure plots.

Two things are deliberately excluded. The old-seed arm, since the post compares only
no-seed against the expanded pool. And the Triton and TorchInductor caches the runs
left behind, which are 1.3 GB of `.cubin`/`.ptx`/`.so` and reproduce nothing.

The B200 linear-attention CSV is published in
[`PYTORCH_BLOG_RAW_DATA`](https://github.com/calebmkim/helion/blob/pytorch-blog-heuristics-results/PYTORCH_BLOG_RAW_DATA/linear_attention_e2e_per_cell.csv).
The H100 input is the authoritative same-process, sample-interleaved run in this
repository.

## Notes

`figures/diagrams/` contains six diagrams. The draft currently uses A1, D and E; A2, B
and C are built but unused, kept because they may earn a place later.

The `research/` notes cite absolute paths on the machine where the analysis ran, so the
file references in them will not resolve here. The reasoning and the numbers stand on
their own.
