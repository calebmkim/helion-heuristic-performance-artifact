# Compile-Time Heuristics: blog post and supporting material

Draft of the PyTorch blog post on Helion's compile-time config heuristics, plus the
figures it uses and the notes the numbers were checked against.

| Path | What it is |
|---|---|
| `BLOG_POST.md` | The draft. An HTML comment at the end lists every number with its source and the open questions before publication. |
| `figures/` | The figures the draft references, with their generating scripts. |
| `research/` | Notes from a verification pass: every headline number re-derived from raw result files, the heuristic implementation read against the traces, and the voice/positioning of the earlier posts in the series. `research/00-sources.md` links the eight published Helion posts the voice notes are based on. |

## Regenerating the figures

The two results comparisons are generated from this repo's own data by the scripts
that sit with each comparison:

    cd ../linear-attention-fla/scripts
    python plot_blog_figures.py --h100 ../generated/h100-same-process/results.json \
        --b200 <path>/linear_attention_e2e_per_cell.csv \
        --outdir ../../blog/figures

    cd ../vllm-reductions/scripts
    python plot_blog_figures.py --summary ../generated/h100-pr3551-curated/summary.json \
        --gpu H100 --outdir ../../blog/figures

The rest are self-contained:

    cd figures/diagrams && python mechanism.py && python framing.py   # diagrams A1, A2, B, C, D, E
    cd figures && python seed_trajectory.py                            # autotune trajectory

`seed_trajectory.py` reads per-cell autotuner CSVs from the seeded-search run, which
are not in this repo; its paths point at the machine it was run on and need editing to
reproduce elsewhere.

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
