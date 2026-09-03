# Sources: the published Helion posts

The voice and positioning analysis in `01`–`03` is built on the eight Helion posts
published before this one. Their full text is deliberately not committed here -- this
repo is public and the articles are not ours to redistribute -- so this file is the
index. Each is linked; the analysis of each is in the notes that follow.

| Date | Post | Byline |
|---|---|---|
| 2025-10-22 | [Helion: A High-Level DSL for Performant and Portable ML Kernels](https://pytorch.org/blog/helion/) | PyTorch Team at Meta |
| 2026-02-03 | [Portable Paged Attention in Helion](https://pytorch.org/blog/portable-paged-attention-in-helion/) | Burkhard Ringlein and the vLLM Team, IBM Research |
| 2026-02-24 | [Accelerating Autotuning in Helion with Bayesian Optimization](https://pytorch.org/blog/accelerating-autotuning-in-helion/) | Ethan Che, Oguz Ulgen, Max Balandat, Jongsok Choi, Jason Ansel |
| 2026-04-07 | [PyTorch Foundation Welcomes Helion as a Foundation-Hosted Project](https://pytorch.org/blog/pytorch-foundation-welcomes-helion-as-a-foundation-hosted-project-to-standardize-open-portable-and-accessible-ai-kernel-authoring/) | PyTorch Foundation (press release) |
| 2026-06-10 | [Portable vLLM Model Inference Kernels in Helion](https://pytorch.org/blog/portable-vllm-model-inference-kernels-in-helion/) | Sean Chen (Red Hat), Yanan Cao (Meta) |
| 2026-06-18 | [From Minutes to Seconds: LLM-Guided Autotuning for Helion Kernels](https://pytorch.org/blog/from-minutes-to-seconds-llm-guided-autotuning-for-helion-kernels/) | Jongsok Choi, Ethan Che, Jason Ansel, Oguz Ulgen |
| 2026-07-10 | [Towards Free Normalization: Fusing Normalization into GEMM and Attention Kernels](https://pytorch.org/blog/towards-free-normalization-fusing-normalization-into-gemm-and-attention-kernels/) | Jacky (Junqing) Zhou, Hongtao Yu, and others |
| 2026-07-23 | [Helion on TPU: Towards Hardware Heterogeneous Kernel Authoring](https://pytorch.org/blog/helion-on-tpu-towards-hardware-heterogeneous-kernel-authoring/) | Dunfan Lu, Yifei Xu, Jongsok Choi, Ethan Che, Oguz Ulgen, Jason Ansel, and others |

Which notes lean on which:

- `01` — the introduction post and the foundation press release, for house style, the
  canonical description of Helion's configuration space, and the acknowledgements
  convention. Also enumerates the series.
- `02` — the two autotuning posts (LFBO and LLM-guided). This is the positioning note:
  every sentence in either post that quantifies what autotuning costs, quoted, plus
  what each technique does and does not remove.
- `03` — the TPU and paged-attention posts, for how the series presents a compiler
  mechanism and how it presents per-kernel performance tables.

Where those notes refer to a `_source-*.md` file, they mean a local transcript taken
while reading. Those are not in this repo; follow the link above instead.

## Reproduction note

Fetching these pages needs care. On a cold request `pytorch.org` served the wrong
article body for at least two of these slugs -- the returned HTML's own
`<link rel="canonical">` pointed at a different post than the one requested. A
cache-busting query string plus a `Cache-Control: no-cache` header returned the correct
article. If you re-fetch any of these, check the canonical link against the slug you
asked for before trusting the content.
