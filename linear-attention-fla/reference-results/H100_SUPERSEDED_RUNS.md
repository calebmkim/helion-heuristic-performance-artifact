# Superseded H100 Linear-Attention Runs

Do not use the H100 tables or graphs that preceded
`generated/h100-same-process/`.

The first PR #3546 artifact set `HELION_AUTOTUNE_EFFORT=none` for the AOT arm.
Helion therefore resolved the compiler default before consulting
`AOTAutotuneCache`, making the reported AOT measurements a duplicate seed run.

A subsequent `h100-corrected-aot` run fixed AOT selection but retained the
separate-arm timing method. Each Helion arm ran in a different long-lived
process with a separately timed FLA denominator. Moreover, the manifest placed
all dense backward rows at odd indices, and the adapter reversed odd rows, so
every backward cell measured the complete FLA block before the Helion block.
That left process drift and fixed-order effects in the ratios.

The replacement materializes exact configs in isolated discovery processes,
then replays default, seed, and AOT beside FLA in one fresh process per cell.
Cold-L2 CUDA-event samples are interleaved in rotated forward/reverse order,
with equal sample counts and backward gradient clearing outside the timed
region. The old generated data was removed from the current tree; Git history
retains it for forensic comparison.
