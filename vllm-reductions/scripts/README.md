# Starter Scripts

These scripts form an adaptable pipeline:

1. `inspect_vllm.py` verifies the matching vLLM CUDA/C++ implementations.
2. `probe_vllm_cuda.py` locates, loads, and validates the optional compiled
   extension.
3. `discover_manifest.py` finds exact architecture-specific AOT keys and
   selects the requested profile.
4. `run_benchmark.py` isolates each cell in a child process, captures every
   available arm there, and times all of them together.
5. `summarize_results.py` reads the combined three- or four-arm result and
   writes higher-is-better per-kernel geomeans plus raw per-cell CSV.
6. `plot_results.py` renders the same geomeans with the selected reference at
   `1.00x`.
7. `starter.sh` auto-selects the four-arm CUDA path or three-arm AOT path.

For the primary result, `starter.sh` requires PyTorch `2.13.0+cu132` and
Triton `3.7.1`. Override the required values only when intentionally creating
a separately labeled historical or diagnostic dataset.

vLLM 0.24.0's wheel declares `torch==2.11.0`, so do not install its
dependencies into the benchmark environment. Download the wheel with
`pip download --no-deps`, extract it, and set `VLLM_EXTENSION_PATH` to
`vllm/_C_stable_libtorch.abi3.so`. The probe loads that stable-ABI binary
directly and records its distribution version and SHA-256 when wheel metadata
is available beside it.

The benchmark runner deliberately reuses each target Helion module's
`_bench_shapes()`, `correctness_check()`, `main()` call construction, and Torch
reference. This keeps argument semantics and mutation behavior owned by Helion
while allowing the population and selected arm to be controlled externally.

Timing is CUDA Graph device time measured by CUDA events with a cold L2 cache.
All arms share one balanced rotating/reversed timing loop. CPU launch overhead
is deliberately excluded.

The scripts are not a stable interface. Copy adapted versions into the result
directory when current project APIs require changes.
