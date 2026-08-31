# Transformer GPU Kernel Optimization — Challenge 3

An AI-assisted optimization of a Transformer layer's forward pass, built against the "Implement a GPU Kernel for a Transformer Layer" challenge. Instead of one universal kernel, this repo builds a **shape-conditioned dispatcher**: for each of the 14 disclosed test shapes, it searches for the fastest execution plan that still passes the competition's strict per-element correctness gate (`abs_error ≤ 0.002 OR relative_error ≤ 2%`), and only promotes plans that survive an expanded stress-accuracy check (normal / tiny / large / rare-outlier input distributions, not just default random inputs).

**Read `TECH_REPORT.md` first** for the full narrative, environment details, and final results tables. **Read `ATTRIBUTION.md`** for exact paper-by-paper attribution of every optimization idea used. This README covers setup, reproduction, and project meta-info.

## Project overview

- **Baseline**: exact FP32 reference Transformer (separate Q/K/V projections, explicit softmax attention, standard masking) — the ground truth every candidate is checked against.
- **Optimization axes explored**: packed QKV projection, `torch.nn.functional.scaled_dot_product_attention` vs. external FlashAttention-2, no-op mask elimination, `torch.compile`, and a family of selective mixed-precision designs that keep LayerNorm/residual/FFN in FP32 while running only the attention core (or QKV+attention+output-projection) in FP16.
- **Headline result**: `LFA` (FP32 everywhere except an FP16 FlashAttention-2 attention core, compiled) passes 12 of 13 officially-tested shapes and delivers 1.19x–15.12x speedup — categorically ahead of the best all-FP32 option. A per-layer hybrid search then improves on `LFA` for 10 of those 12 shapes further (up to +44%), and resolves the 2 shapes where plain `LFA` fails (shape 6: extreme batch size; shape 14: extreme sequence length) with position-aware per-layer precision mixes.
- **Key finding**: correctness failures at extreme batch size or sequence length are a *statistical scale effect* (FP16's fixed per-element error floor becomes visible once total element count is large enough), not a flaw in a specific design — and the fix (keep FP16 layers later in the stack, not earlier) is the same mechanism in both cases.

Full results tables: `TECH_REPORT.md` §6. Full paper attribution: `ATTRIBUTION.md`.

## Repository layout

```
├── README.md                 (this file)
├── TECH_REPORT.md             environment, methodology, chronological findings, final results
├── ATTRIBUTION.md              exact paper-by-paper attribution for every optimization
├── src/
│   ├── transformer_ablation_benchmark.py   main harness: baseline + ~35 candidate variants,
│   │                                       accuracy/timing infrastructure, competition-shape sweep,
│   │                                       CSV/dispatch-JSON export
│   ├── search_shape6_hybrid.py             binary (FP32-SDPA / LFA-level) per-layer brute force
│   ├── search_layer_precision_boundary.py  generalized 3-way (none/LFA-level/VFA-level) per-layer
│   │                                       brute force, parameterized by shape + GPU device
│   ├── shape14_microbatch.py               batch-microbatched full-length (S=100000) timing
│   ├── shape14_hybrid.py                   2-layer per-layer hybrid resolution for shape 14
│   └── diagnose_sdpa_divergence.py         isolates the TF32-vs-SDPA numerical divergence (see
│                                            TECH_REPORT.md §4.2)
└── results/
    ├── csv/        every ablation/sweep run's structured CSV output
    ├── logs/        full stdout logs for every run referenced in TECH_REPORT.md
    └── dispatch/    machine-readable best-plan-per-shape dispatch tables (JSON)
```

## Setup and installation

Tested on: Rocky Linux 9.8, Python 3.10, PyTorch 2.5.1+cu121, CUDA 12.1, 8× NVIDIA RTX A6000 (Ampere, compute capability 8.6). Full environment details in `TECH_REPORT.md` §1.

```bash
# 1. Create/activate an environment with a CUDA-enabled PyTorch matching your driver.
conda create -n transformer-kernel python=3.10 -y
conda activate transformer-kernel
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# 2. FlashAttention-2 (required for the LFA/NFA/VFA/P family of variants).
#    Try the direct pip install first:
pip install flash-attn --no-build-isolation
#    If it fails on a cross-filesystem "Invalid cross-device link" error (a pip cache-move
#    bug, not a compile failure), download the exact matching wheel it names in that error
#    directly from https://github.com/Dao-AILab/flash-attention/releases and `pip install`
#    the local file instead -- no CUDA toolkit / nvcc install should be necessary if a
#    prebuilt wheel matches your torch/CUDA/Python combination.

# 3. Everything else PyTorch already depends on (no extra requirements.txt needed beyond
#    torch + flash-attn for the full variant set; variants that call SageAttention -- "Q" --
#    are optional and will report ERROR gracefully if the `sageattention` package isn't
#    installed).
```

## Steps to reproduce

All commands assume `cd src/` first.

**1. Core factorial ablation (A–H) on the default dev shape:**

```bash
python transformer_ablation_benchmark.py --device cuda --dtype float32 --variants all \
  --csv ../results/csv/ablation_core_A-H.csv
```

**2. Full competition-shape sweep with the validated `H`/`L`/`LFA` configuration** (float32 reference, FP16 fast-dtype, TF32 disabled — see `TECH_REPORT.md` §4.2 for why `--no-allow-tf32` is required for correctness):

```bash
python transformer_ablation_benchmark.py \
  --device cuda --dtype float32 --fast-dtype float16 --no-allow-tf32 \
  --competition-shapes --shape-ids 1-13 \
  --variants A,H,L,LFA \
  --accuracy-trials 10 --accuracy-profile stress --benchmark-on-failure \
  --dynamo-cache-size-limit 128 \
  --csv ../results/csv/my_run.csv
```

**3. Shape 6's per-layer hybrid search** (binary FP32/LFA-level, then the extended 3-way version):

```bash
python search_shape6_hybrid.py                      # binary search, writes to stdout
python search_layer_precision_boundary.py 6 0        # 3-way search on cuda:0
```

**4. Shape 14 (S=100,000) — requires batch microbatching, see `TECH_REPORT.md` §4.8:**

```bash
python shape14_microbatch.py     # finds the largest feasible batch chunk, times H/LFA
python shape14_hybrid.py         # resolves LFA's failure with a 2-layer FP32/FP16 hybrid
```

**5. Per-layer 3-way search across any other shape(s), optionally on a specific GPU:**

```bash
python search_layer_precision_boundary.py <shape_id>[,<shape_id>...] <cuda_device_index>
```

This is parameterized specifically so it can be run across multiple shapes/GPUs in parallel — see `TECH_REPORT.md` §4.10 for how the full 12-shape search was distributed across an idle GPU fleet.

**6. TF32-divergence diagnostic** (reproduces the root-cause isolation in `TECH_REPORT.md` §4.2):

```bash
python diagnose_sdpa_divergence.py
```

All scripts print human-readable progress and PASS/FAIL/accuracy numbers directly to stdout; CSV-producing runs also write structured output to the path given by `--csv`.

## Correctness criterion

Every candidate is checked per output element against:

```text
abs(candidate - reference) <= 0.002   OR   abs(candidate - reference) <= 0.02 * abs(reference)
```

across 40 trials per configuration (10 each of `normal`, `tiny`, `large`, and rare-`outlier` input distributions — see `--accuracy-profile stress`). A "PASS" anywhere in this repo means **zero failed elements** across all 40 trials, not just the default random-input case.

## AI tools and development process

This project was built interactively with Claude (Claude Code), used for: reading and extending the provided benchmark harness, writing the search scripts in `src/`, diagnosing two non-obvious bugs (TF32-induced numerical divergence, and `torch._dynamo`'s silent recompilation-cache fallback — both documented in `TECH_REPORT.md` §4.2–4.3), designing and running the per-layer precision-boundary searches, and parallelizing the shape-by-shape search across the available multi-GPU host. See `ATTRIBUTION.md` for exactly which optimization ideas trace to which supplied research papers versus which are original synthesis from this project's own measurements.

## Limitations and what we'd improve with more time

See `TECH_REPORT.md` §8 for the full list. In short:

- The per-layer search menu (none / LFA-level / VFA-level) is a fixed 3-way choice per layer, not a fully independent search over QKV precision × out-proj precision × attention backend per layer — a larger search could plausibly do better, especially on the two shapes (7, 11) where no improvement over plain `LFA` was found.
- No custom CUDA/Triton kernels were written; all gains are from PyTorch-level operator/precision/layout choices and `torch.compile`. Concrete next steps (LayerNorm+QKV fusion, FFN fusion, persistent whole-model kernels, tile/warp/pipeline tuning) are identified with paper citations in `ATTRIBUTION.md` §15–18 but not implemented.
- Shape 14's dispatch entry has not been benchmarked with `torch.compile` (each uncompiled forward pass there already takes ~60 seconds, making multi-candidate compilation impractical in the time available).
- The 14-shape table used here should be cross-checked against the official competition appendix before final submission.

## Team member contributions

Single-contributor project; see `ATTRIBUTION.md` for the AI-assistance and paper-attribution breakdown in place of a multi-person contribution table.
