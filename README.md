# Transformer GPU Kernel Optimization — Challenge 3

**One Transformer implementation that looks at its own input shape and picks the fastest execution plan proven to still be numerically correct — up to 15.6x faster than the reference, on every one of the 14 disclosed test shapes, plus a generalization rule for shapes it's never seen.**

```python
from optimized_transformer import UserOptimizedTransformer
model = UserOptimizedTransformer(config)      # config is all it needs to decide
model.load_state_dict(baseline.state_dict())  # same param names as the reference
output = model(x, valid_token_mask)           # dispatches internally, every time
```

## The headline numbers

Every row below is a **verified pass** on the strict competition gate — `abs_error ≤ 0.002 OR relative_error ≤ 2%` per output element, checked across **40 stress trials** per shape (10 each of `normal`, `tiny`, `large`, and rare-`outlier` input distributions — not just default random inputs). Speedup is against the exact FP32 reference Transformer, on an NVIDIA RTX A6000.

| Shape (B, S, D, Heads) | Winning plan | Speedup vs. reference |
|---|---|---|
| 1 — 64, 128, 128, 4 | `LLVV` | **4.98x** |
| 2 — 1, 128, 128, 4 | `LLVV` | **15.23x** |
| 3 — 4, 128, 128, 4 | `LVLV` | **15.62x** |
| 4 — 16, 128, 128, 4 | `LLVV` | **11.54x** |
| 5 — 128, 128, 128, 4 | `LLLV` | **4.33x** |
| 6 — 10000, 128, 128, 4 | `PLLV` | **3.92x** |
| 7 — 64, 128, 32, 4 | `LFA` (all-FP16 core) | **10.73x** |
| 8 — 64, 128, 1024, 4 | `PVPV` | **1.64x** |
| 9 — 64, 128, 128, 1 | `LLVV` | **3.42x** |
| 10 — 64, 128, 128, 2 | `PLVV` | **3.63x** |
| 11 — 64, 128, 128, 16 | `LFA` (all-FP16 core) | **7.96x** |
| 12 — 64, 32, 128, 4 | `PVLV` | **9.24x** |
| 13 — 64, 1024, 128, 4 | `LLLV` | **15.50x** |
| 14 — 32, 100000, 1024, 16 | `LP` (microbatched) | **1.85x** vs. compiled-FP32 |

`L`/`P`/`V` per letter = which precision level that layer runs at (FP16-FlashAttention-2 core / FP32-SDPA / FP16-QKV+attention+out-proj) — see "How it works" below. Full derivation, every intermediate result, and the two real bugs we root-caused along the way are in **[`TECH_REPORT.md`](TECH_REPORT.md)**. Every paper idea used is attributed line-by-line in **[`ATTRIBUTION.md`](ATTRIBUTION.md)**.

## Why this isn't just "use FlashAttention"

The obvious move — swap SDPA for FlashAttention-2 everywhere, cast everything to FP16 — is fast (up to 27x) and **fails the correctness gate on every single one of the 14 shapes.** That's the trap this challenge is actually testing for: the gap between a kernel that's fast in isolation and one that's fast *and* provably correct across a genuinely adversarial shape spread (batch 1 to 10,000; sequence length 32 to 100,000; hidden dim 32 to 1024; heads 1 to 16).

Three findings shaped the final design, each backed by a root-caused bug or a systematic search — not guesses:

1. **A precision boundary that survives the stress test**: keep LayerNorm, residual, QKV/output projections, and the FFN in FP32; run only the attention *core* in FP16 via FlashAttention-2 (`LFA`). This alone passes 12 of 13 officially tested shapes and beats every all-FP32 alternative.
2. **Correctness failures at extreme scale are statistical, not random, and fixable with position**: the 2 shapes where `LFA` fails (batch=10,000; sequence=100,000) fail because FP16's fixed per-element error floor becomes visible once there are enough elements — and the fix is the same in both cases: keep the *earlier* layer(s) FP32, since an early-layer FP16 perturbation has more downstream layers to compound through. This one mechanism resolves both outliers.
3. **No single kernel is optimal everywhere**: shape 8 (large hidden dim, projection/FFN-bound) barely benefits from attention optimization at all (1.64x); shape 13 (long sequence, attention-bound) gets 15.5x from the exact same toolkit. A per-shape dispatcher captures both; a universal kernel would have to compromise on one.

## What's actually in the repo

- **`src/optimized_transformer.py` — the submission.** A `BaselineTransformer` subclass (`UserOptimizedTransformer`) that dynamically dispatches per shape: exact validated plans for all 14 disclosed shapes, plus a regime classifier for unseen shapes (see "How it works"). Run it directly — `python optimized_transformer.py` — for a self-contained correctness self-test.
- **`src/transformer_ablation_benchmark.py` — the research harness.** ~35 candidate variants (packed QKV, SDPA vs. FlashAttention-2, mask-skip, `torch.compile`, 8 different precision-boundary designs), full accuracy/timing infrastructure, and the competition-shape sweep this was all validated with.
- **`src/search_*.py`, `src/shape14_*.py` — the search scripts** that found every plan in the table above, parameterized to run in parallel across a multi-GPU host.
- **`results/`** — every CSV, log, and dispatch table backing every number in this README and the tech report.

## How it works

```
                          input config (known at construction time)
                                        │
                                        ▼
                         exact match in the 14-shape table?
                          │ yes                    │ no
                          ▼                         ▼
                 use the brute-force-      classify by scale:
                 searched winning plan     total elements = B·S·D
                                                │
                                  ┌─────────────┴─────────────┐
                                  │ ≤ 20M (typical)            │ > 20M (large-scale)
                                  ▼                            ▼
                         full LFA: every layer         conservative hybrid: first
                         FP16 FlashAttention-2          ~25% of layers FP32 SDPA,
                         core, FP32 everywhere else     rest FP16 FlashAttention-2
                                        │
                                        ▼
                    at forward() time: non-trivial padding mask, no CUDA,
                    or flash-attn missing?  →  fall back to plain FP32 SDPA
                    (always correct, never crashes, never silently wrong)
```

The "~25% of layers FP32" ratio for unseen large-scale shapes isn't arbitrary: applying it to shape 6 (4 layers) yields exactly 1 FP32 layer, matching the brute-force-searched winner; applying it to shape 14 (2 layers) also yields exactly 1 FP32 layer, matching *its* brute-force-searched winner. Two for two on the only shapes that needed conservatism is the best evidence available that it generalizes. This was verified with a real stress test (not just asserted): a synthetic, never-seen shape deliberately sized to the same extreme element count as shape 6 (163.8M elements, batch=5000×6 layers) passes the same 20-trial stress check with zero failed elements, using the regime classifier alone — no exact-match entry for it exists.

## Setup and installation

Tested on: Rocky Linux 9.8, Python 3.10, PyTorch 2.5.1+cu121, CUDA 12.1, 8× NVIDIA RTX A6000 (Ampere, compute capability 8.6). Full environment details in `TECH_REPORT.md` §1.

```bash
# 1. Create/activate an environment with a CUDA-enabled PyTorch matching your driver.
conda create -n transformer-kernel python=3.10 -y
conda activate transformer-kernel
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# 2. FlashAttention-2 (required for every non-"safe-fallback" execution plan).
pip install flash-attn --no-build-isolation
#    If it fails on a cross-filesystem "Invalid cross-device link" error (a pip cache-move
#    bug, not a compile failure), download the exact matching wheel it names in that error
#    directly from https://github.com/Dao-AILab/flash-attention/releases and `pip install`
#    the local file instead -- no CUDA toolkit / nvcc install should be necessary if a
#    prebuilt wheel matches your torch/CUDA/Python combination.
```

If `flash-attn` isn't installed (or CUDA isn't available), `UserOptimizedTransformer` still runs correctly — it automatically falls back to the plain FP32 SDPA path — it just won't be fast.

## Steps to reproduce

All commands assume `cd src/` first.

**1. Run the submission's self-test** (correctness only, fast, no full benchmark needed):

```bash
python optimized_transformer.py
```

**2. Full competition-shape sweep reproducing the headline table**, using the underlying research harness (float32 reference, FP16 fast-dtype, TF32 disabled — see `TECH_REPORT.md` §4.2 for why `--no-allow-tf32` is required for correctness):

```bash
python transformer_ablation_benchmark.py \
  --device cuda --dtype float32 --fast-dtype float16 --no-allow-tf32 \
  --competition-shapes --shape-ids 1-13 \
  --variants A,H,L,LFA \
  --accuracy-trials 10 --accuracy-profile stress --benchmark-on-failure \
  --dynamo-cache-size-limit 128 \
  --csv ../results/csv/my_run.csv
```

**3. Reproduce the per-layer hybrid searches** that found shapes 6 and 14's exact plans, plus the generalized 3-way search used for the rest:

```bash
python search_shape6_hybrid.py                      # binary FP32/LFA-level search
python search_layer_precision_boundary.py 6 0        # 3-way search on cuda:0
python shape14_microbatch.py && python shape14_hybrid.py   # shape 14's chunked resolution
python search_layer_precision_boundary.py <shape_id>[,<shape_id>...] <cuda_device_index>
```

**4. Reproduce the two root-caused bugs** behind the design (TF32-induced correctness divergence; `torch._dynamo`'s silent recompilation-cache fallback):

```bash
python diagnose_sdpa_divergence.py
```

## Correctness criterion

Every candidate is checked per output element against `abs(candidate - reference) <= 0.002 OR abs(candidate - reference) <= 0.02 * abs(reference)`, across 40 trials per configuration (10 each of `normal`, `tiny`, `large`, and rare-`outlier` input distributions). A "PASS" anywhere in this repo means **zero failed elements** across all 40 trials.

## AI tools and development process

Built interactively with Claude (Claude Code): reading and extending the provided benchmark harness, writing every search script, root-causing two non-obvious bugs (TF32-induced numerical divergence, `torch._dynamo`'s silent recompilation fallback — both in `TECH_REPORT.md` §4.2–4.3), designing and running the per-layer precision-boundary searches, parallelizing the search across a multi-GPU host, and building the final dynamic dispatcher. See `ATTRIBUTION.md` for exactly which ideas trace to which supplied research paper versus which are original synthesis from this project's own measurements.

## Limitations and what we'd improve with more time

- The regime classifier for unseen shapes is validated on one deliberately adversarial synthetic case (matched to shape 6's extreme element count), not an exhaustive sweep of the unseen-shape space — a larger fuzz-test across many synthetic shapes would raise confidence further.
- The per-layer search menu (FP32-SDPA / LFA-level / VFA-level) is a fixed 3-way choice per layer, not a fully independent search over QKV precision × out-proj precision × attention backend — a larger search could plausibly do better, especially on shapes 7 and 11 where no mix beat plain `LFA`.
- No custom CUDA/Triton kernels were written; all gains are from PyTorch-level operator/precision/layout choices and `torch.compile`. Concrete next steps (LayerNorm+QKV fusion, FFN fusion, persistent whole-model kernels, tile/warp/pipeline tuning) are identified with paper citations in `ATTRIBUTION.md` §15–18 but not implemented.
- Shape 14's dispatch entry has not been benchmarked with `torch.compile` (each uncompiled forward pass there already takes ~60 seconds, making multi-candidate compilation impractical in the time available).
- The 14-shape table used here should be cross-checked against the official competition appendix before final submission.

Full list, plus every operational lesson learned running this on a shared multi-GPU host, in `TECH_REPORT.md` §7–8.

## Team member contributions

Single-contributor project; see `ATTRIBUTION.md` for the AI-assistance and paper-attribution breakdown in place of a multi-person contribution table.
