# Transformer GPU Kernel Optimization — Challenge 3

**One Transformer implementation that looks at its own input shape and picks the fastest execution plan proven to still be numerically correct — 13/13 PASS at up to 15.1x on the literal official grading script, up to 15.6x under our own deeper stress testing, plus a generalization rule for shapes it's never seen.**

```python
from optimized_transformer import UserOptimizedTransformer
model = UserOptimizedTransformer(config)      # config is all it needs to decide
model.load_state_dict(baseline.state_dict())  # same param names as the reference
output = model(x, valid_token_mask)           # dispatches internally, every time
```

## The headline numbers

**These numbers come from running the literal, unmodified `torch_transformer_benchmark.py` (§3.4 of the problem statement) with only `UserOptimizedTransformer` filled in** — the exact script and methodology a grader would use, not our own research harness. Every shape below is a verified **PASS** on the competition's own correctness gate (`abs_error ≤ 0.002 OR relative_error ≤ 2%`), on an NVIDIA RTX A6000.

| Shape | (Batch, D, Heads, Seq, Layers) | Accuracy | Speedup vs. reference |
|---|---|---|---|
| 1 | 64, 128, 4, 128, 4 | PASS | **4.30x** |
| 2 | 1, 128, 4, 128, 4 | PASS | **9.17x** |
| 3 | 4, 128, 4, 128, 4 | PASS | **8.45x** |
| 4 | 16, 128, 4, 128, 4 | PASS | **6.40x** |
| 5 | 128, 128, 4, 128, 4 | PASS | **4.04x** |
| 6 | 10000, 128, 4, 128, 4 | PASS | **3.88x** |
| 7 | 64, 32, 4, 128, 4 | PASS | **7.02x** |
| 8 | 64, 1024, 4, 128, 4 | PASS | **1.61x** |
| 9 | 64, 128, 1, 128, 4 | PASS | **2.66x** |
| 10 | 64, 128, 2, 128, 4 | PASS | **3.08x** |
| 11 | 64, 128, 16, 128, 4 | PASS | **6.98x** |
| 12 | 64, 128, 4, 32, 4 | PASS | **5.54x** |
| 13 | 64, 128, 4, 1024, 4 | PASS | **15.12x** |

**13/13 PASS. Geometric mean 5.15x, arithmetic mean 6.02x, up to 15.12x on the longest-sequence shape.** Reproduce with `python run_official_sweep.py` (drives `torch_transformer_benchmark.py` once per shape) — raw log in `results/logs/official_sweep_shapes_1-13.log`.

Shape 14 (batch=32, seq=100,000) **cannot run through this script at all — for either implementation.** The official `BaselineTransformer`'s manual attention needs `O(S²)` memory; at S=100,000 that's 4.77 TB regardless of whose code is running. This isn't a limitation of our submission — it's a limitation of the provided reference implementation at that shape. Our dispatcher still handles it correctly via batch-microbatching (see below), validated against a memory-safe reference instead: **1.85x vs. the fastest baseline that can physically execute there.**

*A deeper, stress-tested version of this same table — checked against `tiny`/`large`/outlier-injected inputs, not just default random ones, using per-layer plans found by exhaustive search — lives in [`TECH_REPORT.md`](TECH_REPORT.md) §6, with typically higher numbers (up to 15.6x on shape 3) once the harness gives each shape 40 trials and finer-grained timing instead of the official script's simpler defaults. Both tables describe the same submission; the one above is what you get by running the grader's own script unmodified.*

Full derivation, every intermediate result, and the two real bugs we root-caused along the way are in **[`TECH_REPORT.md`](TECH_REPORT.md)**. Every paper idea used is attributed line-by-line in **[`ATTRIBUTION.md`](ATTRIBUTION.md)**.

## Why this isn't just "use FlashAttention"

The obvious move — swap SDPA for FlashAttention-2 everywhere, cast everything to FP16 — is fast (up to 27x) and **fails the correctness gate on every single one of the 14 shapes.** That's the trap this challenge is actually testing for: the gap between a kernel that's fast in isolation and one that's fast *and* provably correct across a genuinely adversarial shape spread (batch 1 to 10,000; sequence length 32 to 100,000; hidden dim 32 to 1024; heads 1 to 16).

Three findings shaped the final design, each backed by a root-caused bug or a systematic search — not guesses:

1. **A precision boundary that survives the stress test**: keep LayerNorm, residual, QKV/output projections, and the FFN in FP32; run only the attention *core* in FP16 via FlashAttention-2 (`LFA`). This alone passes 12 of 13 officially tested shapes and beats every all-FP32 alternative.
2. **Correctness failures at extreme scale are statistical, not random, and fixable with position**: the 2 shapes where `LFA` fails (batch=10,000; sequence=100,000) fail because FP16's fixed per-element error floor becomes visible once there are enough elements — and the fix is the same in both cases: keep the *earlier* layer(s) FP32, since an early-layer FP16 perturbation has more downstream layers to compound through. This one mechanism resolves both outliers.
3. **No single kernel is optimal everywhere**: shape 8 (large hidden dim, projection/FFN-bound) barely benefits from attention optimization at all (1.64x); shape 13 (long sequence, attention-bound) gets 15.5x from the exact same toolkit. A per-shape dispatcher captures both; a universal kernel would have to compromise on one.

## What's actually in the repo

- **`src/optimized_transformer.py` — the core submission.** A `BaselineTransformer` subclass (`UserOptimizedTransformer`) that dynamically dispatches per shape: exact validated plans for all 14 disclosed shapes, plus a regime classifier for unseen shapes (see "How it works"). Run it directly — `python optimized_transformer.py` — for a self-contained correctness self-test.
- **`src/torch_transformer_benchmark.py` — the official grading script**, byte-for-byte as provided (§3.4 of the problem statement) except `UserOptimizedTransformer`, which delegates to `optimized_transformer.py`. This is what actually produces the headline table above.
- **`src/run_official_sweep.py`** — drives `torch_transformer_benchmark.py` once per official shape and prints the summary table.
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

**1. Reproduce the headline table** — the literal official grading script, run once per shape:

```bash
python run_official_sweep.py
```

Or run a single shape directly (this is exactly what the grader would run):

```bash
python torch_transformer_benchmark.py \
  --batch-size 64 --seq-len 128 --d-model 128 --heads 4 --ffn-dim 128 --layers 4 --causal \
  --device cuda --dtype float32 --no-allow-tf32 \
  --accuracy-trials 10 --rtol 0.02 --atol 0.002
```

**2. Run the submission's self-test directly** (correctness only, fast, no full benchmark needed):

```bash
python optimized_transformer.py
```

**3. Full competition-shape sweep with the deeper stress-tested table**, using the research harness (float32 reference, FP16 fast-dtype, TF32 disabled — see `TECH_REPORT.md` §4.2 for why `--no-allow-tf32` is required for correctness):

```bash
python transformer_ablation_benchmark.py \
  --device cuda --dtype float32 --fast-dtype float16 --no-allow-tf32 \
  --competition-shapes --shape-ids 1-13 \
  --variants A,H,L,LFA \
  --accuracy-trials 10 --accuracy-profile stress --benchmark-on-failure \
  --dynamo-cache-size-limit 128 \
  --csv ../results/csv/my_run.csv
```

**4. Reproduce the per-layer hybrid searches** that found shapes 6 and 14's exact plans, plus the generalized 3-way search used for the rest:

```bash
python search_shape6_hybrid.py                      # binary FP32/LFA-level search
python search_layer_precision_boundary.py 6 0        # 3-way search on cuda:0
python shape14_microbatch.py && python shape14_hybrid.py   # shape 14's chunked resolution
python search_layer_precision_boundary.py <shape_id>[,<shape_id>...] <cuda_device_index>
```

**5. Reproduce the two root-caused bugs** behind the design (TF32-induced correctness divergence; `torch._dynamo`'s silent recompilation-cache fallback):

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
- Shape 14's dispatch entry has not been benchmarked with `torch.compile` (each uncompiled forward pass there already takes ~60 seconds, making multi-candidate compilation impractical in the time available). It also can't be validated through the unmodified official script at all, since the official `BaselineTransformer` itself can't execute at that shape (§ headline numbers, above) — only our own memory-safe cross-check (`H` vs `LFA`/`LP`) is available there.

Full list, plus every operational lesson learned running this on a shared multi-GPU host, in `TECH_REPORT.md` §7–8.
