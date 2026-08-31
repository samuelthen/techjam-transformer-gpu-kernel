# Technical Report — Transformer GPU Kernel Optimization (Challenge 3)

Status: **shape 6's extended 3-way per-layer search is still running as of this writing** (see the callout in §6.2). Everything else below is complete and validated. This report will be updated with the final shape-6 number once that job finishes.

## 1. Environment

| Component | Detail |
|---|---|
| GPU | 8× NVIDIA RTX A6000 (48 GB each), driver 595.80 |
| CUDA (PyTorch build) | 12.1 |
| cuDNN | 9.1.0 |
| PyTorch | 2.5.1+cu121 |
| FlashAttention | 2.7.4.post1 (installed from the official prebuilt wheel matching torch 2.5.1+cu121/cp310 — no source compile needed) |
| CPU | 2× AMD EPYC 7763 (64 cores / 128 threads each, 256 logical CPUs total) |
| RAM | 1 TB |
| Disk | 7 TB local RAID array (`/local1`, ~657 GB free at time of writing) |
| OS | Rocky Linux 9.8, kernel 5.14 |
| Python | 3.10.20 (conda env `gigatok_vlm`) |

Note: this is a **shared multi-user GPU server**. All experiments below were run on otherwise-idle GPUs, and care was taken (see §7) not to consume shared system RAM irresponsibly — the host was independently under memory pressure from other users' jobs throughout (up to 684 GB in swap) unrelated to this project.

Only the A6000 (Ampere, compute capability 8.6) was targeted. FlashAttention's `flash`/`cudnn` SDPA backends refuse FP32 inputs outright on this hardware ("No available kernel") — Ampere's native FlashAttention/cuDNN fused kernels require FP16/BF16 inputs; `auto` silently falls back to the `efficient` (memory-efficient) kernel for FP32.

## 2. Problem framing

The challenge asks for GPU kernel(s) implementing a Transformer layer that (a) match a PyTorch reference within `abs_error ≤ 0.002 OR relative_error ≤ 2%` per output element, and (b) run faster than the reference, across a disclosed set of shapes that vary batch size, sequence length, hidden dimension, and head count independently (see Appendix — Test Shapes in the problem statement; the 14-shape table used throughout this repo is reproduced in `src/transformer_ablation_benchmark.py::COMPETITION_SHAPES` and should be cross-checked against the official Feishu appendix before final submission, since the problem statement flags that document as authoritative and possibly divergent from any copy circulated elsewhere).

Two design decisions shaped everything downstream:

1. **The correctness gate is per-element and adversarial-input-sensitive.** A configuration that passes on random Gaussian inputs can fail once tested against `tiny`/`large`/`outlier` input distributions (§4), so every "PASS" claim in this repo means it survived that expanded stress profile, not just default random inputs.
2. **One kernel does not dominate every shape.** The 14 disclosed shapes independently vary batch (1 to 10,000), sequence length (32 to 100,000), hidden dim (32 to 1024), and head count (1 to 16) — regimes with fundamentally different bottlenecks (see §6.2, shape 8 vs. shape 13). The final design is a **shape-conditioned dispatcher**, not a single implementation (see `ATTRIBUTION.md` §11–§13 for the paper lineage behind this choice).

## 3. Baseline ablation harness (`src/transformer_ablation_benchmark.py`)

The harness implements an exact FP32 reference Transformer (`BaselineTransformer`: separate Q/K/V projections, explicit `softmax(QKᵀ/√d)V` attention with FP32 softmax, standard causal masking) and a large family of ablation/candidate variants, each independently toggling:

- **QKV layout**: separate projections vs. one packed `Linear(D, 3D)` (§ATTRIBUTION 1).
- **Attention backend**: manual explicit attention / `torch.nn.functional.scaled_dot_product_attention` / external FlashAttention-2 (§ATTRIBUTION 3, 5).
- **Mask handling**: always apply a validity mask vs. skip it when the workload is disclosed as all-valid (§ATTRIBUTION 6).
- **Precision boundary**: FP32 everywhere, whole-model FP16, or a selectively-precision design that keeps LayerNorm/residual/FFN/projections in FP32 while only the attention core (or QKV+attention+out-proj) runs in FP16 (§ATTRIBUTION 8–9).
- **Compilation**: eager vs. `torch.compile` (`reduce-overhead` mode).
- **SDPA backend forcing**: auto / flash / efficient / cuDNN / math.

Every candidate is copied byte-for-byte from the same reference weights (packed-QKV candidates get the reference's Q/K/V weights concatenated, so packed vs. separate projection is a pure layout change, not a different model) and validated against the FP32 reference under an expanded **stress accuracy profile**: `normal`, `tiny` (0.01×), `large` (3×), and `outlier` (0.1% of elements get an injected `N(0,10)` spike) input distributions, 10 trials each (40 trials total), following the FlashAttention-3 numerical stress methodology (§ATTRIBUTION 7).

## 4. Chronological findings (the parts that matter for reproducibility)

This is not a straight-line success story — several findings materially changed the direction, and are recorded here because they are the reason the final design looks the way it does.

### 4.1 Core factorial (A–H)

The original 2³ factorial — packed QKV × SDPA × mask-skip — on the default dev shape (batch=8, seq=128, d_model=512) gives:

| Variant | Median latency | Speedup vs baseline |
|---|---|---|
| A (baseline) | 4.293 ms | 1.00x |
| B (+packed QKV) | 3.824 ms | 1.12x |
| C (+SDPA) | 3.443 ms | 1.25x |
| D (+mask-skip) | 3.175 ms | 1.35x |
| G (SDPA+mask-skip) | 2.076 ms | 2.07x |
| **H (all three)** | **1.764 ms** | **2.43x** |

SDPA and mask-skip compound multiplicatively; mask-skip alone unlocks SDPA's specialized all-valid-causal kernel path, not just a smaller elementwise op.

### 4.2 TF32 was silently causing correctness failures (root-caused, not assumed)

An early full 13-shape sweep with `--dtype bfloat16` showed SDPA-based variants failing accuracy on **every single shape**. Root cause, isolated with a dedicated single-attention-layer diagnostic (`results/logs/` contains the sweep logs; the diagnostic script itself was exploratory and not retained as a standalone file, but its method is: build one `BaselineSelfAttention` and one SDPA-based attention module with identical weights, sweep `torch.backends.cuda.matmul.allow_tf32` on/off and force each SDPA backend, and measure divergence against the FP32 manual reference): **TF32** (enabled by default on Ampere) causes SDPA's internal matmul to round differently than the manual FP32 reference — a single attention layer diverges ~1.1e-3 max-abs with TF32 on vs ~1e-6 with it off. That per-layer gap compounds through a 4-layer stack until it crosses the accuracy gate on a handful of elements. **Fix: `--no-allow-tf32` (or `--allow-tf32=False`)** — with it, `H`/`L` pass strict stress-accuracy on all 13 official shapes (`results/csv/a6000_competition_fp32_fp16_notf32.csv`).

### 4.3 `torch.compile`'s recompilation cache limit silently degrades multi-shape sweeps

`torch._dynamo`'s default per-function cache limit (8 distinct compiled graphs) is exhausted partway through a 13-shape sweep containing compiled variants, after which `torch.compile` **silently falls back to eager execution** for the remaining shapes — no error, no warning that would obviously stand out, just `L`'s timing quietly converging to `H`'s from shape 5 onward. **Fix:** `--dynamo-cache-size-limit` (default raised to 128 in this repo) set via `torch._dynamo.config.cache_size_limit` before any compilation happens. Verified by re-running the full sweep and confirming no `cache_size_limit` warnings and that `L`'s speedup over `H` stays consistent across all 13 shapes (`results/logs/run_LFA_final_sweep.log`).

### 4.4 FlashAttention-2 installation

`pip install flash-attn` found and downloaded the exact matching prebuilt wheel for torch 2.5.1+cu121/cp310 automatically, but failed at a trivial `os.rename` step due to a cross-filesystem link error between pip's cache directory and the download location — unrelated to compilation. Fix: download the wheel directly from the GitHub release and `pip install` the local file. No CUDA toolkit / `nvcc` was ultimately needed for this path (a `cuda-nvcc`/`cuda-cudart-dev` conda install was explored as a fallback for source-compiling FlashAttention, but proved unnecessary once the prebuilt-wheel path was fixed).

### 4.5 LFA: the FP16-FlashAttention-core design

Building on §4.2 and the selective-precision (`U`/`V`/`W`/`X`/`Y`) experiments, we defined **`LFA`**: FP32 LayerNorm/QKV-projection/output-projection/residual/FFN, with only the attention core itself computed in FP16 via external FlashAttention-2, plus `torch.compile` around the whole graph. This is not the naive combination of "FlashAttention + compile" (`NFA`, whole-model FP16 + FlashAttention-2 + compile) — `NFA` is dramatically faster (up to 27x on long sequences) but **fails the stress-accuracy gate on every shape**, the same way whole-model-FP16 (`N`) does; the failure is intrinsic to FP16's 10-bit mantissa, independent of which attention kernel runs it.

`LFA` passes on **12 of 13** officially tested shapes with comfortable margin (0 failed elements out of the full 40-trial stress check), delivering **1.19x–15.12x** speedup depending on shape (full table in §6.1) — categorically ahead of the best all-FP32 option (`L`: 1.20x–13.89x).

### 4.6 VFA (more aggressive: QKV + attention + out-proj all FP16) fails everywhere

To confirm `LFA` wasn't leaving free accuracy margin on the table, we tested `VFA` (the more aggressive precision boundary — everything but LayerNorm/residual/FFN in FP16) across all 13 shapes. **It fails on every single shape**, confirming `LFA`'s precision boundary is close to the practical ceiling for a *uniform, whole-model* choice (`results/csv/a6000_vfa_boundary_check.csv`).

### 4.7 Shape 6 (batch=10,000): the outlier

`LFA` fails shape 6 specifically — 4–5 failed elements out of ~6.55 billion per the 40-trial check. Both `U` (the SDPA-based version of the same precision boundary) and `LFA` fail there, at nearly identical, tiny failure counts, which is the signature of a **statistical boundary effect**: FP16's ~1e-3 error floor, combined with shape 6's enormous batch size (≈160M+ output elements per trial), makes a handful of rare per-element failures nearly certain to appear, even though the *design* is otherwise sound (it passes cleanly everywhere else). This is not a flaw specific to FlashAttention or to shape 6's particular dimensions — it is what happens whenever total element count grows large enough to sample the tail of a fixed per-element error distribution (the same mechanism reappears at shape 14's extreme sequence length; see §4.8).

### 4.8 Shape 14 (S=100,000): a second scale wall, plus a genuine memory limit

Shape 14 could not be tested with the standard harness path at all: a single FP32 activation tensor at full batch (32) and `S=100,000` is 12.2 GB, and `LFA`'s packed-QKV output is 3× that — a genuine memory wall (not a bug) that has nothing to do with attention algorithm choice, since FlashAttention avoids the *O(S²)* score matrix but not the *O(B·S·D)* activations. **Fix:** batch-microbatch to the largest chunk that fits (chunk=8 of 32, found empirically; chunk=16 and 32 OOM). Separately, the *exact* FP32 manual-attention reference is unconditionally infeasible at this length regardless of chunk size (its own `S×S` score matrix alone needs 4.77 TB at any batch size) — the same reason the harness's own "long-safe" mode validates only a prefix and uses an SDPA-based memory-safe variant as the full-length reference, which we mirrored (cross-checking `LFA` against `H`, both already prefix-validated, rather than against the infeasible exact baseline).

At the true full length, `LFA` (both of shape 14's 2 layers FP16) **diverges from `H`** — 33,910 failed elements out of ~9.83 billion — the same scale-driven statistical effect as shape 6, just triggered by sequence length instead of batch size. A 2-layer binary hybrid search (§4.9) resolved this.

### 4.9 Per-layer position matters: put FP16 layers *later*, not earlier

The shape-6 exhaustive 16-pattern search (`F`=FP16 FlashAttention core, `P`=FP32 SDPA, per layer) found **12 of 16 patterns pass**, and the passing/failing split has a clean mechanistic explanation: **error scales with how many downstream FP32 layers a given FP16 layer's perturbation has to propagate through**, not simply with the count of FP16 layers. Among the four 3-FP16-layer patterns, `PFFF` (FP32 only in layer 1) passes at max-abs error 0.00176, while `FFFP` (FP32 only in the *last* layer) fails at 0.00222 — same FP16 layer count, opposite outcome, because an early-layer perturbation has more subsequent layers in which to compound. The fastest passing pattern, **`PFFF`** (layer 1 FP32 SDPA, layers 2–4 FP16 FlashAttention-2, compiled), reaches **3.587x** vs. baseline — a 31% improvement over the naive fallback (`L`, 2.74x) while still passing the exact same strict gate everywhere else does.

The same mechanism, applied to shape 14 (2 layers), resolves its failure entirely: **`LP`** (layer 1 FP32 SDPA, layer 2 FP16 FlashAttention-2) **agrees with `H` with zero failures** (0/2.46B elements), while `PL` and `LL` (pure `LFA`) both diverge. `LP` delivers **1.85x over `H`**, even uncompiled, at the true full length of 100,000 tokens.

### 4.10 Pushing already-passing shapes further: mixing LFA-level and VFA-level per layer

Since `LFA` already passes 12 shapes with comfortable margin (not just barely), and `VFA` is strictly faster than `LFA` *when it works* (it also lowers the QKV/out-projection GEMMs, not just attention), we generalized the per-layer search to a **3-way** per-layer menu (`none`=FP32 SDPA / `core`=LFA-level / `proj`=VFA-level) and brute-forced all `3⁴=81` patterns per shape, run in parallel across the idle GPU fleet (one shape, or a pair of shapes, per GPU). Result: **10 of the 12 searched shapes improve over plain `LFA`**, by margins from a marginal +0.7% (shape 2, already near its ceiling) up to **+44%** (shape 4) and **+38%** (shape 8 — previously the *hardest* shape for any optimization to help). Two shapes (7, 11) showed no improvement — plain `LFA` was already their optimum. Full numbers in §6.2.

## 5. Correctness methodology

Every "PASS" in this repository means: **zero failed elements** across 40 trials (10 each of `normal`/`tiny`/`large`/`outlier` input distributions), where a failed element is one where `abs(candidate − reference) > 0.002 AND abs(candidate − reference) > 0.02 × abs(reference)` (i.e. it fails *both* halves of the disclosed OR-tolerance, matching the competition's exact rule). No candidate is promoted into the final dispatch table unless it clears this bar on the specific shape it's dispatched for (§ATTRIBUTION 12).

Two important negative results are recorded, not hidden, because they materially inform any future work on this benchmark:

- **`NFA`/`N`/`O` (whole-model FP16)**: extremely fast (up to 27x), fails the accuracy gate on every shape tested. Kept in the harness as a diagnostic ceiling, never promoted.
- **`VFA`**: fails on every one of the 13 officially tested shapes as a *uniform* choice; only becomes viable when mixed per-layer with more conservative precision on the remaining layers (§4.10).

## 6. Results

### 6.1 Core validated single-configuration results (dtype=float32, `--no-allow-tf32`, stress accuracy profile, 10 trials/pattern)

| Shape | (B, S, D, H) | `H` | `L` (+compile) | `LFA` (FP16 FA2 core, all layers) |
|---|---|---|---|---|
| 1 | 64, 128, 128, 4 | 1.99x PASS | 2.75x PASS | 4.14x PASS |
| 2 | 1, 128, 128, 4 | 2.46x PASS | 13.84x PASS | 15.12x PASS |
| 3 | 4, 128, 128, 4 | 2.48x PASS | 11.50x PASS | 13.68x PASS |
| 4 | 16, 128, 128, 4 | 2.45x PASS | 5.70x PASS | 8.01x PASS |
| 5 | 128, 128, 128, 4 | 2.03x PASS | 2.83x PASS | 4.07x PASS |
| 6 | 10000, 128, 128, 4 | 2.04x PASS | 2.73x PASS | **FAIL** (4-5/6.55B elements) |
| 7 | 64, 128, 32, 4 | 2.39x PASS | 4.05x PASS | 10.73x PASS |
| 8 | 64, 128, 1024, 4 | 1.17x PASS | 1.20x PASS | 1.19x PASS |
| 9 | 64, 128, 128, 1 | 1.43x PASS | 2.15x PASS | 2.57x PASS |
| 10 | 64, 128, 128, 2 | 1.68x PASS | 2.39x PASS | 3.28x PASS |
| 11 | 64, 128, 128, 16 | 3.00x PASS | 3.76x PASS | 7.96x PASS |
| 12 | 64, 32, 128, 4 | 2.43x PASS | 5.43x PASS | 7.37x PASS |
| 13 | 64, 1024, 128, 4 | 3.95x PASS | 4.56x PASS | 14.34x PASS |
| 14 | 32, 100000, 1024, 16 | infeasible at full length* | infeasible (compile too slow to be worth it at this scale) | **FAIL** (33,910/9.83B elements) |

\*"Infeasible" for `A` (exact manual attention) — its own `S×S` score matrix needs 4.77 TB regardless of batch/chunk size. `H` and `L` *can* run (via SDPA, which never materializes the full score matrix) but require batch-microbatching (chunk=8 of 32) to fit in 48 GB; see §4.8.

### 6.2 Final per-shape dispatch (after per-layer hybrid search)

`F`/`L` = FP16 FlashAttention-2 attention core (LFA-level); `V` = FP16 QKV+attention+out-proj (VFA-level); `P` = FP32 SDPA. Letters read layer 1→N left to right.

| Shape | Winning plan | Speedup | vs. plain `LFA` |
|---|---|---|---|
| 1 | `LLVV` | 4.98x | +20% |
| 2 | `LLVV` | 15.23x | +0.7% |
| 3 | `LVLV` | 15.62x | +14% |
| 4 | `LLVV` | 11.54x | +44% |
| 5 | `LLLV` | 4.33x | +6% |
| 6 | `PFFF` (binary search; 3-way search **pending**, see callout below) | 3.59x | n/a — `LFA` fails here |
| 7 | plain `LFA` (no mix found an improvement) | 10.73x | — |
| 8 | `PVPV` | 1.64x | +38% |
| 9 | `LLVV` | 3.42x | +33% |
| 10 | `PLVV` | 3.63x | +11% |
| 11 | plain `LFA` (no mix found an improvement) | 7.96x | — |
| 12 | `PVLV` | 9.24x | +25% |
| 13 | `LLLV` | 15.50x | +8% |
| 14 | `LP` (layer 1 FP32 SDPA, layer 2 FP16 FA2, uncompiled, chunk=8 of 32) | 1.85x vs `H` | n/a — `LFA` fails here |

> **Pending update (shape 6):** the binary (`none`/LFA-level) search found `PFFF` at 3.587x. A follow-up 3-way search (adding the more aggressive VFA-level option per layer, the same generalization used for shapes 1–5/7–13) was launched across 81 patterns and was still running at the time of writing, given shape 6's batch size (10,000) makes each of the 81×40 accuracy-screening trials materially more expensive than at the other shapes. **This section will be updated with the final number once that job completes** — see `results/logs/search_boundary_shape6_3way.log` for the raw output when available.

Raw CSVs and logs backing every number above are in `results/csv/` and `results/logs/`; the machine-readable dispatch table is `results/dispatch/a6000_dispatch_final.json`.

## 7. Operational lessons (worth keeping for anyone extending this)

- **This is a shared server.** An early fix for a GPU-memory OOM (caching all 40 accuracy trials on CPU instead of GPU) accidentally introduced a 52 GB *host RAM* footprint, which is dangerous on a box that was independently sitting at ~66–130 GB available out of 1 TB with hundreds of GB already in swap from other users. The fix was to regenerate trial inputs on demand (accepting ~16x more baseline forward passes) instead of caching, keeping the process's RSS under ~1 GB. **Always profile host RAM, not just GPU VRAM, when parallelizing across a shared multi-GPU box.**
- **`torch.cuda.reset_peak_memory_stats(device)` (and similar) requires the CUDA context to already be initialized on that device** — call `torch.cuda.set_device(device)` (or otherwise touch the device) before any memory-stats call, or it raises `RuntimeError: Invalid device argument`.
- **Exact/manual attention cannot be used as a reference at extreme sequence lengths, at any batch size** — its own `O(S²)` memory requirement, not the candidate's, becomes the limiting factor. Use a memory-safe (SDPA/FlashAttention-based) variant that has already been validated on a feasible prefix as the reference instead.
- **A "top-K by heuristic" sort for benchmarking search survivors can silently exclude the true baseline from comparison**, producing false "beats the baseline" claims. Always include the known-best reference configuration explicitly in whatever gets benchmarked and compared, rather than trusting a proxy sort order to surface it.
- **`torch._dynamo`'s default recompilation cache limit (8) is easy to exhaust silently** in any script that compiles more than a handful of distinct model/shape combinations in one process; raise `torch._dynamo.config.cache_size_limit` explicitly and verify no `cache_size_limit` warnings appear in the log for a multi-shape sweep.

## 8. Limitations and future work

- Shape 6's 3-way (VFA-inclusive) per-layer search is not yet complete (§6.2 callout).
- The per-layer search space explored is still a fixed menu (none/LFA-level/VFA-level) rather than a fully independent per-layer choice of QKV precision, out-proj precision, and attention backend — a larger, more expensive search could plausibly find further gains, especially on shapes 7 and 11 where no improvement was found within the current menu.
- No custom CUDA/Triton kernels were written; all gains come from PyTorch-level operator choice (SDPA vs. FlashAttention-2), precision boundaries, layout (packed QKV), and `torch.compile`. §17–18 of `ATTRIBUTION.md` outline concrete next steps (LayerNorm+QKV fusion, FFN fusion, persistent/whole-model kernels, tile/warp/pipeline tuning) that were identified but not implemented in the time available.
- Shape 14's dispatch entry (`LP`, uncompiled) has not been benchmarked with `torch.compile`, given the extreme per-forward cost at that shape (~60s/forward for `H` alone) made compiling multiple candidates impractical within the available time; a compiled `LP` would very likely be faster still.
- The 14-shape table in this repo should be cross-checked against the official Feishu appendix before final submission, per the problem statement's own note that content may have diverged.
- Everything here targets a single GPU model (A6000, Ampere). The attribution document (§19–20) is explicit about which paper-derived directions (Hopper FA3 scheduling, Blackwell FP4 attention) do **not** transfer to this hardware and would need separate validation on different GPUs.
