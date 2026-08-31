# Evidence-Grade Contribution Lineage

## Exact attribution to the papers supplied for this project

This document ties every optimization in this repository to a **specific supplied paper**, a page/section in that paper, and a statement of exactly what that paper supports. It replaces any looser "idea lineage" narrative.

A crucial distinction is used throughout:

- **Direct source** — the paper explicitly describes essentially the same optimization.
- **Motivation/adaptation** — the paper supplies the systems principle, but our implementation is adapted to this benchmark.
- **Our synthesis** — the exact mechanism was derived from our benchmark results by combining ideas from multiple supplied papers; it should not be presented as copied from a single paper.

---

## 1. Packed QKV projection

### Our change

Baseline:

```text
Q = X Wq
K = X Wk
V = X Wv
```

Optimized:

```text
[Q K V] = X [Wq Wk Wv]
```

implemented as one packed `Linear(D, 3D)` with baseline Q/K/V weights concatenated.

### Exact source

**`2007.00072v3.pdf`** — *Data Movement Is All You Need: A Case Study on Optimizing Transformers*, Andrei Ivanov et al., MLSys 2021.

**Appendix A.5.2 — Algebraic Fusion, page 19.** The paper explicitly enumerates three ways to compute the self-attention input projections. Its third option stacks all three matrices:

```text
All three can be stacked:
[Q K V] = [WQ WK WV] X
```

and explains the tradeoff: stacking improves reuse because `X` is used once.

### Attribution strength

**DIRECT SOURCE.** This is the cleanest citation for our packed-QKV optimization. FlashAttention is not the primary source for packed QKV; this paper describes the algebraic fusion directly.

---

## 2. Reducing Transformer data movement as the main optimization objective

### Our change

The project prioritizes removing intermediate tensors, removing unnecessary layout/mask traffic, packing projections, using fused attention, and considering LayerNorm→matmul and FFN fusion.

### Exact source

**`2007.00072v3.pdf`** — *Data Movement Is All You Need*, Abstract and Sections 3–4 (Section 4 "Fusion", page 1 onward).

The paper states Transformer execution has become memory/data-movement constrained and proposes: (1) identifying data-movement reduction opportunities, (2) exploring layouts, (3) fusing compatible operations, (4) choosing configurations based on measured end-to-end performance. Section 4 states fusion is valuable when it reduces kernel launches and reduces loads/stores between operators.

### Attribution strength

**DIRECT SYSTEMS MOTIVATION** for the project-wide "optimize movement, not only FLOPs" philosophy.

---

## 3. Fused exact attention / SDPA path

### Our change

Baseline explicit attention (`QKᵀ → scale → mask → softmax → P@V`) was replaced by PyTorch SDPA and, where legal, FlashAttention-2.

### Exact source

**`2205.14135v2.pdf`** — *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, Tri Dao et al., NeurIPS 2022.

Page 1 abstract; pages 2 and 4–5, Section 3.1. The paper argues attention must be IO-aware and uses tiling to avoid repeatedly reading/writing the large `N × N` attention matrix between HBM and SRAM, fusing matmul → softmax → optional mask/dropout → matmul into one kernel.

### Attribution strength

**DIRECT SOURCE** for the fused-attention principle. `torch.nn.functional.scaled_dot_product_attention` is the PyTorch implementation route for this idea; the exact API is not introduced by the paper.

---

## 4. Long-sequence memory-safe attention

### Our change

For shape 14 (`S = 100000`), the benchmark never materializes a full `S × S` attention matrix in the full-size performance path; it uses memory-efficient attention kernels and, where feasible, validates the exact baseline only on a manageable prefix.

### Exact source

**`2205.14135v2.pdf`** — FlashAttention, pages 1–5, especially Figure 1 and Section 3.1. The paper's central mechanism is avoiding materialization of the full attention matrix in HBM.

### Attribution strength

**DIRECT SOURCE** for the memory-safe attention mechanism. The **prefix-validation harness** and the **batch-microbatching fallback** used when even the memory-safe path exceeds a single GPU's memory (see §13/§Repo Section on shape 14) are our own benchmark engineering, not a FlashAttention contribution.

---

## 5. Better attention work partitioning and shape sensitivity

### Our change / design implication

We do not assume one attention kernel configuration is optimal for all `(batch, heads, seq_len, head_dim)` regimes, and instead tune/dispatch per shape.

### Exact source

**`2307.08691v1.pdf`** — *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*, Tri Dao, 2023.

Pages 8–9, Sections 3.2–3.3. Original FlashAttention parallelized mainly over batch and heads; for long sequences / small batch / small head counts, FlashAttention-2 additionally parallelizes over the sequence dimension. Block sizes such as `{64,128} × {64,128}` are manually tuned based on head dimension and shared-memory constraints.

### Attribution strength

**DIRECT SOURCE** for attention work-partitioning and tile/shape sensitivity; motivates our shape-specific dispatch (§11–§13).

---

## 6. Removing useless padding/mask work

### Our change

When the configuration guarantees every token is valid, the optimized path avoids constructing and applying a validity mask and lets causal attention use the specialized causal route directly.

### Exact source

**`2210.03052v4.pdf`** — *ByteTransformer: A High-Performance Transformer Boosted for Variable-Length Inputs*, Yujia Zhai et al.

Page 1 abstract/introduction: ByteTransformer identifies padding as redundant computation/memory overhead and proposes a padding-free pipeline that removes calculations on zero-padded tokens.

### Attribution strength

**MOTIVATION / ADAPTATION, not an identical implementation.** ByteTransformer removes padding for variable-length sequences using packed representations and offsets; our benchmark has a simpler special case (`known all-valid mask → remove the no-op mask entirely`). Framing used in this repo: *"Inspired by ByteTransformer's principle of eliminating redundant padded-token work, we specialize the disclosed all-valid workload by removing the no-op validity mask."* We do **not** claim our `mask=None` mechanism is ByteTransformer's algorithm.

---

## 7. Stress-testing low precision with outliers

### Our change

The accuracy harness expanded from ordinary random inputs to four input patterns — `normal`, `tiny`, `large`, `outlier` — with repeated trials. The outlier profile injects rare, much larger values.

### Exact source

**`2407.08608v2.pdf`** — *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision*, Jay Shah et al., 2024.

Page 11, numerical-error experiment around Table 3: the paper tests a distribution of the form `N(0,1) + N(0,100)·Bernoulli(0.001)` (≈0.1% of entries receive a large outlier term), comparing numerical error of baseline FP16, FlashAttention-2 FP16, and FlashAttention-3 FP16/FP8.

### Attribution strength

**DIRECT SOURCE** for the rare-outlier stress profile. The additional `tiny` and `large` profiles are our own extensions.

---

## 8. Keeping numerically sensitive work at higher precision

### Our change

Selective-precision variants (`U`, `X`, `Y`, `V`, `W` in the harness) keep some regions in FP32 while moving others to FP16/BF16 — e.g. `U`: FP32 QKV, FP16 attention core, FP32 output projection/residual/FFN.

### Exact sources

- **`2407.08608v2.pdf`** — FlashAttention-3, page 1 and numerical-accuracy discussion (~page 11): FP16 FlashAttention keeps intermediate quantities such as softmax rescaling in FP32.
- **`2505.11594v3.pdf`** — *SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-bit Training*, page 2: identifies the most accuracy-sensitive matmul in its low-bit training path and retains it at higher precision while quantizing other operations more aggressively.

### Attribution strength

**MOTIVATION / ADAPTATION.** Neither paper implements our exact "FP32 Transformer state + FP16 attention-only boundary." The paper-derived principle is *lower precision selectively; identify sensitive operations and preserve their precision*. Our exact U/X/Y/V/W boundary search is our own experimental realization of that principle.

---

## 9. LFA: FP32 Transformer + FP16 FlashAttention-2 core

### Our change

```text
FP32 LayerNorm → FP32 packed QKV → cast Q/K/V → FP16 FlashAttention-2
  → cast attention result → FP32 output projection → FP32 residual → FP32 FFN
  + torch.compile around the graph
```

This is the strongest single-configuration path found on the A6000, passing 12 of 13 tested shapes (and, in per-shape hybrid form, all remaining shapes; see §13).

### Exact sources

- **`2307.08691v1.pdf`** — FlashAttention-2: direct source for the FA2 attention kernel family.
- **`2407.08608v2.pdf`** — FlashAttention-3: evidence that low-precision attention retains strong numerical properties when sensitive intermediate state is handled carefully.
- **`2505.11594v3.pdf`** — SageAttention3: supports selective preservation of precision for accuracy-sensitive operations rather than making every operation low precision.

### Attribution strength

**OUR SYNTHESIS.** No supplied paper says "keep the entire Transformer FP32 and cast only Q/K/V into FP16 FA2." That design came from our measured sequence: whole-model FP16 (`NFA`) → fast but fails the accuracy gate everywhere; attention-only FP16 (`U`, SDPA-based) → much safer; FlashAttention-2 → faster still on the A6000; therefore combine attention-only precision with FA2 + compile. Report wording: *"Our hybrid precision boundary, motivated by FlashAttention-2/3 and selective low-precision observations, rather than a direct reproduction of one paper."*

---

## 10. `torch.compile` path (`L` = `H` + compile)

### Our change

`L = H + torch.compile`, plus a fix to `torch._dynamo`'s recompilation cache-size limit so a full multi-shape sweep stays genuinely compiled instead of silently falling back to eager execution after the default limit of 8 distinct graphs (see `--dynamo-cache-size-limit` in `src/transformer_ablation_benchmark.py`).

### Related sources

- **`2505.22758v2.pdf`** — *FlashFormer: Whole-Model Kernels for Efficient Low-Batch Inference*, pages 1–2: low-batch inference is dominated by memory traffic and kernel-launch overhead; argues for more fusion/specialization.
- **`2512.22219v2.pdf`** — Mirage Persistent Kernel (MPK): contrasts kernel-per-operator execution with persistent/mega-kernel execution, emphasizing elimination of repeated launch overhead and cross-operator materialization.
- **`2505.07829v1.pdf`** — *Blockbuster, Part 1: Block-level AI Operator Fusion*: operator fusion reduces intermediate memory traffic and kernel invocation overhead.

### Attribution strength

**PAPER-MOTIVATED ENGINEERING CHOICE.** `torch.compile` itself is a PyTorch technology, not a contribution of these papers. The Dynamo cache-size/recompile fix is entirely our engineering work. Report wording: *"FlashFormer/Blockbuster/MPK motivated reducing launch and cross-operator overhead; we used PyTorch `torch.compile` as the practical implementation mechanism available in the benchmark."*

---

## 11. Shape-specific dispatch

### Our change

Moving from one universal implementation to: `known Transformer configuration → precomputed fastest correctness-passing route` (e.g. most shapes → LFA; shape 6 → a validated per-layer hybrid; shape 14 → a dedicated long-sequence/microbatched path).

### Exact source

**`2607.17979v1.pdf`** — *Harness Engineering for LLM-Driven GPU Kernel Generation*, Yue Shui et al., 2026.

- Page 2, "Workload-Grounded Shape Dispatch": optimization regimes are derived from workload axes and measured latency distributions; shape-specialized routes are introduced when profiling/latency evidence shows different limiting factors.
- Pages 7–8, GDN Decode: an explicit dispatch example across batch regimes (Triton recurrent route for small batches, one-warp specialization for batch eight, CuTe pretranspose for mid-sized batches, a larger-value route for the highest batch regime) — the paper states directly that a single kernel did not generalize across batch regimes.

### Attribution strength

**DIRECT SOURCE.** The best-supported citation for our shape-conditioned dispatcher.

> Concrete implementation: `src/optimized_transformer.py::choose_execution_plan`, `_EXACT_DISPATCH`. See `TECH_REPORT.md` §9.

---

## 12. Correctness-gated promotion of shape routes

### Our change

A candidate is retained only if it compiles, passes correctness, survives stress testing, *and* wins for the relevant shape. Incorrect-but-fast paths (e.g. whole-FP16 `NFA`) remain diagnostic only and are never promoted into the dispatch table.

### Exact source

**`2607.17979v1.pdf`** — Harness Engineering, pages 2 and 8: correctness and latency are both mandatory; workload-grounded route promotion; profiler-based decisions; rejecting local wins that don't survive the full workload distribution; archiving accepted and rejected probes alike.

### Attribution strength

**DIRECT SOURCE** for the methodology. The exact numerical gate (`abs_error ≤ 0.002 OR relative_error ≤ 2%`) comes from the competition specification, not this paper.

---

## 13. Hybrid per-layer FP16-FA2 / FP32-SDPA (and FP16-QKV+out-proj) search

### Our change

For shape 6 (batch=10,000, 4 layers), we brute-forced all `2^4 = 16` per-layer plans (`F` = FP16 FlashAttention-2 core, `P` = FP32 compiled SDPA), then extended to a 3-way per-layer menu (`none` / LFA-level / VFA-level) across the remaining shapes, and a 2-layer binary search for shape 14. Only zero-failure plans are benchmarked; the fastest passing plan is stored in the shape dispatcher. See `src/search_shape6_hybrid.py`, `src/search_layer_precision_boundary.py`, `src/shape14_hybrid.py`, and the final per-shape table in `TECH_REPORT.md`.

### Paper lineage

- **`2607.17979v1.pdf`** — Harness Engineering: directly supports workload/shape-conditioned route selection.
- **`2505.11594v3.pdf`** — SageAttention3: directly supports the idea that some operations are more accuracy-sensitive and should retain higher precision.
- The specific finding that **error compounds with the number of downstream FP32 layers following a low-precision layer** (i.e., placing FP16 layers later in the stack is safer than placing them earlier) is **our own benchmark evidence**, discovered from the shape-6 and shape-14 search results.

### Attribution strength

**OUR SYNTHESIS — NOT A DIRECT PAPER CONTRIBUTION.** Report wording: *"Inspired by workload-grounded route dispatch in Harness Engineering and selective preservation of accuracy-sensitive computation in SageAttention3, we introduce a per-layer precision/backend search. The exact per-layer FP32-SDPA/FP16-FA2(/FP16-projection) bitmask search, and the finding that FP16 layer position determines safety, are our contribution."*

> The "FP16 layers later = safer" finding is also what the unseen-shape generalization rule in `src/optimized_transformer.py::_classify_regime` is built on (put the first ~25% of layers at FP32, later layers at FP16) — see `TECH_REPORT.md` §9.1 for why that specific ratio was chosen.

---

## 14. Shape-specific fallback is independently reinforced by KernelEvolve

### Our design

Use an optimized specialized kernel only in the shape region where it wins; fall back elsewhere (e.g. `L` on shape 6, plain `H`/`L`/`LP`-hybrid on shape 14).

### Exact source

**`2512.23236v4.pdf`** — *KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta*, Section 5.3/page 30: deployment strategy where optimized kernels are used on production shapes where they win, while larger/out-of-distribution configurations fall back to the unfused PyTorch baseline ("shape-specific dispatch"). Its PFFN case also demonstrates shape-aware tiling adapting to SRAM capacity and dimensions.

### Attribution strength

**DIRECT SUPPORTING SOURCE.** Harness Engineering is the primary citation for our dispatcher; KernelEvolve is a second, independent supporting source.

---

## 15. Future optimization: LayerNorm → packed-QKV fusion

### Proposed change (not yet implemented)

Fuse LayerNorm with the following packed-QKV matmul so the normalized intermediate does not need to round-trip through HBM.

### Exact source

**`2505.07829v1.pdf`** — Blockbuster, Part 1, page 1 abstract, page 4, Examples ~pages 21–22: presents "Flash-LayerNorm+Matmul", fusing LayerNorm with matrix multiplication.

### Attribution strength

**DIRECT SOURCE** for LayerNorm+Matmul fusion; our adaptation specifically targets LayerNorm + packed-QKV projection, since QKV is the matmul immediately following `norm1` in this Transformer.

---

## 16. Future optimization: FFN fusion

### Proposed change (not yet implemented)

Fuse FFN operations (matmul → bias → GELU → matmul) to avoid intermediate round-trips.

### Exact sources

- **`2512.23236v4.pdf`** — KernelEvolve, Section 5.3.2, pages 30–32: the InterFormer PFFN case fuses matmul, bias, GELU, and RMSNorm while tiles remain in SRAM.
- **`2602.11808v1.pdf`** — *Deep Kernel Fusion for Transformers*: proposes a fused Transformer MLP/SwiGLU kernel to reduce HBM traffic and improve cache reuse.
- **`2505.07829v1.pdf`** — Blockbuster: presents a fused RMSNorm + FFN-SwiGLU mega-kernel.

### Attribution strength

**DIRECT SOURCE** for the general FFN-fusion direction. Our baseline uses GELU rather than SwiGLU, so KernelEvolve's PFFN example is the closest match.

---

## 17. Future optimization: persistent / whole-model kernels

### Proposed direction (not yet implemented)

If PyTorch-level and local fusion plateaus, explore larger fused or persistent execution regions rather than continuing to add framework-level flags.

### Exact sources

- **`2505.22758v2.pdf`** — FlashFormer: fuses the entire Transformer forward pass into one specialized kernel, motivated by memory bandwidth and launch overhead in low-batch inference.
- **`2512.22219v2.pdf`** — Mirage Persistent Kernel (MPK): argues kernel-per-operator execution causes repeated launch overhead and prevents cross-operator software pipelining; proposes persistent mega-kernel execution.

### Attribution strength

**DIRECT SOURCE** for this future direction. Not implemented in this repo — labeled future work, not an achieved contribution.

---

## 18. Future custom-kernel tuning: tiles, pipelines, warp/grid scheduling

### Proposed direction (not yet implemented)

After selecting the correct graph/backend per shape, tune tile sizes, warp counts, pipeline stages, grid order, and persistent scheduling.

### Exact sources

- **`2410.20399v1.pdf`** — *ThunderKittens: Simple, Fast, and Adorable AI Kernels*: structures GPU optimization at warp/tile, async-pipeline, and grid levels.
- **`osdi25-cheng.pdf`** — *PipeThreader: Software-Defined Pipelining for Efficient DNN Execution*: exposes fine-grained tasks and specialized hardware units, searches scheduling/pipelining strategies, and notes different tensor shapes can require different handcrafted kernels.
- **`2504.17577v2.pdf`** — *TileLang: A Composable Tiled Programming Model for AI Systems*: exposes tiled dataflow/scheduling controls; fixed tile choices can be suboptimal across sequence lengths, and identifies dynamic-shape tile selection as an important tuning direction.

### Attribution strength

**DIRECT SOURCE** for the tuning dimensions. Specific tile numbers would be our measured implementation choices if pursued.

---

## 19. Blackwell / RTX 5090 low-bit route (future hardware target only)

### Proposed direction only — not applicable to this A6000 submission

**`2505.11594v3.pdf`** — SageAttention3 targets Blackwell FP4 Tensor Cores (reports 1038 TOPS on RTX 5090) using microscaling/two-level quantization. **Not suitable as justification for the A6000 path**, since Ampere has no Blackwell FP4 Tensor Cores.

---

## 20. FlashAttention-3 and FlashAttention-4 should not be credited to the A6000 implementation

- **`2407.08608v2.pdf`** — FlashAttention-3 is primarily a Hopper/H100 design (TMA, WGMMA, warp specialization, GEMM-softmax overlap, FP8). We used it for numerical-stress methodology, mixed-precision insight, and future scheduling ideas — our A6000 cannot execute the Hopper-specific FA3 design.
- **`2603.05451v1.pdf`** — FlashAttention-4 is explicitly a Blackwell/B200 design. It is a future target-GPU comparison, not a source for the current A6000 implementation.

---

## 21. Exact final attribution table

| Our contribution/change | Exact supplied paper | Relationship |
|---|---|---|
| Packed QKV | `2007.00072v3.pdf` — Data Movement Is All You Need, Appendix A.5.2 | **Direct** |
| Data-movement-first optimization | `2007.00072v3.pdf` | **Direct principle** |
| Fused/memory-efficient attention | `2205.14135v2.pdf` — FlashAttention | **Direct** |
| Long-sequence memory-safe attention | `2205.14135v2.pdf` — FlashAttention | **Direct** |
| Better attention work partitioning | `2307.08691v1.pdf` — FlashAttention-2 | **Direct** |
| No-op padding/mask elimination | `2210.03052v4.pdf` — ByteTransformer | **Adaptation** |
| Outlier numerical stress testing | `2407.08608v2.pdf` — FlashAttention-3 | **Direct** |
| Selectively preserve sensitive precision | `2407.08608v2.pdf` + `2505.11594v3.pdf` | **Adaptation** |
| U/X/Y/V/W precision-boundary search | FA3 + SageAttention3 principles | **Our implementation** |
| `L` / compiled execution | FlashFormer + Blockbuster + MPK (motivate fusion/launch reduction) | **Our implementation mechanism** |
| `LFA` | FA2 + FA3 + SageAttention3 | **Our synthesis** |
| Shape-conditioned dispatcher | `2607.17979v1.pdf` — Harness Engineering | **Direct** |
| Shape-specific safe fallback | `2512.23236v4.pdf` — KernelEvolve | **Direct supporting evidence** |
| Per-layer precision/backend hybrid search | Harness Engineering + SageAttention3 + our measurements | **Our synthesis / new contribution** |
| LayerNorm→QKV fusion | `2505.07829v1.pdf` — Blockbuster | **Direct adaptation (future work)** |
| FFN fusion | `2512.23236v4.pdf` KernelEvolve; `2602.11808v1.pdf` Deep Kernel Fusion | **Direct direction (future work)** |
| Persistent/whole-model kernel | `2505.22758v2.pdf` FlashFormer; `2512.22219v2.pdf` MPK | **Future direction** |
| Tile/warp/pipeline search | `2410.20399v1.pdf` ThunderKittens; `osdi25-cheng.pdf` PipeThreader; `2504.17577v2.pdf` TileLang | **Future tuning direction** |
| Blackwell FP4 attention | `2505.11594v3.pdf` SageAttention3 | **Future hardware-specific direction** |
| Hopper FA3 scheduling | `2407.08608v2.pdf` FlashAttention-3 | **Future hardware-specific direction** |
| B200 FA4 comparison | `2603.05451v1.pdf` FlashAttention-4 | **Future hardware-specific direction** |

---

## 22. Recommended wording for the written project description

> We did not derive the final implementation from a single kernel paper. We first adopted algebraic QKV stacking from *Data Movement Is All You Need* and IO-aware fused attention from *FlashAttention*/*FlashAttention-2*. FlashAttention-3 and SageAttention3 motivated our accuracy-aware precision experiments. After measurements showed that low-precision attention was fast but shape-dependent under the strict checker, we adopted workload-grounded dispatch following the methodology demonstrated in *Harness Engineering for LLM-Driven GPU Kernel Generation*. Our final per-layer FP32-SDPA/FP16-FlashAttention search is a new synthesis: it combines shape-aware dispatch with operation-sensitive precision selection and chooses the fastest zero-failure plan for each disclosed workload.

This separates **paper-derived ideas** from **our implementation** from **our new experimental synthesis** — the safest and strongest attribution structure for this submission.

> **Note on primary-source PDFs:** the papers cited above (arXiv IDs `2007.00072`, `2205.14135`, `2307.08691`, `2210.03052`, `2407.08608`, `2505.11594`, `2607.17979`, `2512.23236`, `2505.07829`, `2602.11808`, `2505.22758`, `2512.22219`, `2410.20399`, `2504.17577`, `2603.05451`, and `osdi25-cheng.pdf`) were supplied as project reference material outside this repository and are not redistributed here; consult arXiv/the original venue for the source text when preparing the written project description.
