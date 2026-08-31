#!/usr/bin/env python3
"""Per-layer 3-way precision boundary search: none / LFA-level / VFA-level.

For each of a set of shapes, brute-forces every 3^num_layers combination of:
  none = FP32 SDPA (H/L-style)
  core = attention_core: FP32 QKV/out-proj, FP16 FlashAttention-2 core (LFA)
  proj = attention_projection: QKV + attention + out-proj all FP16 via FA2 (VFA)

LFA (all layers "core") already passes with comfortable margin on the shapes
this script targets, and full VFA (all layers "proj") fails on every official
shape. This asks the more precise question: does mixing in one or two "proj"
layers, in the right position, let some shapes beat pure LFA while still
passing -- the same trick that recovered speed on shape 6 (there, mixing
"none" and "core"; here, mixing "core" and "proj").

Usage: python search_layer_precision_boundary.py <shape_id>[,<shape_id>...] <cuda_device_index>
"""
import copy as copy_module
import itertools
import sys

sys.path.insert(0, "/local1/stsj/misc")

import torch

from transformer_ablation_benchmark import (
    AccuracySummary,
    BaselineTransformer,
    SelectivePrecisionTransformer,
    competition_config,
    copy_baseline_weights_selective,
    generate_random_case,
    maybe_compile,
    update_accuracy_summary,
    warmup_model,
    benchmark_once,
)

torch.manual_seed(1234)
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

import torch._dynamo as torch_dynamo
torch_dynamo.config.cache_size_limit = 256

shape_ids = [int(s) for s in sys.argv[1].split(",")]
device_index = int(sys.argv[2])
device = torch.device(f"cuda:{device_index}")
torch.cuda.set_device(device)

fast_dtype = torch.float16
RTOL, ATOL = 0.02, 0.002
ACCURACY_TRIALS = 10
SEED = 1234
LEVELS = ("none", "core", "proj")
LEVEL_LABEL = {"none": "P", "core": "L", "proj": "V"}  # P=FP32 SDPA, L=LFA-level, V=VFA-level
LEVEL_TO_PRECISION_MODE = {
    "none": "none",
    "core": "attention_core",
    "proj": "attention_projection",
}


def build_candidate(config, baseline_shadow, layer_modes):
    resolved_modes = [LEVEL_TO_PRECISION_MODE[m] for m in layer_modes]
    m = SelectivePrecisionTransformer(
        config, fast_dtype=fast_dtype, precision_mode="attention_core",
        attention_impl="flash_attn", sdpa_backend="auto",
        layer_precision_modes=resolved_modes,
    )
    copy_baseline_weights_selective(baseline_shadow, m)
    m.set_assume_all_valid_mask(True)
    m = m.to(device=device, dtype=torch.float32).eval()
    m.configure_selected_modules()
    return m


def check_accuracy(baseline, candidate, config):
    summary = AccuracySummary()
    patterns = ("normal", "tiny", "large", "outlier")
    with torch.inference_mode():
        for p_idx, pattern in enumerate(patterns):
            for trial in range(ACCURACY_TRIALS):
                seed = SEED + p_idx * 100_000 + trial
                x, mask = generate_random_case(
                    config=config, device=device, dtype=torch.float32,
                    seed=seed, padding_ratio=0.0, input_scale=1.0, pattern=pattern,
                )
                try:
                    ref = baseline(x, mask)
                    out = candidate(x, mask)
                    update_accuracy_summary(summary, ref, out, rtol=RTOL, atol=ATOL)
                    del ref, out
                except Exception as exc:
                    summary.passed = False
                    summary.error = f"{type(exc).__name__}: {exc}"
                    del x, mask
                    return summary
                del x, mask
    return summary


for shape_id in shape_ids:
    config = competition_config(shape_id)
    print(f"\n{'='*80}\n=== shape {shape_id}: {config} ===")

    baseline = BaselineTransformer(config).to(device=device, dtype=torch.float32).eval()
    baseline_shadow = BaselineTransformer(config)
    baseline_shadow.load_state_dict(
        copy_module.deepcopy({k: v.detach().cpu() for k, v in baseline.state_dict().items()})
    )

    all_patterns = list(itertools.product(LEVELS, repeat=config.num_layers))
    print(f"{len(all_patterns)} per-layer patterns to screen ({len(LEVELS)}^{config.num_layers})")

    passing = []
    for pattern in all_patterns:
        label = "".join(LEVEL_LABEL[p] for p in pattern)
        candidate = build_candidate(config, baseline_shadow, list(pattern))
        summary = check_accuracy(baseline, candidate, config)
        status = "ERROR" if summary.error else ("PASS" if summary.passed else "FAIL")
        n_proj = sum(1 for p in pattern if p == "proj")
        n_core = sum(1 for p in pattern if p == "core")
        if status == "PASS":
            passing.append((pattern, n_proj, n_core))
            print(f"  {label}  PASS  max_abs={summary.max_abs_error:.6g} n_proj={n_proj} n_core={n_core}")
        del candidate
    torch.cuda.empty_cache()

    print(f"{len(passing)}/{len(all_patterns)} patterns PASS for shape {shape_id}")

    # Prefer more "proj" (fastest), then more "core", i.e. as aggressive as possible.
    passing.sort(key=lambda t: (-t[1], -t[2]))

    if not passing:
        print(f"shape {shape_id}: NO passing pattern found (unexpected -- even all-none should pass)")
        continue

    pure_lfa = tuple(["core"] * config.num_layers)
    best_beats_lfa = passing[0][0] != pure_lfa

    print(f"\n=== Benchmarking top passing patterns for shape {shape_id} (compiled) ===")
    x_bench, mask_bench = generate_random_case(
        config=config, device=device, dtype=torch.float32,
        seed=SEED + 100000, padding_ratio=0.0, input_scale=1.0, pattern="normal",
    )
    warmup_model(baseline, x_bench, mask_bench, 10, device)
    a_samples = benchmark_once(baseline, x_bench, mask_bench, 30, device)
    a_median = sorted(a_samples)[len(a_samples) // 2]

    bench_results = []
    # Only benchmark the most-aggressive handful to keep compile time bounded.
    for pattern, n_proj, n_core in passing[:5]:
        label = "".join(LEVEL_LABEL[p] for p in pattern)
        candidate = build_candidate(config, baseline_shadow, list(pattern))
        try:
            compiled = maybe_compile(candidate, enabled=True, mode="reduce-overhead", fullgraph=False)
            warmup_model(compiled, x_bench, mask_bench, 10, device)
            samples = []
            for _ in range(3):
                samples.extend(benchmark_once(compiled, x_bench, mask_bench, 30, device))
            median_ms = sorted(samples)[len(samples) // 2]
            speedup = a_median / median_ms
            print(f"  {label}  median={median_ms:.4f} ms  speedup={speedup:.3f}x")
            bench_results.append((label, median_ms, speedup))
        except Exception as exc:
            print(f"  {label}  COMPILE/BENCH ERROR: {type(exc).__name__}: {exc}")
        del candidate
        torch.cuda.empty_cache()

    if bench_results:
        best = max(bench_results, key=lambda t: t[2])
        print(f"\nshape {shape_id} BEST: {best[0]}  {best[2]:.3f}x vs A"
              + ("  <-- BEATS pure LFA" if best[0] != "".join(["L"] * config.num_layers) else "  (== pure LFA, nothing better found)"))
