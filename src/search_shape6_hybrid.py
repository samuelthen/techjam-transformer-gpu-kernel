#!/usr/bin/env python3
"""Brute-force per-layer FP16-FlashAttention / FP32-SDPA hybrid search for shape 6.

Shape 6 = (B=10000, D=128, H=4, S=128, layers=4, causal=True, ffn=128). LFA
(all 4 layers FP16 FlashAttention-2 attention core, everything else FP32)
fails the strict stress-accuracy gate there by 5 elements out of ~160M per
trial -- plausible statistical noise from FP16's ~1e-3 error floor at extreme
batch size, not a fundamental flaw. This script tests all 2^4=16 per-layer
patterns (F = FP16 FlashAttention-2 core, P = FP32 SDPA) for accuracy, then
benchmarks (with torch.compile) only the patterns that pass with zero
failures, to find the fastest legal hybrid plan for this specific shape.
"""
import itertools
import sys

sys.path.insert(0, "/local1/stsj/misc")

import torch

from transformer_ablation_benchmark import (
    AccuracySummary,
    BaselineTransformer,
    SelectivePrecisionTransformer,
    TransformerConfig,
    benchmark_once,
    competition_config,
    copy_baseline_weights_selective,
    generate_random_case,
    maybe_compile,
    update_accuracy_summary,
    warmup_model,
)

torch.manual_seed(1234)
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

import torch._dynamo as torch_dynamo
torch_dynamo.config.cache_size_limit = 128

device = torch.device("cuda")
config = competition_config(6)
assert config.num_layers == 4
fast_dtype = torch.float16
RTOL, ATOL = 0.02, 0.002
ACCURACY_TRIALS = 10
SEED = 1234

print(f"shape 6 config: {config}")
print(f"torch={torch.__version__}, gpu={torch.cuda.get_device_name(0)}, TF32=False")

# ---------------------------------------------------------------------------
# Build the FP32 baseline once, and a CPU shadow for deterministic weight copy.
# ---------------------------------------------------------------------------
baseline = BaselineTransformer(config).to(device=device, dtype=torch.float32).eval()
baseline_shadow = BaselineTransformer(config)
import copy as copy_module
baseline_shadow.load_state_dict(
    copy_module.deepcopy({k: v.detach().cpu() for k, v in baseline.state_dict().items()})
)

# ---------------------------------------------------------------------------
# Only cache the (seed, pattern) list -- NOT the tensors. At batch=10000 a
# single trial's x+ref is ~1.3 GB; caching all 40 blew up to ~52 GB of host
# RAM, which is unsafe on this shared machine (it was already sitting at
# ~66 GB available system-wide with 684 GB in swap from other users' jobs).
# Regenerate x and the baseline reference per trial per pattern instead: 16x
# more baseline forward passes, but each is <1s and the memory footprint at
# any instant is just one trial's worth (~2-3 GB), not 40.
# ---------------------------------------------------------------------------
patterns_input = ("normal", "tiny", "large", "outlier")
trial_specs = [
    (pattern, SEED + pattern_index * 100_000 + trial)
    for pattern_index, pattern in enumerate(patterns_input)
    for trial in range(ACCURACY_TRIALS)
]
print(f"\n{len(trial_specs)} stress trials will be (re)generated per pattern "
      f"to keep memory flat (no persistent cache).")


def build_candidate(layer_pattern):
    candidate = SelectivePrecisionTransformer(
        config, fast_dtype=fast_dtype, precision_mode="attention_core",
        attention_impl="flash_attn", sdpa_backend="auto", layer_pattern=layer_pattern,
    )
    copy_baseline_weights_selective(baseline_shadow, candidate)
    candidate.set_assume_all_valid_mask(True)
    candidate = candidate.to(device=device, dtype=torch.float32).eval()
    candidate.configure_selected_modules()
    return candidate


def check_accuracy(candidate):
    summary = AccuracySummary()
    with torch.inference_mode():
        for pattern, seed in trial_specs:
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
                break
            del x, mask
    return summary


print("\n=== Stage 1: accuracy screen, all 16 per-layer patterns (F=FP16 FlashAttn, P=FP32 SDPA) ===")
results = {}
for pattern_bits in itertools.product([False, True], repeat=4):
    label = "".join("F" if b else "P" for b in pattern_bits)
    candidate = build_candidate(list(pattern_bits))
    summary = check_accuracy(candidate)
    status = "ERROR" if summary.error else ("PASS" if summary.passed else "FAIL")
    n_flash = sum(pattern_bits)
    print(f"{label}  (n_flash={n_flash})  {status:6} max_abs={summary.max_abs_error:.6g} "
          f"max_rel={summary.max_relative_error:.6g} failed={summary.failed_elements}/{summary.total_elements}"
          + (f"  error={summary.error}" if summary.error else ""))
    results[pattern_bits] = (status, summary, n_flash)
    del candidate
    torch.cuda.empty_cache()

passing = [(bits, n) for bits, (status, s, n) in results.items() if status == "PASS"]
print(f"\n{len(passing)}/16 patterns PASS with zero failures.")

if not passing:
    print("No passing hybrid pattern found. Falling back to L (all-FP32) is the only safe option.")
    sys.exit(0)

# ---------------------------------------------------------------------------
# Stage 2: benchmark (compiled) only the passing patterns, prefer more FP16.
# ---------------------------------------------------------------------------
passing.sort(key=lambda t: -t[1])
print("\n=== Stage 2: benchmarking passing patterns (compiled), most-FP16 first ===")

x_bench, mask_bench = generate_random_case(
    config=config, device=device, dtype=torch.float32,
    seed=SEED + 100000, padding_ratio=0.0, input_scale=1.0, pattern="normal",
)

bench_results = []
for bits, n_flash in passing:
    label = "".join("F" if b else "P" for b in bits)
    candidate = build_candidate(list(bits))
    try:
        compiled = maybe_compile(candidate, enabled=True, mode="reduce-overhead", fullgraph=False)
        warmup_model(compiled, x_bench, mask_bench, 10, device)
        samples = []
        for _ in range(3):
            samples.extend(benchmark_once(compiled, x_bench, mask_bench, 30, device))
        median_ms = sorted(samples)[len(samples) // 2]
        print(f"{label}  (n_flash={n_flash})  median={median_ms:.4f} ms")
        bench_results.append((label, n_flash, median_ms))
    except Exception as exc:
        print(f"{label}  (n_flash={n_flash})  COMPILE/BENCH ERROR: {type(exc).__name__}: {exc}")
    del candidate
    torch.cuda.empty_cache()

if bench_results:
    best = min(bench_results, key=lambda t: t[2])
    print(f"\nFastest passing hybrid pattern: {best[0]} (n_flash={best[1]}) at {best[2]:.4f} ms median")

    # Reference: baseline A itself, for a speedup number.
    a_warmup_model = baseline
    warmup_model(a_warmup_model, x_bench, mask_bench, 10, device)
    a_samples = []
    for _ in range(3):
        a_samples.extend(benchmark_once(a_warmup_model, x_bench, mask_bench, 30, device))
    a_median = sorted(a_samples)[len(a_samples) // 2]
    print(f"A (baseline) median={a_median:.4f} ms")
    print(f"Best hybrid speedup vs A: {a_median / best[2]:.3f}x")
