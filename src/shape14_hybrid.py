#!/usr/bin/env python3
"""Shape 14 (S=100000, 2 layers) per-layer hybrid check, binary none/core only.

Earlier full-length cross-check found LFA (both layers FP16-FlashAttn core)
DIVERGES from H (33910/9.83e9 elements) at true S=100000, despite passing at
the S=512 prefix -- same "rare per-element failure becomes visible at extreme
element count" pattern as shape 6, just triggered by sequence length instead
of batch size here. With only 2 layers, the binary search space is just 4
patterns (PP/PL/LP/LL); "proj" (VFA-level) is skipped given each H forward
pass alone costs ~60s at this length, making a 3-way 9-pattern search too
expensive to justify without first knowing if the 2-level search resolves it.
"""
import copy as copy_module
import sys

sys.path.insert(0, "/local1/stsj/misc")

import torch

from transformer_ablation_benchmark import (
    AblationTransformer,
    AccuracySummary,
    BaselineTransformer,
    SelectivePrecisionTransformer,
    TransformerConfig,
    VARIANT_SPECS,
    benchmark_once,
    competition_config,
    copy_baseline_weights,
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
torch_dynamo.config.cache_size_limit = 64

device = torch.device("cuda:6")
torch.cuda.set_device(device)

config = competition_config(14)
CHUNK = 8
fast_dtype = torch.float16
print(f"shape 14 config: {config}, chunk_size={CHUNK}")


def sub_config(chunk):
    return TransformerConfig(
        batch_size=chunk, seq_len=config.seq_len, d_model=config.d_model,
        num_heads=config.num_heads, ffn_dim=config.ffn_dim,
        num_layers=config.num_layers, causal=config.causal,
    )


weight_source = BaselineTransformer(sub_config(1))
weight_cpu_state = {k: v.detach().cpu() for k, v in weight_source.state_dict().items()}


def build_h():
    shadow = BaselineTransformer(sub_config(CHUNK))
    shadow.load_state_dict(copy_module.deepcopy(weight_cpu_state))
    spec = VARIANT_SPECS["H"]
    m = AblationTransformer(sub_config(CHUNK), spec, sdpa_backend="auto")
    copy_baseline_weights(shadow, m)
    m.set_assume_all_valid_mask(True)
    return m.to(device=device, dtype=torch.float32).eval()


def build_hybrid(layer_pattern):
    """layer_pattern: list of bool, True = FP16 FlashAttn core (LFA-level), False = FP32 SDPA."""
    shadow = BaselineTransformer(sub_config(CHUNK))
    shadow.load_state_dict(copy_module.deepcopy(weight_cpu_state))
    m = SelectivePrecisionTransformer(
        sub_config(CHUNK), fast_dtype=fast_dtype, precision_mode="attention_core",
        attention_impl="flash_attn", sdpa_backend="auto", layer_pattern=layer_pattern,
    )
    copy_baseline_weights_selective(shadow, m)
    m.set_assume_all_valid_mask(True)
    m = m.to(device=device, dtype=torch.float32).eval()
    m.configure_selected_modules()
    return m


h_ref = build_h()

patterns = [
    (False, False),  # PP = pure H
    (False, True),   # PL
    (True, False),   # LP
    (True, True),    # LL = pure LFA (known to diverge from H)
]

print("\n=== Cross-check vs H at chunk_size=8, full S=100000, 3 trials/pattern (12 total) ===")
results = {}
for pattern in patterns:
    label = "".join("L" if b else "P" for b in pattern)
    if pattern == (False, False):
        print(f"{label}  (== H itself, skipping self-comparison)")
        continue
    candidate = build_hybrid(list(pattern))
    summary = AccuracySummary()
    stress_patterns = ("normal", "tiny", "large")  # skip "outlier" to save time; core signal is in normal/large
    with torch.inference_mode():
        for p_idx, sp in enumerate(stress_patterns):
            for trial in range(1):
                seed = 1234 + p_idx * 100_000 + trial
                x, mask = generate_random_case(
                    config=sub_config(CHUNK), device=device, dtype=torch.float32,
                    seed=seed, padding_ratio=0.0, input_scale=1.0, pattern=sp,
                )
                ref = h_ref(x, mask)
                out = candidate(x, mask)
                update_accuracy_summary(summary, ref, out, rtol=0.02, atol=0.002)
                del x, mask, ref, out
    torch.cuda.empty_cache()
    status = "AGREE" if summary.passed else "DIVERGE"
    print(f"{label}  {status}  max_abs={summary.max_abs_error:.6g}  "
          f"failed={summary.failed_elements}/{summary.total_elements}")
    results[pattern] = (status, candidate)

print("\n=== Timing agreeing patterns (uncompiled -- compile cost not worth it at this scale for a quick check) ===")
x_bench, mask_bench = generate_random_case(
    config=sub_config(CHUNK), device=device, dtype=torch.float32,
    seed=42, padding_ratio=0.0, input_scale=1.0, pattern="normal",
)
warmup_model(h_ref, x_bench, mask_bench, 1, device)
h_samples = benchmark_once(h_ref, x_bench, mask_bench, 2, device)
h_median = sorted(h_samples)[len(h_samples) // 2]
print(f"H (PP) median={h_median:.2f} ms/chunk")

for pattern, (status, candidate) in results.items():
    label = "".join("L" if b else "P" for b in pattern)
    if status != "AGREE":
        print(f"{label}  DIVERGE -- not timed (would not pass the correctness gate)")
        continue
    warmup_model(candidate, x_bench, mask_bench, 1, device)
    samples = benchmark_once(candidate, x_bench, mask_bench, 2, device)
    median_ms = sorted(samples)[len(samples) // 2]
    print(f"{label}  median={median_ms:.2f} ms/chunk  speedup vs H={h_median/median_ms:.3f}x")
