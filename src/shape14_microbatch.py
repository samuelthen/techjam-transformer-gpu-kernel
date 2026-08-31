#!/usr/bin/env python3
"""Full-length (S=100000) timing for shape 14 via batch microbatching.

Shape 14 = (B=32, D=1024, H=16, S=100000, layers=2, causal=True, ffn=1024).
A single activation tensor at full batch is B*S*D*4 bytes = 12.2 GiB in FP32,
and packed QKV triples that. That's a genuine memory wall independent of
attention algorithm (FlashAttention avoids the O(S^2) score matrix, not the
O(B*S*D) activations) -- the earlier full-batch run OOM'd inside a plain
Linear projection, before attention even ran. Fix: split the batch dimension
into chunks that fit, run sequentially, and report per-chunk latency (which
is what a data-parallel deployment across multiple idle GPUs would achieve
as its actual full-batch latency, run concurrently instead of sequentially).
"""
import sys

sys.path.insert(0, "/local1/stsj/misc")

import copy as copy_module
import torch

from transformer_ablation_benchmark import (
    AblationTransformer,
    AccuracySummary,
    BaselineTransformer,
    VARIANT_SPECS,
    benchmark_once,
    competition_config,
    copy_baseline_weights,
    copy_baseline_weights_selective,
    generate_random_case,
    maybe_compile,
    resolve_fast_dtype,
    SelectivePrecisionTransformer,
    update_accuracy_summary,
    warmup_model,
)

torch.manual_seed(1234)
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

import torch._dynamo as torch_dynamo
torch_dynamo.config.cache_size_limit = 128

device = torch.device("cuda:1")
torch.cuda.set_device(device)
config = competition_config(14)
fast_dtype = torch.float16
print(f"shape 14 config: {config}")


def sub_config(chunk):
    from transformer_ablation_benchmark import TransformerConfig
    return TransformerConfig(
        batch_size=chunk, seq_len=config.seq_len, d_model=config.d_model,
        num_heads=config.num_heads, ffn_dim=config.ffn_dim,
        num_layers=config.num_layers, causal=config.causal,
    )


# Build the FP32 reference once at a small chunk size to get comparable weights.
weight_source = BaselineTransformer(sub_config(1))
weight_cpu_state = {k: v.detach().cpu() for k, v in weight_source.state_dict().items()}


def build_baseline(chunk):
    m = BaselineTransformer(sub_config(chunk))
    m.load_state_dict(copy_module.deepcopy(weight_cpu_state))
    return m.to(device=device, dtype=torch.float32).eval()


def build_lfa(chunk):
    shadow = BaselineTransformer(sub_config(chunk))
    shadow.load_state_dict(copy_module.deepcopy(weight_cpu_state))
    m = SelectivePrecisionTransformer(
        sub_config(chunk), fast_dtype=fast_dtype, precision_mode="attention_core",
        attention_impl="flash_attn", sdpa_backend="auto",
    )
    copy_baseline_weights_selective(shadow, m)
    m.set_assume_all_valid_mask(True)
    m = m.to(device=device, dtype=torch.float32).eval()
    m.configure_selected_modules()
    return m


def build_h(chunk):
    shadow = BaselineTransformer(sub_config(chunk))
    shadow.load_state_dict(copy_module.deepcopy(weight_cpu_state))
    spec = VARIANT_SPECS["H"]
    m = AblationTransformer(sub_config(chunk), spec, sdpa_backend="auto")
    copy_baseline_weights(shadow, m)
    m.set_assume_all_valid_mask(True)
    return m.to(device=device, dtype=torch.float32).eval()


def try_chunk(build_fn, chunk):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        model = build_fn(chunk)
        x, mask = generate_random_case(
            config=sub_config(chunk), device=device, dtype=torch.float32,
            seed=999, padding_ratio=0.0, input_scale=1.0, pattern="normal",
        )
        with torch.inference_mode():
            _ = model(x, mask)
        peak_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
        del model, x, mask
        torch.cuda.empty_cache()
        return True, peak_gib
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return False, None


print("\n=== Finding largest feasible batch chunk (H, uncompiled) ===")
for chunk in (32, 16, 8, 4, 2, 1):
    ok, peak = try_chunk(build_h, chunk)
    print(f"chunk={chunk:>3}  {'OK' if ok else 'OOM'}"
          + (f"  peak={peak:.2f} GiB" if ok else ""))
    if ok:
        chunk_size = chunk
        break
else:
    print("Even chunk=1 OOM'd. Aborting.")
    sys.exit(1)

print(f"\nUsing chunk_size={chunk_size} for full timing ({config.batch_size // chunk_size} "
      f"sequential chunks + {config.batch_size % chunk_size} remainder to cover full batch={config.batch_size})")

# ---------------------------------------------------------------------------
# The exact manual-attention baseline (A) cannot run at full S=100000 at ANY
# batch size -- its S x S score matrix alone would need B*heads*S*S*4 bytes =
# 4.77 TB even at chunk=1. That's exactly why the main harness's long-safe
# mode validates exact-A only on a manageable prefix (already done: S=512,
# 10 trials/pattern, H/L/LFA/G all PASS with 0 failures) and uses a
# memory-efficient variant as the full-size reference instead. Mirror that
# here: cross-check LFA against H (both memory-safe, SDPA/FlashAttn-based,
# both already prefix-validated against exact A) at the real S=100000.
# ---------------------------------------------------------------------------
print(f"\n=== Cross-check at chunk_size={chunk_size}, full S={config.seq_len} "
      f"(LFA vs H, since exact-A is infeasible at this length regardless of chunk size) ===")
lfa = build_lfa(chunk_size)
h = build_h(chunk_size)

summary_lfa_vs_h = AccuracySummary()
patterns = ("normal", "tiny", "large", "outlier")
with torch.inference_mode():
    for p_idx, pattern in enumerate(patterns):
        for trial in range(3):
            seed = 1234 + p_idx * 100_000 + trial
            x, mask = generate_random_case(
                config=sub_config(chunk_size), device=device, dtype=torch.float32,
                seed=seed, padding_ratio=0.0, input_scale=1.0, pattern=pattern,
            )
            out_h = h(x, mask)
            out_lfa = lfa(x, mask)
            update_accuracy_summary(summary_lfa_vs_h, out_h, out_lfa, rtol=0.02, atol=0.002)
            del x, mask, out_h, out_lfa
    torch.cuda.empty_cache()

print(f"LFA vs H: {'AGREE' if summary_lfa_vs_h.passed else 'DIVERGE'}  "
      f"max_abs={summary_lfa_vs_h.max_abs_error:.6g}  "
      f"failed={summary_lfa_vs_h.failed_elements}/{summary_lfa_vs_h.total_elements}")
# ---------------------------------------------------------------------------
# Timing: one chunk's latency at full S=100000 IS the number that matters --
# it's what the full batch=32 would cost if split across ceil(32/chunk_size)
# GPUs running concurrently (you have 7 idle A6000s for exactly this). Note:
# A (manual attention) is excluded here too -- it would hit the exact same
# 4.77 TB score-matrix OOM during the forward pass, at any chunk size.
# ---------------------------------------------------------------------------
print(f"\n=== Timing at chunk_size={chunk_size}, full S={config.seq_len} (single GPU, single chunk) ===")
x_bench, mask_bench = generate_random_case(
    config=sub_config(chunk_size), device=device, dtype=torch.float32,
    seed=42, padding_ratio=0.0, input_scale=1.0, pattern="normal",
)

for name, model in (("H", h), ("LFA", lfa)):
    warmup_model(model, x_bench, mask_bench, 1, device)
    samples = benchmark_once(model, x_bench, mask_bench, 3, device)
    median_ms = sorted(samples)[len(samples) // 2]
    n_chunks_for_full_batch = -(-config.batch_size // chunk_size)  # ceil
    print(f"{name:4} chunk_size={chunk_size}  median={median_ms:.2f} ms/chunk  "
          f"(sequential full-batch equivalent: {median_ms * n_chunks_for_full_batch:.2f} ms over "
          f"{n_chunks_for_full_batch} chunks; parallel-across-GPUs equivalent: {median_ms:.2f} ms)")
