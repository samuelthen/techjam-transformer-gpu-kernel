#!/usr/bin/env python3
"""Isolate why SDPA attention diverges from manual attention under outlier inputs.

Hypotheses tested:
  1. TF32 matmul inside the fused SDPA kernel (independent of the manual path).
  2. Which concrete SDPA backend 'auto' actually selects on this GPU/dtype.
  3. Whether forcing math/flash/efficient/cudnn backends changes the divergence.
  4. Whether the divergence is caused by the causal mask path specifically
     (compare non-causal too) or is present regardless.
"""
import sys
sys.path.insert(0, "/local1/stsj/misc")

import torch
import torch.nn.functional as F
from transformer_ablation_benchmark import (
    BaselineSelfAttention,
    AblationSelfAttention,
    generate_random_case,
    TransformerConfig,
    sdpa_backend_context,
    update_accuracy_summary,
    AccuracySummary,
)

torch.manual_seed(0)
device = torch.device("cuda")
d_model, num_heads, seq_len, batch = 128, 4, 128, 64
config = TransformerConfig(
    batch_size=batch, seq_len=seq_len, d_model=d_model,
    num_heads=num_heads, ffn_dim=128, num_layers=1, causal=True,
)

print(f"torch={torch.__version__}, gpu={torch.cuda.get_device_name(0)}")
print(f"cuda_capability={torch.cuda.get_device_capability(0)}")
try:
    from torch.backends.cuda import (
        flash_sdp_enabled, mem_efficient_sdp_enabled, math_sdp_enabled, cudnn_sdp_enabled,
    )
    print(f"default backend flags: flash={flash_sdp_enabled()}, mem_eff={mem_efficient_sdp_enabled()}, "
          f"math={math_sdp_enabled()}, cudnn={cudnn_sdp_enabled()}")
except Exception as exc:
    print("could not query default backend flags:", exc)


def build_pair(allow_tf32: bool):
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32

    baseline = BaselineSelfAttention(d_model, num_heads).to(device=device, dtype=torch.float32).eval()

    candidate = AblationSelfAttention(
        d_model, num_heads, packed_qkv=False, sdpa=True,
    ).to(device=device, dtype=torch.float32).eval()

    with torch.no_grad():
        candidate.q_proj.load_state_dict(baseline.q_proj.state_dict())
        candidate.k_proj.load_state_dict(baseline.k_proj.state_dict())
        candidate.v_proj.load_state_dict(baseline.v_proj.state_dict())
        candidate.out_proj.load_state_dict(baseline.out_proj.state_dict())

    return baseline, candidate


def worst_error(ref: torch.Tensor, cand: torch.Tensor, atol=0.002, rtol=0.02):
    diff = (cand.float() - ref.float()).abs()
    denom = ref.float().abs().clamp_min(1e-12)
    rel = diff / denom
    abs_ok = diff <= atol
    rel_ok = diff <= rtol * ref.float().abs()
    failed = int((~(abs_ok | rel_ok)).sum().item())
    return float(diff.max().item()), float(rel.max().item()), failed, ref.numel()


def run_case(baseline, candidate, causal, pattern, seed, backend="auto"):
    x, mask = generate_random_case(
        config=config, device=device, dtype=torch.float32,
        seed=seed, padding_ratio=0.0, input_scale=1.0, pattern=pattern,
    )
    with torch.no_grad():
        ref = baseline(x, None, causal=causal)
        candidate.sdpa_backend = backend
        cand = candidate(x, None, causal=causal)
    return worst_error(ref, cand)


print("\n=== 1. Effect of TF32 on/off, backend=auto, causal, outlier pattern, 20 trials ===")
for allow_tf32 in (True, False):
    baseline, candidate = build_pair(allow_tf32)
    max_abs = max_rel = 0.0
    total_failed = total_elems = 0
    for trial in range(20):
        a, r, f, n = run_case(baseline, candidate, True, "outlier", seed=9000 + trial, backend="auto")
        max_abs, max_rel = max(max_abs, a), max(max_rel, r)
        total_failed += f
        total_elems += n
    print(f"allow_tf32={allow_tf32!s:5} -> max_abs={max_abs:.6f} max_rel={max_rel:.3f} "
          f"failed={total_failed}/{total_elems}")

print("\n=== 2. Effect of forced SDPA backend, TF32 restored to True, outlier pattern, 20 trials ===")
baseline, candidate = build_pair(True)
for backend in ("auto", "math", "flash", "efficient", "cudnn"):
    try:
        max_abs = max_rel = 0.0
        total_failed = total_elems = 0
        ok = True
        for trial in range(20):
            a, r, f, n = run_case(baseline, candidate, True, "outlier", seed=9000 + trial, backend=backend)
            max_abs, max_rel = max(max_abs, a), max(max_rel, r)
            total_failed += f
            total_elems += n
        print(f"backend={backend:10} -> max_abs={max_abs:.6f} max_rel={max_rel:.3f} "
              f"failed={total_failed}/{total_elems}")
    except Exception as exc:
        print(f"backend={backend:10} -> ERROR: {type(exc).__name__}: {exc}")

print("\n=== 3. Causal vs non-causal, backend=auto, TF32=True, outlier pattern, 20 trials ===")
baseline, candidate = build_pair(True)
for causal in (True, False):
    max_abs = max_rel = 0.0
    total_failed = total_elems = 0
    for trial in range(20):
        a, r, f, n = run_case(baseline, candidate, causal, "outlier", seed=9000 + trial, backend="auto")
        max_abs, max_rel = max(max_abs, a), max(max_rel, r)
        total_failed += f
        total_elems += n
    print(f"causal={causal!s:5} -> max_abs={max_abs:.6f} max_rel={max_rel:.3f} "
          f"failed={total_failed}/{total_elems}")

print("\n=== 4. Pattern breakdown, backend=auto, TF32=True, causal, 20 trials each ===")
baseline, candidate = build_pair(True)
for pattern in ("normal", "tiny", "large", "outlier"):
    max_abs = max_rel = 0.0
    total_failed = total_elems = 0
    for trial in range(20):
        a, r, f, n = run_case(baseline, candidate, True, pattern, seed=9000 + trial, backend="auto")
        max_abs, max_rel = max(max_abs, a), max(max_rel, r)
        total_failed += f
        total_elems += n
    print(f"pattern={pattern:8} -> max_abs={max_abs:.6f} max_rel={max_rel:.3f} "
          f"failed={total_failed}/{total_elems}")
