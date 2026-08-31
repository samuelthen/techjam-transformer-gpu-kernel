#!/usr/bin/env python3
"""
Transformer optimization benchmark V6 precision-search.

Automatically evaluates a 2^3 ablation over three optimizations:

A  baseline                 separate QKV + manual attention + normal masking
B  packed_qkv              packed QKV   + manual attention + normal masking
C  sdpa                    separate QKV + SDPA             + normal masking
D  mask_skip               separate QKV + manual attention + skip no-op all-valid masks
E  packed_qkv+sdpa         packed QKV   + SDPA             + normal masking
F  packed_qkv+mask_skip    packed QKV   + manual attention + skip no-op all-valid masks
G  sdpa+mask_skip          separate QKV + SDPA             + skip no-op all-valid masks
H  full                    packed QKV   + SDPA             + skip no-op all-valid masks

Every candidate is compared against the exact baseline output using:

    abs(candidate - reference) <= atol
    OR
    abs(candidate - reference) <= rtol * abs(reference)

The benchmark uses the same fixed input for every variant and rotates measurement
order across rounds to reduce thermal / clock-order bias.

Competition/performance additions:
- A-H remain the original clean 2^3 factorial
- I: exact H + tanh GELU approximation (correctness-gated)
- J/K: reduced-precision candidate compute with output cast back to reference dtype
- L-O: candidate-only torch.compile combinations
- P: optional external FlashAttention packed-layout path (CUDA)
- Q: optional SageAttention low-bit path (CUDA)
- R/S/T: force Flash / cuDNN / memory-efficient PyTorch SDPA backends (CUDA)
- U/V/W: CUDA selective mixed precision that preserves FP32 norms/residuals while
  progressively moving attention and GEMMs to FP16/BF16 Tensor Core paths
- X: low-precision packed QKV + attention, but FP32 output projection
- Y: FP32 packed QKV, low-precision attention + output projection
- Z1-Z6: progressively apply V-style low precision to only the first N layers;
  remaining layers use the H-style FP32 packed-QKV + SDPA path
- P/Q external FlashAttention/SageAttention use the same selective-precision
  boundaries instead of converting the entire Transformer to low precision
- interleaved blocked timing reduces thermal/order bias for small 1-3% effects
- official 14-shape sweep plus JSON best-variant dispatch generation
- long-sequence-safe mode validates exact A on a prefix and times every selected
  memory-efficient candidate at full S without materializing SxS matrices
- stronger optional numerical stress testing with tiny/large/outlier inputs
- automatic low-repeat timing for very long sequences

Useful suites:
  --variants fast       -> A,H,I,J,K,L,M,N,O
  --variants selective        -> A,H,U,X,Y,Z1-Z6,V,W
  --variants precision-search -> A,H,U,X,Y,Z1-Z6,V,W
  --variants cuda-fast        -> fast + precision search + P,Q,R,S,T
  --variants everything       -> every registered variant

Examples
--------
Mac / Apple Silicon:

python transformer_ablation_benchmark.py \
  --device mps --dtype float32 \
  --batch-size 8 --seq-len 128 --d-model 512 \
  --heads 8 --ffn-dim 2048 --layers 6 \
  --accuracy-trials 5 --warmup 20 --repeats 100 --benchmark-rounds 3

Only A/B/C/H:

python transformer_ablation_benchmark.py \
  --device mps --dtype float32 \
  --variants A,B,C,H

CUDA, force FlashAttention SDPA where supported:

python transformer_ablation_benchmark.py \
  --device cuda --dtype bfloat16 \
  --sdpa-backend flash --quick

Official competition shapes 1-13 (shape 14 uses long-safe mode):

python transformer_ablation_benchmark.py \
  --device cuda --dtype bfloat16 \
  --competition-shapes --shape-ids 1-13 \
  --variants A,C,H --csv competition_ablation.csv

Save CSV:

python transformer_ablation_benchmark.py \
  --device mps --dtype float32 \
  --csv ablation_results.csv
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import statistics
import time
from functools import lru_cache
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class TransformerConfig:
    batch_size: int
    seq_len: int
    d_model: int
    num_heads: int
    ffn_dim: int
    num_layers: int
    causal: bool

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if self.ffn_dim <= 0:
            raise ValueError("ffn_dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")


# Official challenge shapes are encoded as:
# (batch_size, d_model, num_heads, seq_len, num_layers, causal, ffn_dim)
COMPETITION_SHAPES: Dict[int, Tuple[int, int, int, int, int, bool, int]] = {
    1: (64, 128, 4, 128, 4, True, 128),
    2: (1, 128, 4, 128, 4, True, 128),
    3: (4, 128, 4, 128, 4, True, 128),
    4: (16, 128, 4, 128, 4, True, 128),
    5: (128, 128, 4, 128, 4, True, 128),
    6: (10000, 128, 4, 128, 4, True, 128),
    7: (64, 32, 4, 128, 4, True, 32),
    8: (64, 1024, 4, 128, 4, True, 1024),
    9: (64, 128, 1, 128, 4, True, 128),
    10: (64, 128, 2, 128, 4, True, 128),
    11: (64, 128, 16, 128, 4, True, 128),
    12: (64, 128, 4, 32, 4, True, 128),
    13: (64, 128, 4, 1024, 4, True, 128),
    14: (32, 1024, 16, 100000, 2, True, 1024),
}


def competition_config(shape_id: int) -> TransformerConfig:
    try:
        batch, d_model, heads, seq_len, layers, causal, ffn_dim = (
            COMPETITION_SHAPES[shape_id]
        )
    except KeyError as exc:
        raise ValueError(f"unknown competition shape id: {shape_id}") from exc

    return TransformerConfig(
        batch_size=batch,
        seq_len=seq_len,
        d_model=d_model,
        num_heads=heads,
        ffn_dim=ffn_dim,
        num_layers=layers,
        causal=causal,
    )


@dataclass(frozen=True)
class VariantSpec:
    key: str
    name: str
    packed_qkv: bool
    sdpa: bool
    skip_all_valid_mask: bool
    gelu_approx: str = "none"
    fast_dtype: bool = False
    compile_candidate: bool = False
    attention_impl: str = "native"  # native / flash_attn / sage
    sdpa_backend_override: Optional[str] = None
    # Selective precision keeps norms/residual state in the reference dtype.
    # Modes:
    #   none                 : H-style FP32 packed-QKV + SDPA
    #   attention_core       : FP32 QKV/out projections, low-precision attention
    #   attention_qkv        : low-precision QKV + attention, FP32 out projection
    #   attention_out        : FP32 QKV, low-precision attention + out projection
    #   attention_projection : low-precision QKV + attention + out projection
    #   attention_ffn        : attention_projection + low-precision FFN GEMMs
    selective_precision: str = "none"
    # Optional first-N layer limit for the selective mode. None means all layers.
    selective_layers: Optional[int] = None


# A-H preserve the original factorial experiment exactly.
# I-T are performance experiments. Every one is still correctness-gated against A.
VARIANT_SPECS: Dict[str, VariantSpec] = {
    "A": VariantSpec("A", "baseline", False, False, False),
    "B": VariantSpec("B", "packed_qkv", True, False, False),
    "C": VariantSpec("C", "sdpa", False, True, False),
    "D": VariantSpec("D", "mask_skip", False, False, True),
    "E": VariantSpec("E", "packed_qkv+sdpa", True, True, False),
    "F": VariantSpec("F", "packed_qkv+mask_skip", True, False, True),
    "G": VariantSpec("G", "sdpa+mask_skip", False, True, True),
    "H": VariantSpec("H", "full", True, True, True),

    # Tolerance-driven / compiler experiments.
    "I": VariantSpec(
        "I", "full+gelu_tanh", True, True, True,
        gelu_approx="tanh",
    ),
    "J": VariantSpec(
        "J", "full+fast_dtype", True, True, True,
        fast_dtype=True,
    ),
    "K": VariantSpec(
        "K", "full+tanh+fast_dtype", True, True, True,
        gelu_approx="tanh", fast_dtype=True,
    ),
    "L": VariantSpec(
        "L", "full+compile", True, True, True,
        compile_candidate=True,
    ),
    "M": VariantSpec(
        "M", "full+tanh+compile", True, True, True,
        gelu_approx="tanh", compile_candidate=True,
    ),
    "N": VariantSpec(
        "N", "full+fast_dtype+compile", True, True, True,
        fast_dtype=True, compile_candidate=True,
    ),
    "O": VariantSpec(
        "O", "full+tanh+fast_dtype+compile", True, True, True,
        gelu_approx="tanh", fast_dtype=True, compile_candidate=True,
    ),

    # N-style whole-model FP16 + compile, but forcing a specific PyTorch SDPA
    # backend instead of letting 'auto' pick (mirrors R/S/T for the fast_dtype
    # + compile combination).
    "NF": VariantSpec(
        "NF", "full+fast_dtype+compile+sdpa_flash", True, True, True,
        fast_dtype=True, compile_candidate=True, sdpa_backend_override="flash",
    ),
    "NE": VariantSpec(
        "NE", "full+fast_dtype+compile+sdpa_efficient", True, True, True,
        fast_dtype=True, compile_candidate=True, sdpa_backend_override="efficient",
    ),
    "NC": VariantSpec(
        "NC", "full+fast_dtype+compile+sdpa_cudnn", True, True, True,
        fast_dtype=True, compile_candidate=True, sdpa_backend_override="cudnn",
    ),
    # N-style whole-model FP16 + compile, but with external FlashAttention-2
    # replacing SDPA entirely (packed QKV feeds flash_attn_qkvpacked_func
    # directly, skipping the SDPA B,H,S,D transpose).
    "NFA": VariantSpec(
        "NFA", "full+fast_dtype+compile+flash_attn", True, False, True,
        fast_dtype=True, compile_candidate=True, attention_impl="flash_attn",
    ),

    # Paper-driven CUDA attention/layout experiments.
    # P keeps packed QKV in [B,S,H,D] and calls external flash-attn directly,
    # avoiding the SDPA head-layout transpose path.
    "P": VariantSpec(
        "P", "selective+flash_attn", True, False, True,
        attention_impl="flash_attn", selective_precision="attention_projection",
    ),
    # LFA: U's precision boundary (FP32 LayerNorm/QKV-proj/out-proj/FFN, only
    # the attention core itself runs in FP16) but with external FlashAttention-2
    # in place of SDPA for that FP16 core, plus torch.compile around the whole
    # graph. This is the "legal-looking" FA2 variant: everything that touches
    # the residual stream stays FP32, only the attention math is low precision.
    "LFA": VariantSpec(
        "LFA", "attention_core+flash_attn+compile", True, False, True,
        attention_impl="flash_attn", selective_precision="attention_core",
        compile_candidate=True,
    ),
    # VFA: V's precision boundary (QKV proj + attention + out proj all FP16,
    # i.e. P's precision mode) with torch.compile added on top. More aggressive
    # than LFA and expected to be riskier for accuracy, since V already failed.
    "VFA": VariantSpec(
        "VFA", "attention_projection+flash_attn+compile", True, False, True,
        attention_impl="flash_attn", selective_precision="attention_projection",
        compile_candidate=True,
    ),
    # Q calls the generic SageAttention entry point when the package is installed.
    # It uses selective precision so LayerNorm, residuals, GELU, and the FFN stay FP32.
    "Q": VariantSpec(
        "Q", "selective+sageattention", True, False, True,
        attention_impl="sage", selective_precision="attention_projection",
    ),

    # Force PyTorch SDPA backends so the known competition shapes can discover
    # which library route actually wins on the target GPU.
    "R": VariantSpec(
        "R", "full+sdpa_flash", True, True, True,
        sdpa_backend_override="flash",
    ),
    "S": VariantSpec(
        "S", "full+sdpa_cudnn", True, True, True,
        sdpa_backend_override="cudnn",
    ),
    "T": VariantSpec(
        "T", "full+sdpa_efficient", True, True, True,
        sdpa_backend_override="efficient",
    ),

    # Selective mixed precision. These variants keep the model state, LayerNorms,
    # residual additions, final norm, and (for W) GELU in FP32 while moving only
    # the expensive Tensor-Core-friendly regions to fast_dtype.
    "U": VariantSpec(
        "U", "selective+attention_core", True, True, True,
        selective_precision="attention_core",
    ),
    "V": VariantSpec(
        "V", "selective+attention_projection", True, True, True,
        selective_precision="attention_projection",
    ),
    "W": VariantSpec(
        "W", "selective+attention+ffn_gemms", True, True, True,
        selective_precision="attention_ffn",
    ),

    # Split V to identify which projection causes the numerical failure.
    # X lowers QKV + attention but deliberately returns to FP32 for out_proj.
    "X": VariantSpec(
        "X", "selective+qkv+attention", True, True, True,
        selective_precision="attention_qkv",
    ),
    # Y keeps QKV in FP32, lowers only the attention core and out projection.
    "Y": VariantSpec(
        "Y", "selective+attention+out_proj", True, True, True,
        selective_precision="attention_out",
    ),

    # Progressive V-style precision: only the first N Transformer layers lower
    # QKV + attention + out projection. Remaining layers are H-style FP32.
    "Z1": VariantSpec(
        "Z1", "selective+proj_first1", True, True, True,
        selective_precision="attention_projection", selective_layers=1,
    ),
    "Z2": VariantSpec(
        "Z2", "selective+proj_first2", True, True, True,
        selective_precision="attention_projection", selective_layers=2,
    ),
    "Z3": VariantSpec(
        "Z3", "selective+proj_first3", True, True, True,
        selective_precision="attention_projection", selective_layers=3,
    ),
    "Z4": VariantSpec(
        "Z4", "selective+proj_first4", True, True, True,
        selective_precision="attention_projection", selective_layers=4,
    ),
    "Z5": VariantSpec(
        "Z5", "selective+proj_first5", True, True, True,
        selective_precision="attention_projection", selective_layers=5,
    ),
    "Z6": VariantSpec(
        "Z6", "selective+proj_first6", True, True, True,
        selective_precision="attention_projection", selective_layers=6,
    ),
}


CORE_VARIANTS = list("ABCDEFGH")
FAST_VARIANTS = ["A", "H", "I", "J", "K", "L", "M", "N", "O"]
SELECTIVE_VARIANTS = [
    "A", "H", "U", "X", "Y",
    "Z1", "Z2", "Z3", "Z4", "Z5", "Z6",
    "V", "W",
]
PRECISION_SEARCH_VARIANTS = list(SELECTIVE_VARIANTS)
CUDA_FAST_VARIANTS = [
    "A", "H", "I", "J", "K", "L", "M", "N", "O",
    "U", "X", "Y", "Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "V", "W",
    "P", "Q", "R", "S", "T",
]


def variant_attention_label(spec: VariantSpec) -> str:
    if spec.attention_impl == "flash_attn":
        return "flash-attn"
    if spec.attention_impl == "sage":
        return "sage"
    if spec.sdpa:
        if spec.sdpa_backend_override:
            return f"sdpa:{spec.sdpa_backend_override}"
        return "sdpa"
    return "manual"


def variant_is_memory_safe(spec: VariantSpec) -> bool:
    """True when the full attention path does not materialize an SxS score/mask."""
    if not spec.skip_all_valid_mask:
        return False
    return spec.sdpa or spec.attention_impl in {"flash_attn", "sage"}


def variant_uses_fast_dtype(spec: VariantSpec) -> bool:
    return spec.fast_dtype or spec.selective_precision != "none"


def variant_selective_label(spec: VariantSpec) -> str:
    if spec.selective_precision == "none":
        return "none"
    if spec.selective_layers is None:
        return spec.selective_precision
    return f"{spec.selective_precision}:first{spec.selective_layers}"


def resolve_fast_dtype(
    fast_dtype_arg: str,
    base_dtype: torch.dtype,
    device: torch.device,
) -> torch.dtype:
    if fast_dtype_arg == "float16":
        return torch.float16
    if fast_dtype_arg == "bfloat16":
        return torch.bfloat16
    if fast_dtype_arg != "auto":
        raise ValueError(f"unknown fast dtype: {fast_dtype_arg}")

    # If the benchmark is already reduced precision, don't add pointless casts.
    if base_dtype in (torch.float16, torch.bfloat16):
        return base_dtype

    # FP16 is the most portable fast path across MPS and NVIDIA GPUs.
    if device.type in {"mps", "cuda"}:
        return torch.float16
    return base_dtype


@lru_cache(maxsize=1)
def load_flash_attn_func():
    errors: List[str] = []
    try:
        from flash_attn import flash_attn_func
        return flash_attn_func
    except Exception as exc:
        errors.append(f"flash_attn: {type(exc).__name__}: {exc}")

    # FlashAttention-3 beta commonly exposes this top-level module.
    try:
        from flash_attn_interface import flash_attn_func
        return flash_attn_func
    except Exception as exc:
        errors.append(f"flash_attn_interface: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        "external flash-attn is not importable; install a compatible "
        "FlashAttention package. Attempts: " + " | ".join(errors)
    )


@lru_cache(maxsize=1)
def load_flash_attn_qkvpacked_func():
    """Prefer the packed-QKV API so projection output can feed attention directly."""
    try:
        from flash_attn import flash_attn_qkvpacked_func
        return flash_attn_qkvpacked_func
    except Exception:
        return None


@lru_cache(maxsize=1)
def load_sage_attn_func():
    try:
        from sageattention import sageattn
        return sageattn
    except Exception as exc:
        raise RuntimeError(
            "SageAttention is not importable; install a compatible "
            f"sageattention package ({type(exc).__name__}: {exc})"
        ) from exc


class ComputeDtypeWrapper(nn.Module):
    """Run a candidate in a lower compute dtype while preserving the input/output contract."""

    def __init__(self, model: nn.Module, compute_dtype: torch.dtype) -> None:
        super().__init__()
        self.model = model
        self.compute_dtype = compute_dtype

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_dtype = x.dtype
        if x.dtype != self.compute_dtype:
            x = x.to(dtype=self.compute_dtype)
        y = self.model(x, valid_token_mask)
        if y.dtype != input_dtype:
            y = y.to(dtype=input_dtype)
        return y


class UnavailableVariant(nn.Module):
    """Model placeholder that turns an unsupported experiment into a reported ERROR."""

    def __init__(self, reason: str) -> None:
        super().__init__()
        self.reason = reason

    def forward(self, *args, **kwargs):
        raise RuntimeError(self.reason)


def sdpa_backend_context(backend: str, device: torch.device):
    """Return a context manager selecting a CUDA SDPA backend when requested."""
    if backend == "auto" or device.type != "cuda":
        return nullcontext()

    # PyTorch 2.5+ API. Keep a compatibility fallback for older builds.
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        mapping = {
            "flash": SDPBackend.FLASH_ATTENTION,
            "efficient": SDPBackend.EFFICIENT_ATTENTION,
            "math": SDPBackend.MATH,
        }
        if hasattr(SDPBackend, "CUDNN_ATTENTION"):
            mapping["cudnn"] = SDPBackend.CUDNN_ATTENTION

        if backend not in mapping:
            if backend == "cudnn":
                raise RuntimeError(
                    "this PyTorch build does not expose the cuDNN SDPA backend"
                )
            raise ValueError(f"unknown SDPA backend: {backend}")

        return sdpa_kernel(mapping[backend])
    except ImportError:
        if backend == "cudnn":
            raise RuntimeError(
                "cuDNN SDPA backend selection requires a newer PyTorch build"
            )

        flags = {
            "flash": dict(
                enable_flash=True,
                enable_math=False,
                enable_mem_efficient=False,
            ),
            "efficient": dict(
                enable_flash=False,
                enable_math=False,
                enable_mem_efficient=True,
            ),
            "math": dict(
                enable_flash=False,
                enable_math=True,
                enable_mem_efficient=False,
            ),
        }
        return torch.backends.cuda.sdp_kernel(**flags[backend])


def estimate_manual_attention_working_set_gib(
    config: TransformerConfig,
    dtype: torch.dtype,
) -> float:
    """Estimate the dominant explicit-attention score + fp32-softmax buffers."""
    score_elements = (
        config.batch_size
        * config.num_heads
        * config.seq_len
        * config.seq_len
    )
    score_element_size = torch.empty((), dtype=dtype).element_size()

    # The manual path materializes scores in input dtype and softmax in fp32.
    # This deliberately ignores smaller Q/K/V/context buffers, so it is a lower
    # bound rather than a full allocator model.
    bytes_estimate = score_elements * (score_element_size + 4)
    return bytes_estimate / (1024**3)


def estimate_tensor_gib(
    config: TransformerConfig,
    dtype: torch.dtype,
    multiplier: float = 1.0,
) -> float:
    """Estimate B*S*D storage, optionally multiplied for QKV-like tensors."""
    elements = config.batch_size * config.seq_len * config.d_model
    element_size = torch.empty((), dtype=dtype).element_size()
    return elements * element_size * multiplier / (1024**3)


def estimate_attention_flops(config: TransformerConfig) -> float:
    """Approximate attention-only forward FLOPs across all Transformer layers.

    Counts QK^T and P@V. For causal attention, approximately half the score
    matrix is useful, matching the convention used by FlashAttention papers.
    Projection and FFN FLOPs are intentionally excluded.
    """
    flops_per_layer = (
        4.0
        * config.batch_size
        * config.seq_len
        * config.seq_len
        * config.d_model
    )
    if config.causal:
        flops_per_layer *= 0.5
    return flops_per_layer * config.num_layers


def parse_shape_ids(raw: str) -> List[int]:
    """Parse forms such as '1-13', '1,2,7-9', or 'all'."""
    raw = raw.strip().lower()
    if raw == "all":
        return list(COMPETITION_SHAPES)

    result: List[int] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            start_s, end_s = piece.split("-", 1)
            start, end = int(start_s), int(end_s)
            if start > end:
                raise ValueError(f"invalid shape range: {piece}")
            result.extend(range(start, end + 1))
        else:
            result.append(int(piece))

    deduped: List[int] = []
    for shape_id in result:
        if shape_id not in COMPETITION_SHAPES:
            raise ValueError(
                f"shape id {shape_id} is not in 1-{max(COMPETITION_SHAPES)}"
            )
        if shape_id not in deduped:
            deduped.append(shape_id)

    if not deduped:
        raise ValueError("shape id selection is empty")
    return deduped


# =============================================================================
# Exact baseline
# =============================================================================

class BaselineSelfAttention(nn.Module):
    """Explicit multi-head self-attention implemented with native PyTorch ops."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return (
            x.view(batch, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if causal:
            causal_mask = torch.ones(
                (seq_len, seq_len),
                device=x.device,
                dtype=torch.bool,
            ).triu(diagonal=1)
            scores = scores.masked_fill(causal_mask, float("-inf"))

        if valid_token_mask is not None:
            invalid_keys = ~valid_token_mask[:, None, None, :]
            scores = scores.masked_fill(invalid_keys, float("-inf"))

        probs = torch.softmax(scores.float(), dim=-1).to(dtype=x.dtype)
        context = torch.matmul(probs, v)

        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch, seq_len, self.d_model)
        )

        output = self.out_proj(context)

        if valid_token_mask is not None:
            output = output.masked_fill(~valid_token_mask[..., None], 0)

        return output


class BaselineTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = BaselineSelfAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, d_model)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        causal: bool,
    ) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), valid_token_mask, causal)
        x = x + self.ffn_out(
            F.gelu(
                self.ffn_in(self.norm2(x)),
                approximate="none",
            )
        )

        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)

        return x


class BaselineTransformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [
                BaselineTransformerBlock(
                    config.d_model,
                    config.num_heads,
                    config.ffn_dim,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, valid_token_mask, self.config.causal)

        x = self.final_norm(x)

        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)

        return x


# =============================================================================
# Generic ablation implementation
# =============================================================================

class AblationSelfAttention(nn.Module):
    """Attention implementation controlled by a VariantSpec."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        packed_qkv: bool,
        sdpa: bool,
        attention_impl: str = "native",
        sdpa_backend: str = "auto",
    ) -> None:
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5
        self.packed_qkv = packed_qkv
        self.use_sdpa = sdpa
        self.attention_impl = attention_impl
        self.sdpa_backend = sdpa_backend

        if packed_qkv:
            self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        else:
            self.q_proj = nn.Linear(d_model, d_model, bias=True)
            self.k_proj = nn.Linear(d_model, d_model, bias=True)
            self.v_proj = nn.Linear(d_model, d_model, bias=True)

        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def _packed_qkv_bshd(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Packed projection preserving [B,S,H,D], useful for flash-attn layout."""
        batch, seq_len, _ = x.shape
        qkv = self.qkv_proj(x).view(
            batch,
            seq_len,
            3,
            self.num_heads,
            self.head_dim,
        )
        return qkv.unbind(dim=2)

    def _project_qkv_hnsd(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape

        if self.packed_qkv:
            q, k, v = self._packed_qkv_bshd(x)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            if not self.use_sdpa and self.attention_impl == "native":
                q = q.contiguous()
                k = k.contiguous()
                v = v.contiguous()
            return q, k, v

        q = (
            self.q_proj(x)
            .view(batch, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(batch, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(batch, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        if not self.use_sdpa and self.attention_impl == "native":
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()

        return q, k, v

    def _manual_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        causal: bool,
    ) -> torch.Tensor:
        seq_len = q.shape[-2]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if causal:
            causal_mask = torch.ones(
                (seq_len, seq_len),
                device=q.device,
                dtype=torch.bool,
            ).triu(diagonal=1)
            scores = scores.masked_fill(causal_mask, float("-inf"))

        if valid_token_mask is not None:
            invalid_keys = ~valid_token_mask[:, None, None, :]
            scores = scores.masked_fill(invalid_keys, float("-inf"))

        probs = torch.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
        return torch.matmul(probs, v)

    def _sdpa_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        causal: bool,
    ) -> torch.Tensor:
        seq_len = q.shape[-2]
        attn_mask: Optional[torch.Tensor] = None
        use_is_causal = causal

        if valid_token_mask is not None:
            key_mask = valid_token_mask[:, None, None, :]
            if causal:
                # Boolean SDPA masks use True = allowed.
                causal_allowed = torch.ones(
                    (seq_len, seq_len),
                    device=q.device,
                    dtype=torch.bool,
                ).tril()
                attn_mask = key_mask & causal_allowed[None, None, :, :]
                use_is_causal = False
            else:
                attn_mask = key_mask

        with sdpa_backend_context(self.sdpa_backend, q.device):
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=use_is_causal,
            )

    def _flash_attention_bshd(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        causal: bool,
    ) -> torch.Tensor:
        if q.device.type != "cuda":
            raise RuntimeError("external flash-attn requires CUDA")
        if q.dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError(
                f"external flash-attn requires fp16/bf16, got {q.dtype}"
            )
        if valid_token_mask is not None:
            raise RuntimeError(
                "external flash-attn experiment only supports the all-valid-mask "
                "specialization in this harness"
            )

        flash_attn_func = load_flash_attn_func()
        return flash_attn_func(
            q,
            k,
            v,
            dropout_p=0.0,
            softmax_scale=None,
            causal=causal,
        )

    def _sage_attention_hnsd(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        causal: bool,
    ) -> torch.Tensor:
        if q.device.type != "cuda":
            raise RuntimeError("SageAttention requires CUDA")
        if q.dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError(
                f"SageAttention requires fp16/bf16 inputs, got {q.dtype}"
            )
        if valid_token_mask is not None:
            raise RuntimeError(
                "SageAttention experiment only supports the all-valid-mask "
                "specialization in this harness"
            )

        sageattn = load_sage_attn_func()
        out = sageattn(
            q,
            k,
            v,
            tensor_layout="HND",
            is_causal=causal,
        )
        if isinstance(out, tuple):
            out = out[0]
        return out

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        if self.attention_impl == "flash_attn":
            if not self.packed_qkv:
                raise RuntimeError("flash-attn layout path requires packed QKV")
            if x.device.type != "cuda":
                raise RuntimeError("external flash-attn requires CUDA")
            if x.dtype not in (torch.float16, torch.bfloat16):
                raise RuntimeError(
                    f"external flash-attn requires fp16/bf16, got {x.dtype}"
                )
            if valid_token_mask is not None:
                raise RuntimeError(
                    "external flash-attn experiment only supports all-valid mask"
                )

            qkv = self.qkv_proj(x).view(
                batch, seq_len, 3, self.num_heads, self.head_dim
            )
            qkvpacked_func = load_flash_attn_qkvpacked_func()
            if qkvpacked_func is not None:
                context = qkvpacked_func(
                    qkv,
                    dropout_p=0.0,
                    softmax_scale=None,
                    causal=causal,
                )
            else:
                q, k, v = qkv.unbind(dim=2)
                context = self._flash_attention_bshd(
                    q, k, v, None, causal
                )

            # Packed FlashAttention consumes [B,S,3,H,D] directly and returns
            # [B,S,H,D], eliminating the SDPA B,H,S,D layout conversion.
            context = context.contiguous().view(batch, seq_len, self.d_model)

        else:
            q, k, v = self._project_qkv_hnsd(x)

            if self.attention_impl == "sage":
                context = self._sage_attention_hnsd(
                    q, k, v, valid_token_mask, causal
                )
            elif self.use_sdpa:
                context = self._sdpa_attention(
                    q, k, v, valid_token_mask, causal
                )
            else:
                context = self._manual_attention(
                    q, k, v, valid_token_mask, causal
                )

            context = (
                context.transpose(1, 2)
                .contiguous()
                .view(batch, seq_len, self.d_model)
            )

        output = self.out_proj(context)

        if valid_token_mask is not None:
            output = output.masked_fill(~valid_token_mask[..., None], 0)

        return output


class AblationTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        *,
        packed_qkv: bool,
        sdpa: bool,
        attention_impl: str = "native",
        gelu_approx: str = "none",
        sdpa_backend: str = "auto",
    ) -> None:
        super().__init__()

        self.gelu_approx = gelu_approx
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = AblationSelfAttention(
            d_model,
            num_heads,
            packed_qkv=packed_qkv,
            sdpa=sdpa,
            attention_impl=attention_impl,
            sdpa_backend=sdpa_backend,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, d_model)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        causal: bool,
    ) -> torch.Tensor:
        x = x + self.attention(
            self.norm1(x),
            valid_token_mask,
            causal,
        )

        x = x + self.ffn_out(
            F.gelu(
                self.ffn_in(self.norm2(x)),
                approximate=self.gelu_approx,
            )
        )

        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)

        return x


class AblationTransformer(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
        spec: VariantSpec,
        sdpa_backend: str = "auto",
    ) -> None:
        super().__init__()

        if spec.key == "A":
            raise ValueError("Variant A uses BaselineTransformer directly")

        self.config = config
        self.spec = spec
        effective_sdpa_backend = spec.sdpa_backend_override or sdpa_backend

        self.layers = nn.ModuleList(
            [
                AblationTransformerBlock(
                    config.d_model,
                    config.num_heads,
                    config.ffn_dim,
                    packed_qkv=spec.packed_qkv,
                    sdpa=spec.sdpa,
                    attention_impl=spec.attention_impl,
                    gelu_approx=spec.gelu_approx,
                    sdpa_backend=effective_sdpa_backend,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(config.d_model)
        self._assume_all_valid_mask = False

    def set_assume_all_valid_mask(self, enabled: bool) -> None:
        self._assume_all_valid_mask = bool(enabled)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        effective_mask = valid_token_mask

        if self.spec.skip_all_valid_mask and self._assume_all_valid_mask:
            effective_mask = None

        for layer in self.layers:
            x = layer(x, effective_mask, self.config.causal)

        x = self.final_norm(x)

        if effective_mask is not None:
            x = x.masked_fill(~effective_mask[..., None], 0)

        return x


# =============================================================================
# Selective mixed precision (CUDA)
# =============================================================================

class SelectivePrecisionSelfAttention(nn.Module):
    """Packed-QKV attention with explicit FP32/fast-dtype boundaries.

    QKV projection precision, attention-core precision, and output-projection
    precision are independently selectable. The enclosing Transformer state is
    always returned in the reference dtype before the residual addition.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        fast_dtype: torch.dtype,
        qkv_low_precision: bool,
        attention_low_precision: bool,
        out_proj_low_precision: bool,
        attention_impl: str = "native",
        sdpa_backend: str = "auto",
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if fast_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("selective precision requires float16 or bfloat16")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.fast_dtype = fast_dtype
        self.qkv_low_precision = qkv_low_precision
        self.attention_low_precision = attention_low_precision
        self.out_proj_low_precision = out_proj_low_precision
        self.attention_impl = attention_impl
        self.sdpa_backend = sdpa_backend

        # Construct in the reference dtype. configure_selected_modules() converts
        # only the projection weights explicitly selected for low precision.
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def configure_selected_modules(self) -> None:
        if self.qkv_low_precision:
            self.qkv_proj.to(dtype=self.fast_dtype)
        if self.out_proj_low_precision:
            self.out_proj.to(dtype=self.fast_dtype)

    def _sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        causal: bool,
    ) -> torch.Tensor:
        seq_len = q.shape[-2]
        attn_mask: Optional[torch.Tensor] = None
        use_is_causal = causal

        if valid_token_mask is not None:
            key_mask = valid_token_mask[:, None, None, :]
            if causal:
                # Competition all-valid runs set the mask to None and avoid this
                # O(S^2) tensor; this branch preserves padded-case correctness.
                causal_allowed = torch.ones(
                    (seq_len, seq_len),
                    device=q.device,
                    dtype=torch.bool,
                ).tril()
                attn_mask = key_mask & causal_allowed[None, None, :, :]
                use_is_causal = False
            else:
                attn_mask = key_mask

        with sdpa_backend_context(self.sdpa_backend, q.device):
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=use_is_causal,
            )

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        residual_dtype = x.dtype

        # X/V/W/P/Q lower QKV. U/Y and inactive Z layers leave QKV FP32.
        proj_input = (
            x.to(dtype=self.fast_dtype)
            if self.qkv_low_precision
            else x
        )
        qkv = self.qkv_proj(proj_input).view(
            batch,
            seq_len,
            3,
            self.num_heads,
            self.head_dim,
        )

        if self.attention_impl == "flash_attn":
            if x.device.type != "cuda":
                raise RuntimeError("external flash-attn requires CUDA")
            if valid_token_mask is not None:
                raise RuntimeError(
                    "external flash-attn selective path supports the all-valid "
                    "mask specialization only"
                )
            if qkv.dtype not in (torch.float16, torch.bfloat16):
                qkv = qkv.to(dtype=self.fast_dtype)

            qkvpacked_func = load_flash_attn_qkvpacked_func()
            if qkvpacked_func is not None:
                context_bshd = qkvpacked_func(
                    qkv,
                    dropout_p=0.0,
                    softmax_scale=None,
                    causal=causal,
                )
            else:
                q, k, v = qkv.unbind(dim=2)
                context_bshd = load_flash_attn_func()(
                    q,
                    k,
                    v,
                    dropout_p=0.0,
                    softmax_scale=None,
                    causal=causal,
                )
            context = context_bshd.contiguous().view(
                batch, seq_len, self.d_model
            )
        else:
            q, k, v = qkv.unbind(dim=2)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            if self.attention_low_precision and q.dtype != self.fast_dtype:
                q = q.to(dtype=self.fast_dtype)
                k = k.to(dtype=self.fast_dtype)
                v = v.to(dtype=self.fast_dtype)

            if self.attention_impl == "sage":
                if x.device.type != "cuda":
                    raise RuntimeError("SageAttention requires CUDA")
                if valid_token_mask is not None:
                    raise RuntimeError(
                        "SageAttention selective path supports the all-valid "
                        "mask specialization only"
                    )
                if q.dtype not in (torch.float16, torch.bfloat16):
                    q = q.to(dtype=self.fast_dtype)
                    k = k.to(dtype=self.fast_dtype)
                    v = v.to(dtype=self.fast_dtype)
                sageattn = load_sage_attn_func()
                context_hnsd = sageattn(
                    q,
                    k,
                    v,
                    tensor_layout="HND",
                    is_causal=causal,
                )
                if isinstance(context_hnsd, tuple):
                    context_hnsd = context_hnsd[0]
            else:
                context_hnsd = self._sdpa(
                    q, k, v, valid_token_mask, causal
                )

            context = (
                context_hnsd.transpose(1, 2)
                .contiguous()
                .view(batch, seq_len, self.d_model)
            )

        if self.out_proj_low_precision:
            # Y/V/W/P/Q lower out_proj, then return to FP32 before residual add.
            if context.dtype != self.fast_dtype:
                context = context.to(dtype=self.fast_dtype)
            output = self.out_proj(context)
            if output.dtype != residual_dtype:
                output = output.to(dtype=residual_dtype)
        else:
            # U/X and inactive Z layers keep out_proj in FP32.
            if context.dtype != residual_dtype:
                context = context.to(dtype=residual_dtype)
            output = self.out_proj(context)

        if valid_token_mask is not None:
            output = output.masked_fill(~valid_token_mask[..., None], 0)
        return output


class SelectivePrecisionTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        *,
        fast_dtype: torch.dtype,
        precision_mode: str,
        attention_impl: str,
        sdpa_backend: str,
    ) -> None:
        super().__init__()
        allowed_modes = {
            "none",
            "attention_core",
            "attention_qkv",
            "attention_out",
            "attention_projection",
            "attention_ffn",
        }
        if precision_mode not in allowed_modes:
            raise ValueError(f"unknown selective precision mode: {precision_mode}")

        self.fast_dtype = fast_dtype
        self.precision_mode = precision_mode

        attention_low = precision_mode != "none"
        qkv_low = precision_mode in {
            "attention_qkv",
            "attention_projection",
            "attention_ffn",
        }
        out_low = precision_mode in {
            "attention_out",
            "attention_projection",
            "attention_ffn",
        }
        self.ffn_low_precision = precision_mode == "attention_ffn"

        # LayerNorms and residual state remain reference dtype (normally FP32).
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = SelectivePrecisionSelfAttention(
            d_model,
            num_heads,
            fast_dtype=fast_dtype,
            qkv_low_precision=qkv_low,
            attention_low_precision=attention_low,
            out_proj_low_precision=out_low,
            attention_impl=attention_impl,
            sdpa_backend=sdpa_backend,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, d_model)

    def configure_selected_modules(self) -> None:
        self.attention.configure_selected_modules()
        if self.ffn_low_precision:
            self.ffn_in.to(dtype=self.fast_dtype)
            self.ffn_out.to(dtype=self.fast_dtype)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        causal: bool,
    ) -> torch.Tensor:
        attn_input = self.norm1(x)
        attn_output = self.attention(attn_input, valid_token_mask, causal)
        if attn_output.dtype != x.dtype:
            attn_output = attn_output.to(dtype=x.dtype)
        x = x + attn_output

        ffn_input = self.norm2(x)
        if self.ffn_low_precision:
            # GEMM1 low precision -> exact GELU FP32 -> GEMM2 low precision.
            hidden = self.ffn_in(ffn_input.to(dtype=self.fast_dtype))
            hidden = F.gelu(hidden.float(), approximate="none")
            ffn_output = self.ffn_out(hidden.to(dtype=self.fast_dtype))
            ffn_output = ffn_output.to(dtype=x.dtype)
        else:
            ffn_output = self.ffn_out(
                F.gelu(self.ffn_in(ffn_input), approximate="none")
            )

        x = x + ffn_output
        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x


class SelectivePrecisionTransformer(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
        *,
        fast_dtype: torch.dtype,
        precision_mode: str,
        attention_impl: str = "native",
        sdpa_backend: str = "auto",
        selective_layers: Optional[int] = None,
        layer_pattern: Optional[Sequence[bool]] = None,
        layer_precision_modes: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.fast_dtype = fast_dtype
        self.precision_mode = precision_mode
        self.selective_layers = selective_layers
        self.layer_pattern = None if layer_pattern is None else list(layer_pattern)
        self.layer_precision_modes = (
            None if layer_precision_modes is None else list(layer_precision_modes)
        )

        if selective_layers is not None and selective_layers <= 0:
            raise ValueError("selective_layers must be positive when provided")
        if self.layer_pattern is not None and len(self.layer_pattern) != config.num_layers:
            raise ValueError(
                f"layer_pattern length ({len(self.layer_pattern)}) must equal "
                f"num_layers ({config.num_layers})"
            )
        if (
            self.layer_precision_modes is not None
            and len(self.layer_precision_modes) != config.num_layers
        ):
            raise ValueError(
                f"layer_precision_modes length ({len(self.layer_precision_modes)}) "
                f"must equal num_layers ({config.num_layers})"
            )

        active_layers = (
            config.num_layers
            if selective_layers is None
            else min(selective_layers, config.num_layers)
        )

        blocks = []
        for layer_index in range(config.num_layers):
            if self.layer_precision_modes is not None:
                # Fully independent per-layer precision mode (e.g. a per-layer
                # search mixing "none" / attention_core / attention_projection),
                # takes precedence over everything else below.
                layer_mode = self.layer_precision_modes[layer_index]
                layer_attention_impl = attention_impl if layer_mode != "none" else "native"
            else:
                # An explicit per-layer boolean pattern (arbitrary subset, e.g.
                # from a brute-force search) takes precedence over the
                # contiguous first-N selective_layers behavior used by Z1-Z6.
                active = (
                    self.layer_pattern[layer_index]
                    if self.layer_pattern is not None
                    else layer_index < active_layers
                )
                layer_mode = precision_mode if active else "none"

                # Partial Z variants use H-style native SDPA in the untouched
                # layers. External P/Q variants currently activate all layers,
                # but this also keeps inactive behavior well-defined if layer
                # gating is added.
                layer_attention_impl = attention_impl if active else "native"

            blocks.append(
                SelectivePrecisionTransformerBlock(
                    config.d_model,
                    config.num_heads,
                    config.ffn_dim,
                    fast_dtype=fast_dtype,
                    precision_mode=layer_mode,
                    attention_impl=layer_attention_impl,
                    sdpa_backend=sdpa_backend,
                )
            )

        self.layers = nn.ModuleList(blocks)
        self.final_norm = nn.LayerNorm(config.d_model)
        self._assume_all_valid_mask = False

    def set_assume_all_valid_mask(self, enabled: bool) -> None:
        self._assume_all_valid_mask = bool(enabled)

    def configure_selected_modules(self) -> None:
        for layer in self.layers:
            layer.configure_selected_modules()

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        effective_mask = valid_token_mask
        if self._assume_all_valid_mask:
            effective_mask = None

        state_dtype = x.dtype
        for layer in self.layers:
            x = layer(x, effective_mask, self.config.causal)
            if x.dtype != state_dtype:
                x = x.to(dtype=state_dtype)

        x = self.final_norm(x)
        if effective_mask is not None:
            x = x.masked_fill(~effective_mask[..., None], 0)
        return x


def copy_baseline_weights_selective(
    baseline: BaselineTransformer,
    candidate: SelectivePrecisionTransformer,
) -> None:
    if len(baseline.layers) != len(candidate.layers):
        raise ValueError("baseline/candidate layer count mismatch")

    with torch.no_grad():
        for src_layer, dst_layer in zip(baseline.layers, candidate.layers):
            dst_layer.norm1.load_state_dict(
                copy.deepcopy(src_layer.norm1.state_dict())
            )
            dst_layer.norm2.load_state_dict(
                copy.deepcopy(src_layer.norm2.state_dict())
            )
            dst_layer.ffn_in.load_state_dict(
                copy.deepcopy(src_layer.ffn_in.state_dict())
            )
            dst_layer.ffn_out.load_state_dict(
                copy.deepcopy(src_layer.ffn_out.state_dict())
            )
            dst_layer.attention.out_proj.load_state_dict(
                copy.deepcopy(src_layer.attention.out_proj.state_dict())
            )
            dst_layer.attention.qkv_proj.weight.copy_(torch.cat((
                src_layer.attention.q_proj.weight,
                src_layer.attention.k_proj.weight,
                src_layer.attention.v_proj.weight,
            ), dim=0))
            dst_layer.attention.qkv_proj.bias.copy_(torch.cat((
                src_layer.attention.q_proj.bias,
                src_layer.attention.k_proj.bias,
                src_layer.attention.v_proj.bias,
            ), dim=0))

        candidate.final_norm.load_state_dict(
            copy.deepcopy(baseline.final_norm.state_dict())
        )


# =============================================================================
# Weight copy
# =============================================================================

def copy_baseline_weights(
    baseline: BaselineTransformer,
    candidate: AblationTransformer,
) -> None:
    if len(baseline.layers) != len(candidate.layers):
        raise ValueError("baseline/candidate layer count mismatch")

    with torch.no_grad():
        for src_layer, dst_layer in zip(
            baseline.layers,
            candidate.layers,
        ):
            dst_layer.norm1.load_state_dict(
                copy.deepcopy(src_layer.norm1.state_dict())
            )
            dst_layer.norm2.load_state_dict(
                copy.deepcopy(src_layer.norm2.state_dict())
            )
            dst_layer.ffn_in.load_state_dict(
                copy.deepcopy(src_layer.ffn_in.state_dict())
            )
            dst_layer.ffn_out.load_state_dict(
                copy.deepcopy(src_layer.ffn_out.state_dict())
            )
            dst_layer.attention.out_proj.load_state_dict(
                copy.deepcopy(src_layer.attention.out_proj.state_dict())
            )

            if candidate.spec.packed_qkv:
                dst_layer.attention.qkv_proj.weight.copy_(
                    torch.cat(
                        (
                            src_layer.attention.q_proj.weight,
                            src_layer.attention.k_proj.weight,
                            src_layer.attention.v_proj.weight,
                        ),
                        dim=0,
                    )
                )
                dst_layer.attention.qkv_proj.bias.copy_(
                    torch.cat(
                        (
                            src_layer.attention.q_proj.bias,
                            src_layer.attention.k_proj.bias,
                            src_layer.attention.v_proj.bias,
                        ),
                        dim=0,
                    )
                )
            else:
                dst_layer.attention.q_proj.load_state_dict(
                    copy.deepcopy(src_layer.attention.q_proj.state_dict())
                )
                dst_layer.attention.k_proj.load_state_dict(
                    copy.deepcopy(src_layer.attention.k_proj.state_dict())
                )
                dst_layer.attention.v_proj.load_state_dict(
                    copy.deepcopy(src_layer.attention.v_proj.state_dict())
                )

        candidate.final_norm.load_state_dict(
            copy.deepcopy(baseline.final_norm.state_dict())
        )


# =============================================================================
# Data and correctness
# =============================================================================

def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(device_arg)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False"
        )

    if device.type == "mps":
        if (
            not hasattr(torch.backends, "mps")
            or not torch.backends.mps.is_available()
        ):
            raise RuntimeError(
                "MPS was requested, but torch.backends.mps.is_available() is False"
            )

    return device


def resolve_dtype(dtype_name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]


def generate_random_case(
    config: TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    padding_ratio: float,
    input_scale: float,
    pattern: str = "normal",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate benchmark inputs.

    pattern="stress" cases are intentionally simple and deterministic:
      normal  : N(0,1)
      tiny    : 0.01 * N(0,1)
      large   : 3.0 * N(0,1)
      outlier : N(0,1) plus 0.1% N(0,10) outliers

    The outlier distribution mirrors the style of numerical stress test used
    by FlashAttention-3 when evaluating low-precision attention.
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    x = torch.randn(
        config.batch_size,
        config.seq_len,
        config.d_model,
        generator=generator,
        device=device,
        dtype=dtype,
    )

    if pattern == "normal":
        x = x * input_scale
    elif pattern == "tiny":
        x = x * (0.01 * input_scale)
    elif pattern == "large":
        x = x * (3.0 * input_scale)
    elif pattern == "outlier":
        outlier_values = torch.randn(
            x.shape,
            generator=generator,
            device=device,
            dtype=dtype,
        ) * 10.0
        outlier_mask = torch.rand(
            x.shape,
            generator=generator,
            device=device,
        ) < 0.001
        x = (x + outlier_values * outlier_mask.to(dtype=dtype)) * input_scale
    else:
        raise ValueError(f"unknown input pattern: {pattern}")

    if padding_ratio <= 0:
        valid_token_mask = torch.ones(
            config.batch_size,
            config.seq_len,
            device=device,
            dtype=torch.bool,
        )
        return x, valid_token_mask

    min_valid = max(
        1,
        int(round(config.seq_len * (1.0 - padding_ratio))),
    )

    lengths = torch.randint(
        low=min_valid,
        high=config.seq_len + 1,
        size=(config.batch_size,),
        generator=generator,
        device=device,
    )

    positions = torch.arange(
        config.seq_len,
        device=device,
    )[None, :]

    valid_token_mask = positions < lengths[:, None]
    x = x.masked_fill(~valid_token_mask[..., None], 0)

    return x, valid_token_mask


@dataclass
class AccuracySummary:
    passed: bool = True
    total_elements: int = 0
    failed_elements: int = 0
    max_abs_error: float = 0.0
    max_relative_error: float = 0.0
    mean_abs_error_sum: float = 0.0
    trials: int = 0
    error: str = ""

    @property
    def mean_abs_error(self) -> float:
        if self.trials == 0:
            return 0.0
        return self.mean_abs_error_sum / self.trials


def update_accuracy_summary(
    summary: AccuracySummary,
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> None:
    if reference.shape != candidate.shape:
        raise AssertionError(
            f"shape mismatch: reference={tuple(reference.shape)} "
            f"candidate={tuple(candidate.shape)}"
        )

    ref = reference.detach().float()
    opt = candidate.detach().float()

    finite_mask = torch.isfinite(ref) & torch.isfinite(opt)
    abs_error = (opt - ref).abs()

    abs_ok = abs_error <= atol
    rel_ok = abs_error <= rtol * ref.abs()
    passed_mask = finite_mask & (abs_ok | rel_ok)

    failed = int((~passed_mask).sum().item())

    denominator = ref.abs().clamp_min(1e-12)
    relative_error = abs_error / denominator

    summary.passed = summary.passed and (failed == 0)
    summary.total_elements += reference.numel()
    summary.failed_elements += failed
    summary.max_abs_error = max(
        summary.max_abs_error,
        float(abs_error.max().item()),
    )
    summary.max_relative_error = max(
        summary.max_relative_error,
        float(relative_error.max().item()),
    )
    summary.mean_abs_error_sum += float(abs_error.mean().item())
    summary.trials += 1


def run_accuracy_suite(
    models: Dict[str, nn.Module],
    variant_keys: Sequence[str],
    config: TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
    trials: int,
    seed: int,
    padding_ratio: float,
    input_scale: float,
    rtol: float,
    atol: float,
    accuracy_profile: str = "basic",
) -> Dict[str, AccuracySummary]:
    print("\n=== Accuracy ablation ===")
    print(
        f"criterion: abs_error <= {atol:g} OR "
        f"relative_error <= {rtol:.2%}"
    )

    if accuracy_profile == "basic":
        patterns = ("normal",)
    elif accuracy_profile == "stress":
        patterns = ("normal", "tiny", "large", "outlier")
    else:
        raise ValueError(f"unknown accuracy profile: {accuracy_profile}")

    print(
        f"accuracy_profile={accuracy_profile}, "
        f"patterns={','.join(patterns)}, trials_per_pattern={trials}"
    )

    summaries: Dict[str, AccuracySummary] = {
        key: AccuracySummary()
        for key in variant_keys
        if key != "A"
    }

    if not summaries:
        return {}

    with torch.inference_mode():
        for pattern_index, pattern in enumerate(patterns):
            for trial in range(trials):
                x, valid_mask = generate_random_case(
                    config=config,
                    device=device,
                    dtype=dtype,
                    seed=seed + pattern_index * 100_000 + trial,
                    padding_ratio=padding_ratio,
                    input_scale=input_scale,
                    pattern=pattern,
                )

                reference = models["A"](x, valid_mask)

                for key in variant_keys:
                    if key == "A":
                        continue
                    summary = summaries[key]
                    if summary.error:
                        continue

                    try:
                        candidate = models[key](x, valid_mask)
                        update_accuracy_summary(
                            summary,
                            reference,
                            candidate,
                            rtol=rtol,
                            atol=atol,
                        )
                    except Exception as exc:
                        summary.passed = False
                        summary.error = (
                            f"{type(exc).__name__}: {str(exc).replace(chr(10), ' ')}"
                        )[:500]

    print(
        f"{'Var':<4} {'Name':<33} {'Status':<7} "
        f"{'Max abs':>12} {'Max rel':>12} {'Failed':>14}"
    )
    print("-" * 94)

    for key in variant_keys:
        if key == "A":
            print(
                f"{'A':<4} {'baseline':<33} {'REF':<7} "
                f"{0.0:>12.6g} {0.0:>12.6g} {'0':>14}"
            )
            continue

        s = summaries[key]
        if s.error:
            status = "ERROR"
        else:
            status = "PASS" if s.passed else "FAIL"

        print(
            f"{key:<4} {VARIANT_SPECS[key].name:<33} {status:<7} "
            f"{s.max_abs_error:>12.6g} "
            f"{s.max_relative_error:>12.6g} "
            f"{s.failed_elements:>14d}"
        )
        if s.error:
            print(f"     error: {s.error}")

    return summaries


# =============================================================================
# Timing
# =============================================================================

@dataclass
class TimingResult:
    samples_ms: List[float]

    @property
    def mean_ms(self) -> float:
        return statistics.fmean(self.samples_ms)

    @property
    def median_ms(self) -> float:
        return statistics.median(self.samples_ms)

    @property
    def p90_ms(self) -> float:
        return percentile(self.samples_ms, 0.90)

    @property
    def min_ms(self) -> float:
        return min(self.samples_ms)


def percentile(values: List[float], q: float) -> float:
    if not values:
        raise ValueError("values must not be empty")

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    weight = position - lower

    return (
        ordered[lower] * (1.0 - weight)
        + ordered[upper] * weight
    )


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def warmup_model(
    model: nn.Module,
    x: torch.Tensor,
    valid_mask: torch.Tensor,
    iterations: int,
    device: torch.device,
) -> None:
    with torch.inference_mode():
        for _ in range(iterations):
            model(x, valid_mask)

    synchronize_device(device)


def benchmark_once(
    model: nn.Module,
    x: torch.Tensor,
    valid_mask: torch.Tensor,
    iterations: int,
    device: torch.device,
) -> List[float]:
    samples_ms: List[float] = []

    with torch.inference_mode():
        if device.type == "cuda":
            starts = [
                torch.cuda.Event(enable_timing=True)
                for _ in range(iterations)
            ]
            ends = [
                torch.cuda.Event(enable_timing=True)
                for _ in range(iterations)
            ]

            torch.cuda.synchronize(device)

            for index in range(iterations):
                starts[index].record()
                model(x, valid_mask)
                ends[index].record()

            torch.cuda.synchronize(device)

            samples_ms.extend(
                start.elapsed_time(end)
                for start, end in zip(starts, ends)
            )

        elif device.type == "mps":
            # MPS is asynchronous, so synchronize around every measured call.
            for _ in range(iterations):
                torch.mps.synchronize()
                start = time.perf_counter_ns()
                model(x, valid_mask)
                torch.mps.synchronize()
                end = time.perf_counter_ns()
                samples_ms.append((end - start) / 1e6)

        else:
            for _ in range(iterations):
                start = time.perf_counter_ns()
                model(x, valid_mask)
                end = time.perf_counter_ns()
                samples_ms.append((end - start) / 1e6)

    return samples_ms


def rotated_order(
    keys: Sequence[str],
    round_index: int,
) -> List[str]:
    """
    Rotate and reverse measurement order to reduce systematic order bias.

    Round 0: A B C D ...
    Round 1: ... D C B A
    Round 2: C D ... A B
    """
    keys = list(keys)

    if not keys:
        return []

    shift = (round_index // 2) % len(keys)
    order = keys[shift:] + keys[:shift]

    if round_index % 2 == 1:
        order.reverse()

    return order


def benchmark_variants(
    models: Dict[str, nn.Module],
    variant_keys: Sequence[str],
    config: TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    padding_ratio: float,
    input_scale: float,
    warmup: int,
    repeats: int,
    rounds: int,
    block_size: int = 5,
) -> Dict[str, TimingResult]:
    print("\n=== Performance ablation ===")
    print("all variants use the same fixed benchmark input")

    if device.type == "cuda":
        print("CUDA: torch.cuda.Event timing")
    elif device.type == "mps":
        print("MPS: synchronized wall-clock timing")
    else:
        print("CPU: wall-clock timing")

    x, valid_mask = generate_random_case(
        config=config,
        device=device,
        dtype=dtype,
        seed=seed + 100000,
        padding_ratio=padding_ratio,
        input_scale=input_scale,
        pattern="normal",
    )

    print("warming up:", ", ".join(variant_keys))
    for key in variant_keys:
        warmup_model(
            models[key],
            x,
            valid_mask,
            warmup,
            device,
        )

    samples: Dict[str, List[float]] = {
        key: []
        for key in variant_keys
    }

    block_size = max(1, min(block_size, repeats))
    blocks = math.ceil(repeats / block_size)
    print(
        f"timing design: interleaved blocks of <= {block_size} calls "
        f"({blocks} blocks/round)"
    )

    for round_index in range(rounds):
        print(f"round {round_index + 1}/{rounds}")
        remaining = {key: repeats for key in variant_keys}

        for block_index in range(blocks):
            order = rotated_order(
                variant_keys,
                round_index * blocks + block_index,
            )
            for key in order:
                count = min(block_size, remaining[key])
                if count <= 0:
                    continue
                samples[key].extend(
                    benchmark_once(
                        models[key],
                        x,
                        valid_mask,
                        count,
                        device,
                    )
                )
                remaining[key] -= count

    return {
        key: TimingResult(samples[key])
        for key in variant_keys
    }


# =============================================================================
# Reporting
# =============================================================================

def print_variant_definition_table(
    variant_keys: Sequence[str],
    mask_skip_effective: bool,
    fast_dtype: torch.dtype,
) -> None:
    print("\n=== Variants ===")
    print(
        f"{'Var':<4} {'Name':<34} {'QKV':<7} {'Attention':<18} "
        f"{'Mask':<8} {'GELU':<7} {'Fast dtype':<12} {'Selective':<30} {'Compile':<8}"
    )
    print("-" * 142)

    for key in variant_keys:
        spec = VARIANT_SPECS[key]
        if key == "A":
            mask_text = "no"
        elif spec.skip_all_valid_mask:
            mask_text = "yes" if mask_skip_effective else "inactive"
        else:
            mask_text = "no"

        fast_text = (
            str(fast_dtype).replace("torch.", "")
            if variant_uses_fast_dtype(spec) else "-"
        )
        print(
            f"{key:<4} {spec.name:<34} "
            f"{('packed' if spec.packed_qkv else 'sep'):<7} "
            f"{variant_attention_label(spec):<18} "
            f"{mask_text:<8} "
            f"{spec.gelu_approx:<7} "
            f"{fast_text:<12} "
            f"{variant_selective_label(spec):<30} "
            f"{('yes' if spec.compile_candidate else 'no'):<8}"
        )


def print_timing_table(
    timings: Dict[str, TimingResult],
    variant_keys: Sequence[str],
    config: TransformerConfig,
    accuracy: Dict[str, AccuracySummary],
    reference_key: str = "A",
) -> None:
    if reference_key not in timings:
        raise ValueError(f"reference variant {reference_key} is not timed")

    reference_ms = timings[reference_key].median_ms
    tokens_per_call = config.batch_size * config.seq_len

    print("\n=== Ablation summary ===")
    print(f"timing reference={reference_key} ({VARIANT_SPECS[reference_key].name})")
    print(
        f"{'Var':<4} {'Name':<34} {'Accuracy':<9} "
        f"{'Median ms':>11} {'Speedup':>9} {'Latency Δ':>10} {'Token/s':>13}"
    )
    print("-" * 103)

    for key in variant_keys:
        result = timings[key]
        speedup = reference_ms / result.median_ms
        latency_delta = ((result.median_ms / reference_ms) - 1.0) * 100.0
        token_s = tokens_per_call * 1000.0 / result.median_ms

        if key == reference_key:
            accuracy_text = "REF"
        elif key == "A":
            accuracy_text = "PASS"
        else:
            summary = accuracy.get(key, AccuracySummary())
            if summary.error:
                accuracy_text = "ERROR"
            else:
                accuracy_text = "PASS" if summary.passed else "FAIL"

        print(
            f"{key:<4} {VARIANT_SPECS[key].name:<34} {accuracy_text:<9} "
            f"{result.median_ms:>11.4f} {speedup:>8.3f}x "
            f"{latency_delta:>+9.2f}% {token_s:>13.2f}"
        )

    print("\nLatency distribution:")
    print(f"{'Var':<4} {'Mean ms':>11} {'P90 ms':>11} {'Min ms':>11}")
    print("-" * 44)
    for key in variant_keys:
        result = timings[key]
        print(
            f"{key:<4} {result.mean_ms:>11.4f} "
            f"{result.p90_ms:>11.4f} {result.min_ms:>11.4f}"
        )


def pairwise_speedup(
    timings: Dict[str, TimingResult],
    before: str,
    after: str,
) -> Optional[float]:
    if before not in timings or after not in timings:
        return None
    return timings[before].median_ms / timings[after].median_ms


def geometric_mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        raise ValueError("cannot compute geometric mean of an empty set")
    return math.exp(statistics.fmean(math.log(v) for v in vals))


def print_effect_analysis(
    timings: Dict[str, TimingResult],
    reference_key: str = "A",
) -> None:
    print("\n=== Optimization effect analysis ===")

    comparisons = {
        "Packed QKV": [
            ("A", "B"), ("C", "E"), ("D", "F"), ("G", "H"),
        ],
        "SDPA": [
            ("A", "C"), ("B", "E"), ("D", "G"), ("F", "H"),
        ],
        "Mask skip": [
            ("A", "D"), ("B", "F"), ("C", "G"), ("E", "H"),
        ],
    }

    any_effect = False
    for factor, pairs in comparisons.items():
        ratios: List[float] = []
        labels: List[str] = []
        for before, after in pairs:
            ratio = pairwise_speedup(timings, before, after)
            if ratio is not None:
                ratios.append(ratio)
                labels.append(f"{before}->{after} {ratio:.3f}x")
        if not ratios:
            continue
        any_effect = True
        print(
            f"{factor:<12}: matched geometric-mean effect = "
            f"{geometric_mean(ratios):.3f}x"
        )
        print(" " * 14 + " | ".join(labels))

    if not any_effect:
        print("Need more matched A-H variants to estimate factorial effects.")

    if "H" in timings:
        h_ms = timings["H"].median_ms
        extras = [
            key for key in timings
            if key not in CORE_VARIANTS and key != reference_key
        ]
        if extras:
            print("\n=== New fast-path effects vs H ===")
            for key in extras:
                ratio = h_ms / timings[key].median_ms
                delta = (timings[key].median_ms / h_ms - 1.0) * 100.0
                print(
                    f"H->{key} {ratio:.3f}x  "
                    f"({delta:+.2f}% latency)  {VARIANT_SPECS[key].name}"
                )

    precision_keys = [
        key for key in ("U", "X", "Y", "Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "V", "W")
        if key in timings
    ]
    if "H" in timings and precision_keys:
        print("\n=== Precision-boundary search vs H ===")
        h_ms = timings["H"].median_ms
        for key in precision_keys:
            ratio = h_ms / timings[key].median_ms
            delta = (timings[key].median_ms / h_ms - 1.0) * 100.0
            print(
                f"H->{key:<2} {ratio:.3f}x  ({delta:+.2f}% latency)  "
                f"{VARIANT_SPECS[key].name}"
            )

    if "A" in timings and "H" in timings:
        print(
            f"\nFull A->H speedup: "
            f"{timings['A'].median_ms / timings['H'].median_ms:.3f}x"
        )
    elif reference_key in timings and "H" in timings:
        print(
            f"\n{reference_key}->H full-size speedup: "
            f"{timings[reference_key].median_ms / timings['H'].median_ms:.3f}x"
        )

    if reference_key in timings:
        valid = [
            key for key in timings
            if timings[key].median_ms > 0
        ]
        best = min(valid, key=lambda key: timings[key].median_ms)
        print(
            f"Best timed variant: {best} ({VARIANT_SPECS[best].name}), "
            f"{timings[reference_key].median_ms / timings[best].median_ms:.3f}x "
            f"vs {reference_key}"
        )

    print(
        "\nNote: low-precision / approximate / external variants are only valid "
        "when the accuracy table says PASS."
    )


def save_csv_results(
    path: str,
    timings: Dict[str, TimingResult],
    variant_keys: Sequence[str],
    config: TransformerConfig,
    accuracy: Dict[str, AccuracySummary],
    device: torch.device,
    dtype: torch.dtype,
    padding_ratio: float,
    sdpa_backend: str,
    fast_dtype: torch.dtype,
    reference_key: str = "A",
    shape_id: Optional[int] = None,
    append: bool = False,
) -> None:
    if reference_key not in timings:
        raise ValueError(f"reference variant {reference_key} is not timed")
    reference_ms = timings[reference_key].median_ms
    tokens_per_call = config.batch_size * config.seq_len

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    should_write_header = (not append) or (not output_path.exists())
    mode = "a" if append else "w"

    fields = [
        "shape_id", "variant", "name", "packed_qkv", "attention_impl",
        "sdpa", "sdpa_backend", "mask_skip", "gelu_approx",
        "fast_dtype", "selective_precision", "selective_layers", "compile_candidate", "accuracy", "error",
        "failed_elements", "max_abs_error", "max_relative_error",
        "median_ms", "mean_ms", "p90_ms", "min_ms",
        "reference_variant", "speedup_vs_reference", "speedup_vs_A",
        "latency_delta_percent", "tokens_per_second", "device", "dtype",
        "batch_size", "seq_len", "d_model", "heads", "ffn_dim", "layers",
        "causal", "padding_ratio",
    ]

    with output_path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if should_write_header:
            writer.writeheader()

        for key in variant_keys:
            spec = VARIANT_SPECS[key]
            result = timings[key]
            if key == "A":
                accuracy_text = "REF"
                summary = AccuracySummary()
            else:
                summary = accuracy.get(key, AccuracySummary())
                if summary.error:
                    accuracy_text = "ERROR"
                else:
                    accuracy_text = "PASS" if summary.passed else "FAIL"

            effective_backend = spec.sdpa_backend_override or sdpa_backend
            writer.writerow(
                {
                    "shape_id": shape_id if shape_id is not None else "",
                    "variant": key,
                    "name": spec.name,
                    "packed_qkv": spec.packed_qkv,
                    "attention_impl": variant_attention_label(spec),
                    "sdpa": spec.sdpa,
                    "sdpa_backend": effective_backend if spec.sdpa else "",
                    "mask_skip": spec.skip_all_valid_mask,
                    "gelu_approx": spec.gelu_approx,
                    "fast_dtype": (
                        str(fast_dtype) if variant_uses_fast_dtype(spec) else ""
                    ),
                    "selective_precision": spec.selective_precision,
                    "selective_layers": (
                        spec.selective_layers if spec.selective_layers is not None else ""
                    ),
                    "compile_candidate": spec.compile_candidate,
                    "accuracy": accuracy_text,
                    "error": summary.error,
                    "failed_elements": summary.failed_elements,
                    "max_abs_error": summary.max_abs_error,
                    "max_relative_error": summary.max_relative_error,
                    "median_ms": result.median_ms,
                    "mean_ms": result.mean_ms,
                    "p90_ms": result.p90_ms,
                    "min_ms": result.min_ms,
                    "reference_variant": reference_key,
                    "speedup_vs_reference": (
                        reference_ms / result.median_ms
                    ),
                    "speedup_vs_A": (
                        timings["A"].median_ms / result.median_ms
                        if "A" in timings else ""
                    ),
                    "latency_delta_percent": (
                        (result.median_ms / reference_ms) - 1.0
                    ) * 100.0,
                    "tokens_per_second": (
                        tokens_per_call * 1000.0 / result.median_ms
                    ),
                    "device": str(device),
                    "dtype": str(dtype),
                    "batch_size": config.batch_size,
                    "seq_len": config.seq_len,
                    "d_model": config.d_model,
                    "heads": config.num_heads,
                    "ffn_dim": config.ffn_dim,
                    "layers": config.num_layers,
                    "causal": config.causal,
                    "padding_ratio": padding_ratio,
                }
            )

    print(f"\nCSV written to: {output_path}")


# =============================================================================
# Compilation
# =============================================================================

def maybe_compile(
    model: nn.Module,
    enabled: bool,
    mode: str,
    fullgraph: bool = False,
) -> nn.Module:
    if not enabled:
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("this PyTorch build does not provide torch.compile")

    return torch.compile(
        model,
        mode=mode,
        fullgraph=fullgraph,
        dynamic=False,
    )


# =============================================================================
# CLI
# =============================================================================

def parse_variant_keys(raw: str) -> List[str]:
    raw = raw.strip().upper()

    aliases = {
        "ALL": CORE_VARIANTS,
        "CORE": CORE_VARIANTS,
        "FAST": FAST_VARIANTS,
        "SELECTIVE": SELECTIVE_VARIANTS,
        "PRECISION-SEARCH": PRECISION_SEARCH_VARIANTS,
        "PRECISION_SEARCH": PRECISION_SEARCH_VARIANTS,
        "CUDA-FAST": CUDA_FAST_VARIANTS,
        "CUDA_FAST": CUDA_FAST_VARIANTS,
        "EVERYTHING": list(VARIANT_SPECS.keys()),
    }
    if raw in aliases:
        return list(aliases[raw])

    keys = [item.strip() for item in raw.split(",") if item.strip()]

    if "A" not in keys:
        # A stays the exact semantic reference for normal-size correctness.
        keys.insert(0, "A")

    unknown = [key for key in keys if key not in VARIANT_SPECS]
    if unknown:
        raise ValueError(
            "unknown variants: "
            + ", ".join(unknown)
            + ". Valid variants are "
            + ",".join(VARIANT_SPECS)
            + "; aliases: all/core, fast, cuda-fast, everything"
        )

    deduped: List[str] = []
    for key in keys:
        if key not in deduped:
            deduped.append(key)
    return deduped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transformer optimization / precision-search benchmark V6"
    )

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument(
        "--causal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use causal attention. Default: enabled for competition parity.",
    )

    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, mps, cuda, cuda:0, ...",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
        help="Reference/model dtype for A-H.",
    )
    parser.add_argument(
        "--fast-dtype",
        choices=("auto", "float16", "bfloat16"),
        default="auto",
        help=(
            "Compute dtype for J/K/N/O and selective U/V/W/P/Q. auto uses fp16 "
            "when the reference is fp32 on MPS/CUDA; pass bfloat16 to compare."
        ),
    )

    parser.add_argument("--padding-ratio", type=float, default=0.0)
    parser.add_argument("--input-scale", type=float, default=1.0)

    parser.add_argument(
        "--accuracy-profile",
        choices=("basic", "stress"),
        default="basic",
        help=(
            "basic uses normal random inputs. stress additionally checks tiny, "
            "large, and sparse-outlier distributions."
        ),
    )
    parser.add_argument("--accuracy-trials", type=int, default=5)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--atol", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--benchmark-rounds", type=int, default=3)
    parser.add_argument(
        "--timing-block-size",
        type=int,
        default=5,
        help=(
            "Interleave variants after this many calls instead of running all "
            "repeats for one variant consecutively. Smaller reduces thermal/order bias."
        ),
    )

    parser.add_argument(
        "--variants",
        default="all",
        help=(
            "A-H preserve the old factorial. Aliases: all/core=A-H; "
            "fast=A,H,I-O; selective=A,H,U,V,W; cuda-fast adds selective plus "
            "external/forced attention P-T; everything=A-W."
        ),
    )
    parser.add_argument(
        "--benchmark-on-failure",
        action="store_true",
        help=(
            "Also time candidates that numerically fail. Variants that raise "
            "runtime errors are always excluded from timing."
        ),
    )

    # Old control: compile every selected model including A.
    parser.add_argument(
        "--compile",
        action="store_true",
        help=(
            "Compile every selected model, including A. Keep disabled when "
            "measuring candidate-only compiler variants L-O."
        ),
    )
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="reduce-overhead",
    )
    parser.add_argument(
        "--compile-fullgraph",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use fullgraph=True for candidate compile variants L-O.",
    )

    parser.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "medium"),
        default="high",
    )
    parser.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable TF32 on CUDA.",
    )
    parser.add_argument(
        "--sdpa-backend",
        choices=("auto", "flash", "efficient", "cudnn", "math"),
        default="auto",
        help=(
            "Default backend for ordinary SDPA variants. R/S/T override this "
            "with flash/cuDNN/efficient respectively."
        ),
    )
    parser.add_argument(
        "--dynamo-cache-size-limit",
        type=int,
        default=128,
        help=(
            "torch._dynamo's per-function recompilation cache limit. Each new "
            "shape in a --competition-shapes sweep is a new set of guards for "
            "the same compiled forward(), so this must exceed "
            "shapes * compiled-variants or later shapes silently fall back to "
            "eager execution once the default limit of 8 is exhausted."
        ),
    )

    parser.add_argument(
        "--competition-shapes",
        action="store_true",
        help="Sweep official challenge shapes instead of the single CLI shape.",
    )
    parser.add_argument(
        "--shape-ids",
        default="1-13",
        help=(
            "Competition shape IDs, e.g. '1-13', '1,2,7-9', or 'all'. "
            "Shape 14 is supported by long-safe mode."
        ),
    )
    parser.add_argument(
        "--dispatch-json",
        default=None,
        help=(
            "During a competition sweep, write the fastest correctness-passing "
            "variant for each shape to this JSON file."
        ),
    )
    parser.add_argument(
        "--max-manual-attention-gib",
        type=float,
        default=12.0,
        help=(
            "Safety guard for explicit score+softmax buffers. Set <=0 to disable."
        ),
    )
    parser.add_argument(
        "--force-manual-attention",
        action="store_true",
        help="Override the explicit-attention memory guard (OOM risk).",
    )
    parser.add_argument(
        "--long-seq-accuracy-prefix",
        type=int,
        default=512,
        help=(
            "When full explicit attention is unsafe, validate selected variants "
            "against exact A on this prefix before full-size timing."
        ),
    )
    parser.add_argument(
        "--auto-long-seq-timing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--long-seq-threshold", type=int, default=8192)
    parser.add_argument("--long-seq-warmup", type=int, default=1)
    parser.add_argument("--long-seq-repeats", type=int, default=3)
    parser.add_argument("--long-seq-rounds", type=int, default=1)

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast development pass: accuracy=1, warmup=2, repeats=10, rounds=1.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional CSV output path.",
    )
    parser.add_argument(
        "--strict-exit",
        action="store_true",
        help=(
            "Return a non-zero exit code when any experimental candidate fails "
            "accuracy or is unavailable. Default keeps the sweep running."
        ),
    )

    return parser.parse_args()


def validate_args(
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    if not 0.0 <= args.padding_ratio < 1.0:
        raise ValueError("padding_ratio must be in [0, 1)")
    if args.input_scale <= 0:
        raise ValueError("input_scale must be positive")
    if args.accuracy_trials <= 0:
        raise ValueError("accuracy_trials must be positive")
    if args.rtol < 0 or args.atol < 0:
        raise ValueError("rtol and atol must be non-negative")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if args.repeats <= 0 or args.benchmark_rounds <= 0:
        raise ValueError("repeats and benchmark_rounds must be positive")
    if args.timing_block_size <= 0:
        raise ValueError("timing_block_size must be positive")

    if args.sdpa_backend != "auto" and device.type != "cuda":
        raise ValueError(
            "forcing --sdpa-backend requires CUDA; R/S/T are also CUDA-only "
            "and will report ERROR when explicitly selected elsewhere."
        )

    if args.max_manual_attention_gib < 0:
        raise ValueError("max_manual_attention_gib must be >= 0")
    if args.long_seq_accuracy_prefix <= 0:
        raise ValueError("long_seq_accuracy_prefix must be positive")
    if args.long_seq_threshold <= 0:
        raise ValueError("long_seq_threshold must be positive")
    if args.long_seq_warmup < 0:
        raise ValueError("long_seq_warmup must be non-negative")
    if args.long_seq_repeats <= 0 or args.long_seq_rounds <= 0:
        raise ValueError("long_seq_repeats and long_seq_rounds must be positive")

    if device.type == "cpu" and dtype == torch.float16:
        print("[warning] float16 CPU kernels may be unsupported or slow")


# =============================================================================
# Main
# =============================================================================

@dataclass
class ShapeRunResult:
    shape_id: Optional[int]
    config: TransformerConfig
    timings: Dict[str, TimingResult]
    accuracy: Dict[str, AccuracySummary]
    reference_key: str = "A"


def variant_precheck_error(
    spec: VariantSpec,
    device: torch.device,
    base_dtype: torch.dtype,
    fast_dtype: torch.dtype,
) -> Optional[str]:
    if spec.sdpa_backend_override and device.type != "cuda":
        return (
            f"{spec.sdpa_backend_override} SDPA backend experiment requires CUDA"
        )

    if spec.selective_precision != "none":
        if device.type != "cuda":
            return "selective mixed precision variants require CUDA"
        if base_dtype != torch.float32:
            return (
                "selective mixed precision expects --dtype float32 so norms, "
                "residuals, GELU, and final norm remain FP32"
            )
        if fast_dtype not in (torch.float16, torch.bfloat16):
            return "selective mixed precision requires fp16 or bf16 fast dtype"

    if spec.attention_impl == "flash_attn":
        if device.type != "cuda":
            return "external flash-attn requires CUDA"
        try:
            load_flash_attn_func()
        except Exception as exc:
            return str(exc)

    if spec.attention_impl == "sage":
        if device.type != "cuda":
            return "SageAttention requires CUDA"
        try:
            load_sage_attn_func()
        except Exception as exc:
            return str(exc)

    return None


def build_candidate_model(
    baseline: BaselineTransformer,
    config: TransformerConfig,
    spec: VariantSpec,
    args: argparse.Namespace,
    device: torch.device,
    base_dtype: torch.dtype,
    fast_dtype: torch.dtype,
) -> nn.Module:
    precheck = variant_precheck_error(
        spec, device, base_dtype, fast_dtype
    )
    if precheck:
        return UnavailableVariant(precheck).to(device=device)

    if spec.selective_precision != "none":
        effective_sdpa_backend = (
            spec.sdpa_backend_override or args.sdpa_backend
        )
        candidate = SelectivePrecisionTransformer(
            config,
            fast_dtype=fast_dtype,
            precision_mode=spec.selective_precision,
            attention_impl=spec.attention_impl,
            sdpa_backend=effective_sdpa_backend,
            selective_layers=spec.selective_layers,
        )
        copy_baseline_weights_selective(baseline, candidate)
        candidate.set_assume_all_valid_mask(args.padding_ratio == 0.0)
        # First move the FP32 state to the target GPU, then convert only the
        # selected projection/FFN modules to the Tensor-Core dtype.
        candidate = candidate.to(
            device=device, dtype=base_dtype
        ).eval()
        candidate.configure_selected_modules()
        model: nn.Module = candidate
    else:
        candidate = AblationTransformer(
            config,
            spec,
            sdpa_backend=args.sdpa_backend,
        )
        copy_baseline_weights(baseline, candidate)
        candidate.set_assume_all_valid_mask(args.padding_ratio == 0.0)

        candidate_dtype = fast_dtype if spec.fast_dtype else base_dtype
        candidate = candidate.to(
            device=device, dtype=candidate_dtype
        ).eval()

        model = candidate
        if spec.fast_dtype and candidate_dtype != base_dtype:
            model = ComputeDtypeWrapper(candidate, candidate_dtype).eval()

    if spec.compile_candidate:
        try:
            model = maybe_compile(
                model,
                enabled=True,
                mode=args.compile_mode,
                fullgraph=args.compile_fullgraph,
            )
        except Exception as exc:
            return UnavailableVariant(
                f"torch.compile setup failed: {type(exc).__name__}: {exc}"
            ).to(device=device)

    return model


def run_one_configuration(
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    variant_keys: Sequence[str],
    config: TransformerConfig,
    *,
    shape_id: Optional[int] = None,
    csv_append: bool = False,
) -> Optional[ShapeRunResult]:
    config.validate()
    fast_dtype = resolve_fast_dtype(args.fast_dtype, dtype, device)

    estimated_gib = estimate_manual_attention_working_set_gib(config, dtype)
    input_gib = estimate_tensor_gib(config, dtype, 1.0)
    packed_qkv_gib = estimate_tensor_gib(config, dtype, 3.0)
    attention_flops = estimate_attention_flops(config)
    label = f"competition shape {shape_id}" if shape_id is not None else "configuration"

    print("\n" + "=" * 112)
    print(f"=== {label} ===")
    print(config)
    print(
        f"estimated explicit-attention score+softmax working set "
        f"(lower bound)={estimated_gib:.2f} GiB"
    )
    print(
        f"activation scale: input={input_gib:.2f} GiB, "
        f"packed-QKV output={packed_qkv_gib:.2f} GiB"
    )
    print(
        f"attention-only forward work across layers="
        f"{attention_flops / 1e12:.3f} TFLOP "
        f"({attention_flops / 1e15:.6f} PFLOP)"
    )

    guard_enabled = args.max_manual_attention_gib > 0
    unsafe_manual = (
        guard_enabled
        and estimated_gib > args.max_manual_attention_gib
        and not args.force_manual_attention
    )

    if unsafe_manual and args.padding_ratio != 0.0:
        print(
            "[SKIP] long-sequence-safe mode requires padding_ratio=0; "
            "dense padding+causal masks are themselves O(S^2)."
        )
        return None

    accuracy_config = config
    accuracy_keys = list(variant_keys)
    performance_keys = list(variant_keys)
    model_keys = list(variant_keys)
    reference_key = "A"

    if unsafe_manual:
        prefix_len = min(config.seq_len, args.long_seq_accuracy_prefix)
        accuracy_config = replace(config, seq_len=prefix_len)

        if "G" not in model_keys:
            model_keys.append("G")
        if "G" not in accuracy_keys:
            accuracy_keys.append("G")

        # Full-size timing accepts every selected path that is genuinely
        # memory-efficient under the all-valid-mask specialization.
        performance_keys = ["G"]
        for key in variant_keys:
            if key == "A":
                continue
            if variant_is_memory_safe(VARIANT_SPECS[key]) and key not in performance_keys:
                performance_keys.append(key)

        reference_key = "G"
        print(
            "[long-safe] exact A is validated only on a manageable prefix; "
            "full-size timing never materializes SxS scores/masks."
        )
        print(
            f"[long-safe] correctness prefix: S={prefix_len}; "
            f"full-size reference={reference_key}"
        )
        print(
            "[long-safe] full-size candidate set: " + ",".join(performance_keys)
        )

    baseline = BaselineTransformer(config)
    baseline = baseline.to(device=device, dtype=dtype).eval()
    models: Dict[str, nn.Module] = {"A": baseline}

    # Build candidates from a CPU/base-dtype shadow baseline so packed weight
    # concatenation is deterministic even when an experimental candidate uses
    # lower precision internally.
    baseline_shadow = BaselineTransformer(config)
    baseline_shadow.load_state_dict(copy.deepcopy(
        {k: v.detach().cpu() for k, v in baseline.state_dict().items()}
    ))

    for key in model_keys:
        if key == "A":
            continue
        models[key] = build_candidate_model(
            baseline_shadow,
            config,
            VARIANT_SPECS[key],
            args,
            device,
            dtype,
            fast_dtype,
        )

    if args.compile:
        print(
            f"[compile-all] compiling every required model including A "
            f"with mode={args.compile_mode}"
        )
        for key in model_keys:
            try:
                models[key] = maybe_compile(
                    models[key],
                    enabled=True,
                    mode=args.compile_mode,
                    fullgraph=args.compile_fullgraph,
                )
            except Exception as exc:
                if key == "A":
                    raise
                models[key] = UnavailableVariant(
                    f"compile-all failed: {type(exc).__name__}: {exc}"
                ).to(device=device)

    print("=== Configuration ===")
    print(
        f"device={device}, dtype={dtype}, fast_dtype={fast_dtype}, "
        f"torch={torch.__version__}, sdpa_backend={args.sdpa_backend}"
    )
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")
        try:
            capability = torch.cuda.get_device_capability(device)
            print(f"cuda_compute_capability={capability[0]}.{capability[1]}")
        except Exception:
            pass

    print(
        f"padding_ratio={args.padding_ratio:g}, "
        f"mask-skip effective={args.padding_ratio == 0.0}"
    )
    print(f"selected variants={','.join(variant_keys)}")
    if performance_keys != list(variant_keys):
        print(f"full-size requested timing set={','.join(performance_keys)}")

    print_variant_definition_table(
        variant_keys,
        mask_skip_effective=(args.padding_ratio == 0.0),
        fast_dtype=fast_dtype,
    )

    if accuracy_config.seq_len != config.seq_len:
        print(
            f"\n[accuracy] exact-A prefix S={accuracy_config.seq_len} "
            f"(full S={config.seq_len})"
        )

    accuracy = run_accuracy_suite(
        models=models,
        variant_keys=accuracy_keys,
        config=accuracy_config,
        device=device,
        dtype=dtype,
        trials=args.accuracy_trials,
        seed=args.seed,
        padding_ratio=args.padding_ratio,
        input_scale=args.input_scale,
        rtol=args.rtol,
        atol=args.atol,
        accuracy_profile=args.accuracy_profile,
    )

    numerical_failures = [
        key for key, summary in accuracy.items()
        if not summary.passed and not summary.error
    ]
    runtime_errors = [
        key for key, summary in accuracy.items()
        if summary.error
    ]

    if numerical_failures:
        print("\nNumerical failures: " + ", ".join(numerical_failures))
    if runtime_errors:
        print("Unavailable/error variants: " + ", ".join(runtime_errors))

    # Experimental failures do not suppress good candidates. This is crucial for
    # tolerance-search: benchmark everything that passes, and let the checker
    # reject only the unsafe approximation.
    filtered_performance: List[str] = []
    for key in performance_keys:
        if key == reference_key or key == "A":
            filtered_performance.append(key)
            continue
        summary = accuracy.get(key, AccuracySummary())
        if summary.error:
            continue
        if summary.passed or args.benchmark_on_failure:
            filtered_performance.append(key)

    performance_keys = filtered_performance
    if reference_key not in performance_keys:
        print(f"[SKIP] timing reference {reference_key} is unavailable")
        return ShapeRunResult(
            shape_id, config, {}, accuracy, reference_key=reference_key
        )

    timing_warmup = args.warmup
    timing_repeats = args.repeats
    timing_rounds = args.benchmark_rounds
    if args.auto_long_seq_timing and config.seq_len >= args.long_seq_threshold:
        timing_warmup = args.long_seq_warmup
        timing_repeats = args.long_seq_repeats
        timing_rounds = args.long_seq_rounds
        print(
            "[long-seq timing] "
            f"warmup={timing_warmup}, repeats={timing_repeats}, "
            f"rounds={timing_rounds}"
        )

    timings = benchmark_variants(
        models=models,
        variant_keys=performance_keys,
        config=config,
        device=device,
        dtype=dtype,
        seed=args.seed,
        padding_ratio=args.padding_ratio,
        input_scale=args.input_scale,
        warmup=timing_warmup,
        repeats=timing_repeats,
        rounds=timing_rounds,
        block_size=args.timing_block_size,
    )

    print_timing_table(
        timings=timings,
        variant_keys=performance_keys,
        config=config,
        accuracy=accuracy,
        reference_key=reference_key,
    )
    print_effect_analysis(timings, reference_key=reference_key)

    if args.csv:
        save_csv_results(
            path=args.csv,
            timings=timings,
            variant_keys=performance_keys,
            config=config,
            accuracy=accuracy,
            device=device,
            dtype=dtype,
            padding_ratio=args.padding_ratio,
            sdpa_backend=args.sdpa_backend,
            fast_dtype=fast_dtype,
            reference_key=reference_key,
            shape_id=shape_id,
            append=csv_append,
        )

    del models
    del baseline
    del baseline_shadow
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return ShapeRunResult(
        shape_id,
        config,
        timings,
        accuracy,
        reference_key=reference_key,
    )


def print_competition_sweep_summary(
    results: Sequence[Optional[ShapeRunResult]],
    variant_keys: Sequence[str],
) -> None:
    print("\n=== Competition sweep summary ===")
    print(
        f"{'Shape':>5} {'B':>7} {'S':>8} {'D':>6} {'H':>4} "
        f"{'Ref':>5} {'Best':>6} {'Best speedup':>13} {'H speedup':>11}"
    )
    print("-" * 83)

    for result in results:
        if result is None:
            continue
        cfg = result.config
        if not result.timings:
            print(
                f"{str(result.shape_id):>5} {cfg.batch_size:>7} {cfg.seq_len:>8} "
                f"{cfg.d_model:>6} {cfg.num_heads:>4} {result.reference_key:>5} "
                f"{'FAIL':>6} {'-':>13} {'-':>11}"
            )
            continue

        reference_key = result.reference_key
        reference_ms = result.timings[reference_key].median_ms
        valid_keys = [
            key for key in result.timings
            if (
                key == reference_key
                or key == "A"
                or (
                    key in result.accuracy
                    and result.accuracy[key].passed
                    and not result.accuracy[key].error
                )
            )
        ]
        best_key = min(valid_keys, key=lambda key: result.timings[key].median_ms)
        best_speedup = reference_ms / result.timings[best_key].median_ms

        h_text = "-"
        if (
            "H" in result.timings
            and result.accuracy.get("H", AccuracySummary()).passed
        ):
            h_text = f"{reference_ms / result.timings['H'].median_ms:.3f}x"

        print(
            f"{str(result.shape_id):>5} {cfg.batch_size:>7} {cfg.seq_len:>8} "
            f"{cfg.d_model:>6} {cfg.num_heads:>4} {reference_key:>5} {best_key:>6} "
            f"{best_speedup:>12.3f}x {h_text:>11}"
        )


def write_dispatch_json(
    path: str,
    results: Sequence[Optional[ShapeRunResult]],
) -> None:
    dispatch: Dict[str, dict] = {}
    for result in results:
        if result is None or result.shape_id is None or not result.timings:
            continue
        ref = result.reference_key
        ref_ms = result.timings[ref].median_ms
        valid = [
            key for key in result.timings
            if (
                key == ref
                or key == "A"
                or (
                    key in result.accuracy
                    and result.accuracy[key].passed
                    and not result.accuracy[key].error
                )
            )
        ]
        best = min(valid, key=lambda key: result.timings[key].median_ms)
        dispatch[str(result.shape_id)] = {
            "variant": best,
            "name": VARIANT_SPECS[best].name,
            "reference": ref,
            "median_ms": result.timings[best].median_ms,
            "speedup_vs_reference": ref_ms / result.timings[best].median_ms,
        }

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dispatch, indent=2) + "\n", encoding="utf-8")
    print(f"\nDispatch JSON written to: {output}")


def main() -> int:
    args = parse_args()

    if args.quick:
        args.accuracy_trials = 1
        args.warmup = 2
        args.repeats = 10
        args.benchmark_rounds = 1
        print(
            "[quick] accuracy_trials=1, warmup=2, repeats=10, benchmark_rounds=1"
        )

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    variant_keys = parse_variant_keys(args.variants)
    validate_args(args, device, dtype)

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision(args.matmul_precision)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.allow_tf32 = args.allow_tf32

    if hasattr(torch, "compile"):
        try:
            import torch._dynamo as torch_dynamo

            torch_dynamo.config.cache_size_limit = args.dynamo_cache_size_limit
        except Exception as exc:
            print(f"[warning] could not raise torch._dynamo cache_size_limit: {exc}")

    if args.competition_shapes:
        shape_ids = parse_shape_ids(args.shape_ids)
        print(
            "=== Official competition sweep ===\n"
            f"shape_ids={shape_ids}\n"
            "Shape 14 uses exact-A prefix validation plus full-size "
            "memory-efficient candidate timing."
        )

        results: List[Optional[ShapeRunResult]] = []
        wrote_csv = False
        for shape_id in shape_ids:
            config = competition_config(shape_id)
            result = run_one_configuration(
                args,
                device,
                dtype,
                variant_keys,
                config,
                shape_id=shape_id,
                csv_append=wrote_csv,
            )
            results.append(result)
            if args.csv and result is not None and result.timings:
                wrote_csv = True

        print_competition_sweep_summary(results, variant_keys)
        if args.dispatch_json:
            write_dispatch_json(args.dispatch_json, results)

        failures = [
            result
            for result in results
            if result is not None
            and result.accuracy
            and any(not s.passed for s in result.accuracy.values())
        ]
        return 2 if failures and args.strict_exit else 0

    config = TransformerConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        num_heads=args.heads,
        ffn_dim=args.ffn_dim,
        num_layers=args.layers,
        causal=args.causal,
    )

    result = run_one_configuration(
        args,
        device,
        dtype,
        variant_keys,
        config,
    )
    if result is None:
        return 3
    failures = (
        result.accuracy
        and any(not s.passed for s in result.accuracy.values())
    )
    return 2 if failures and args.strict_exit else 0


if __name__ == "__main__":
    raise SystemExit(main())
