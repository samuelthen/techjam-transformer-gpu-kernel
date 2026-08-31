#!/usr/bin/env python3
"""Run the literal official torch_transformer_benchmark.py once per official
shape (1-13; shape 14 is handled separately, see TECH_REPORT.md §4.8/§9 -- the
official script's own BaselineTransformer cannot execute shape 14 at all,
regardless of implementation, since manual attention there needs 4.77 TB)."""
import re
import subprocess
import sys

SHAPES = {
    # id: (batch, d_model, heads, seq_len, layers, ffn_dim)
    1: (64, 128, 4, 128, 4, 128),
    2: (1, 128, 4, 128, 4, 128),
    3: (4, 128, 4, 128, 4, 128),
    4: (16, 128, 4, 128, 4, 128),
    5: (128, 128, 4, 128, 4, 128),
    6: (10000, 128, 4, 128, 4, 128),
    7: (64, 32, 4, 128, 4, 32),
    8: (64, 1024, 4, 128, 4, 1024),
    9: (64, 128, 1, 128, 4, 128),
    10: (64, 128, 2, 128, 4, 128),
    11: (64, 128, 16, 128, 4, 128),
    12: (64, 128, 4, 32, 4, 128),
    13: (64, 128, 4, 1024, 4, 128),
}

results = {}
for shape_id, (batch, d_model, heads, seq_len, layers, ffn_dim) in SHAPES.items():
    cmd = [
        sys.executable, "torch_transformer_benchmark.py",
        "--batch-size", str(batch), "--seq-len", str(seq_len), "--d-model", str(d_model),
        "--heads", str(heads), "--ffn-dim", str(ffn_dim), "--layers", str(layers), "--causal",
        "--device", "cuda", "--dtype", "float32", "--no-allow-tf32",
        "--accuracy-trials", "10", "--rtol", "0.02", "--atol", "0.002",
        "--warmup", "20", "--repeats", str(50 if batch < 5000 else 20),
        "--benchmark-rounds", "3",
    ]
    print(f"=== shape {shape_id}: batch={batch} d_model={d_model} heads={heads} "
          f"seq_len={seq_len} layers={layers} ffn_dim={ffn_dim} ===", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    out = proc.stdout + proc.stderr
    accuracy_match = re.search(r"summary: (PASS|FAIL)", out)
    speedup_match = re.search(r"speedup\s*:\s*([\d.]+)x", out)
    accuracy = accuracy_match.group(1) if accuracy_match else "ERROR"
    speedup = float(speedup_match.group(1)) if speedup_match else None
    results[shape_id] = (accuracy, speedup)
    print(f"  -> accuracy={accuracy}  speedup={speedup}", flush=True)
    if accuracy == "ERROR" or speedup is None:
        print("  --- full output for debugging ---")
        print(out[-3000:])

print("\n=== Summary (literal official torch_transformer_benchmark.py, --no-allow-tf32) ===")
print(f"{'Shape':>5} {'Accuracy':>9} {'Speedup':>9}")
for shape_id in sorted(results):
    accuracy, speedup = results[shape_id]
    speedup_str = f"{speedup:.3f}x" if speedup is not None else "n/a"
    print(f"{shape_id:>5} {accuracy:>9} {speedup_str:>9}")
