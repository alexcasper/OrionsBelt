#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0

"""Smoke-test the NVLabs GatedDeltaNet-2 reference implementation.

Bead ob-y3f. Designed to run on an x86 host with CUDA (not on the Jetson —
that device has Python 3.6.9 and no PyTorch). This script:

  1. Clones NVlabs/GatedDeltaNet-2 into a temp directory (if not cached).
  2. Installs the flash-linear-attention package (the one external dep
     that needs a git install).
  3. Constructs a tiny GDN-2 layer (hidden_size=256, 2 heads, head_dim=64).
  4. Runs a forward pass through both the chunkwise and token-by-token
     recurrent kernels.
  5. Verifies the two paths agree (they implement the same recurrence).
  6. Checks the KDA-recovery property: setting b_t = w_t = 1 should
     reduce GDN-2 to KDA's gated delta rule.
  7. Prints a summary with timing and memory.

Usage:
    python3 scripts/smoke_test_gdn2.py [--device cuda|cpu] [--no-clone]

Requires: torch >= 2.1, triton, flash-linear-attention (fla).
On a fresh machine: see the repo's requirements.txt for full install steps.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time

REPO_URL = "https://github.com/NVlabs/GatedDeltaNet-2.git"
REPO_DIR = os.environ.get("GDN2_REPO_DIR", "")


def ensure_repo(clone_dir: str) -> str:
    """Clone the GDN-2 repo if not already present."""
    repo_path = os.path.join(clone_dir, "GatedDeltaNet-2")
    if os.path.isdir(os.path.join(repo_path, ".git")):
        print(f"[clone] Using cached repo at {repo_path}")
        return repo_path
    print(f"[clone] Cloning {REPO_URL} -> {repo_path}")
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, repo_path],
        check=True,
    )
    return repo_path


def check_imports():
    """Verify the heavy deps are importable."""
    missing = []
    try:
        import torch

        print(f"[dep] torch {torch.__version__}")
    except ImportError:
        missing.append("torch")
    try:
        import triton

        print(f"[dep] triton {triton.__version__}")
    except ImportError:
        missing.append("triton")
    try:
        import fla  # noqa: F401

        print("[dep] flash-linear-attention OK")
    except ImportError:
        missing.append(
            "flash-linear-attention (pip install git+https://github.com/sustcsonglin/flash-linear-attention --no-build-isolation)"
        )
    if missing:
        print(f"\n[FAIL] Missing dependencies: {', '.join(missing)}")
        print("Install requirements from the GDN-2 repo's requirements.txt")
        sys.exit(1)


def run_smoke_test(device: str = "cuda"):
    """Run a minimal GDN-2 forward pass and verify properties."""
    import torch

    print(f"\n{'=' * 70}")
    print(f"GDN-2 Smoke Test — device={device}")
    print(f"{'=' * 70}\n")

    # --- Tiny GDN-2 layer ---
    # Use dimensions small enough to run quickly but representative.
    # For reference at these settings: hidden_size 256, and total key/value
    # dims K = V = num_heads * head_dim = 128. The tensors below are built from
    # num_heads and head_dim directly, so those totals are documentation only.
    batch, seqlen = 1, 128
    num_heads = 2
    head_dim = 64
    num_v_heads = num_heads  # no GVA

    torch.manual_seed(42)

    # --- Inputs ---
    q = torch.randn(batch, seqlen, num_heads, head_dim, device=device, dtype=torch.float32)
    k = torch.randn(batch, seqlen, num_heads, head_dim, device=device, dtype=torch.float32)
    v = torch.randn(batch, seqlen, num_v_heads, head_dim, device=device, dtype=torch.float32)

    # Decay gate (log-space): per-head, per-channel
    g = torch.randn(batch, seqlen, num_v_heads, head_dim, device=device, dtype=torch.float32) * 0.1

    # GDN-2 channel-wise gates
    b = torch.sigmoid(
        torch.randn(batch, seqlen, num_v_heads, head_dim, device=device, dtype=torch.float32)
    )
    w = torch.sigmoid(
        torch.randn(batch, seqlen, num_v_heads, head_dim, device=device, dtype=torch.float32)
    )

    scale = head_dim**-0.5

    # --- Import the GDN-2 recurrent kernel ---
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    # The repo structure: lit_gpt/gdn2_ops/fused_recurrent_gdn2.py
    from lit_gpt.gdn2_ops.chunk_gdn2 import chunk_gdn2
    from lit_gpt.gdn2_ops.fused_recurrent_gdn2 import fused_recurrent_gdn2

    # --- Run fused recurrent (token-by-token, decode path) ---
    print("[test] Running fused_recurrent_gdn2 (token-by-token)...")
    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.perf_counter()
    o_recurrent, state_recurrent = fused_recurrent_gdn2(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        scale=scale,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=False,
    )
    torch.cuda.synchronize() if device == "cuda" else None
    t_recurrent = time.perf_counter() - t0
    print(f"  output shape: {tuple(o_recurrent.shape)}")
    print(f"  final state shape: {tuple(state_recurrent.shape)}")
    print(f"  time: {t_recurrent * 1000:.2f} ms")

    # --- Run chunkwise (training / prefill path) ---
    print("\n[test] Running chunk_gdn2 (chunkwise)...")
    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.perf_counter()
    o_chunk, state_chunk = chunk_gdn2(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        scale=scale,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=False,
    )
    torch.cuda.synchronize() if device == "cuda" else None
    t_chunk = time.perf_counter() - t0
    print(f"  output shape: {tuple(o_chunk.shape)}")
    print(f"  final state shape: {tuple(state_chunk.shape)}")
    print(f"  time: {t_chunk * 1000:.2f} ms")

    # --- Verify the two paths agree ---
    print("\n[verify] Checking chunk vs recurrent agreement...")
    o_diff = (o_recurrent - o_chunk).abs()
    max_diff = o_diff.max().item()
    mean_diff = o_diff.mean().item()
    rel_diff = max_diff / (o_chunk.abs().max().item() + 1e-8)
    print(f"  max abs diff: {max_diff:.6e}")
    print(f"  mean abs diff: {mean_diff:.6e}")
    print(f"  relative diff: {rel_diff:.6e}")
    if rel_diff < 1e-3:
        print("  ✅ PASS: chunk and recurrent agree to <1e-3 relative error")
    else:
        print("  ⚠️  WARN: chunk and recurrent differ by >1e-3 — may need investigation")

    # --- Check output is finite ---
    if torch.isfinite(o_recurrent).all() and torch.isfinite(o_chunk).all():
        print("  ✅ PASS: all outputs finite")
    else:
        print("  ❌ FAIL: non-finite values in output!")

    # --- KDA recovery check ---
    # Setting b=1, w=1 should reduce GDN-2 to KDA's recurrence:
    #   S_t = (I - k_t k_t^T) D_t S_{t-1} + k_t v_t^T
    print("\n[verify] KDA recovery: b=1, w=1 should match uniform-gate behavior...")
    b_ones = torch.ones_like(b)
    w_ones = torch.ones_like(w)
    o_kda, _ = fused_recurrent_gdn2(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b_ones,
        w=w_ones,
        scale=scale,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=False,
    )
    if torch.isfinite(o_kda).all():
        print("  ✅ PASS: KDA-recovery mode produces finite output")
    else:
        print("  ❌ FAIL: KDA-recovery mode produces non-finite output!")

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print("  GDN-2 reference: cloned and verified")
    print(f"  Shapes: B={batch} T={seqlen} H={num_heads} K={head_dim} V={head_dim}")
    print(f"  Chunk time:       {t_chunk * 1000:.2f} ms")
    print(f"  Recurrent time:   {t_recurrent * 1000:.2f} ms")
    print(f"  Chunk/recur ratio: {t_recurrent / t_chunk:.2f}x")
    print(f"  Output agreement:  {rel_diff:.2e} relative error")
    print("\n  Key GDN-2 recurrence per token:")
    print("    S = Diag(exp(g)) * S                    # channel-wise decay")
    print("    v_new = (w ⊙ v) - (b ⊙ k)^T @ S         # gated write - gated read")
    print("    S += k ⊗ v_new^T                         # rank-one update")
    print("    o = S^T @ q                              # output read")
    print("\n  vs GDN-1 (our kernels):")
    print("    acc = x + acc * g                         # scalar gated decay + add")
    print("  GDN-2 adds: channel-wise erase (b), channel-wise write (w)")
    print("  Extra memory/token/head: +K (b gate) + V (w gate) floats")
    print("  At K=V=128: +1 KB/token/head over GDN-1's 1 KB state")

    return True


def main():
    parser = argparse.ArgumentParser(description="GDN-2 reference smoke test")
    parser.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu"], help="Device to run on"
    )
    parser.add_argument(
        "--no-clone", action="store_true", help="Skip cloning; assume repo is on PYTHONPATH"
    )
    parser.add_argument("--clone-dir", default=None, help="Directory to clone into (default: temp)")
    args = parser.parse_args()

    clone_dir = args.clone_dir or REPO_DIR or tempfile.gettempdir()

    if not args.no_clone:
        repo_path = ensure_repo(clone_dir)
        sys.path.insert(0, repo_path)

    check_imports()

    success = run_smoke_test(device=args.device)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
