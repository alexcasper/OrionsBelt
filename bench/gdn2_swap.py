#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0
"""GDN-2 layer swap experiment (bead ob-68l).

Replaces GDN-1 (Gated DeltaNet) linear-attention layers with GDN-2
(GatedDeltaNet-2) layers in a Qwen3.5-0.8B checkpoint, then runs brief
CPU adaptation to show the swapped layers can recover.

GDN-2 vs GDN-1 difference (see bench/gdn2_reference.py for full derivation):
  - GDN-1 uses a single input gate β:  delta = (v − kᵀS) · β
  - GDN-2 splits this into an erase gate b and a write gate w:
      erase = (b ⊙ k)ᵀ S
      v_new = (w ⊙ v) − erase

When b = w = β, GDN-2 reduces to GDN-1.  The extra gate capacity is the point:
GDN-2 can learn to selectively forget (b < 1) or selectively write (w < 1)
per channel, which GDN-1's single β cannot.

Usage:
    python3 bench/gdn2_swap.py [--layers L1,L2,...] [--steps N] [--seq-len S]

Outputs JSON results to stdout and writes CSV to results/raw/ if --csv.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── GDN-2 recurrence (pure PyTorch, token-by-token) ──────────────────────────


def gdn2_recurrent(
    query,  # [B, H, T, K_dim]
    key,  # [B, H, T, K_dim]
    value,  # [B, H, T, V_dim]
    g,  # [B, H, T]  pre-computed log-decay (already negative)
    b_gate,  # [B, H, T, K_dim]  erase gate (already sigmoid'd, in [0,1])
    w_gate,  # [B, H, T, V_dim]  write gate (already sigmoid'd, in [0,1])
    use_qk_l2norm=True,
):
    """Token-by-token GDN-2 recurrence. Returns [B, H, T, V_dim]."""
    initial_dtype = query.dtype
    if use_qk_l2norm:
        query = F.normalize(query, dim=-1, eps=1e-6)
        key = F.normalize(key, dim=-1, eps=1e-6)

    query, key, value, g, b_gate, w_gate = [
        x.transpose(1, 2).contiguous().float() for x in (query, key, value, g, b_gate, w_gate)
    ]

    B, H, T, K_dim = key.shape
    V_dim = value.shape[-1]
    scale = 1.0 / (K_dim**0.5)
    query = query * scale

    state = torch.zeros(B, H, K_dim, V_dim, dtype=torch.float32, device=query.device)
    output = torch.zeros(B, H, T, V_dim, dtype=torch.float32, device=query.device)

    for i in range(T):
        q_t = query[:, :, i]  # [B, H, K]
        k_t = key[:, :, i]  # [B, H, K]
        v_t = value[:, :, i]  # [B, H, V]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]
        b_t = b_gate[:, :, i]  # [B, H, K]
        w_t = w_gate[:, :, i]  # [B, H, V]

        # Decay
        state = state * g_t

        # Gated erase: (b ⊙ k)ᵀ S  →  [B, H, V]
        erase = (state * (b_t * k_t).unsqueeze(-1)).sum(dim=-2)

        # Gated write minus erase
        v_new = w_t * v_t - erase

        # Rank-one update: S += k ⊗ v_new
        state = state + k_t.unsqueeze(-1) * v_new.unsqueeze(-2)

        # Read: qᵀ S  →  [B, H, V]
        output[:, :, i] = (state * q_t.unsqueeze(-1)).sum(dim=-2)

    output = output.transpose(1, 2).contiguous().to(initial_dtype)  # [B, T, H, V]
    return output


# ── GDN-2 attention module ────────────────────────────────────────────────────
#
# Subclasses the existing Qwen3_5GatedDeltaNet.  Inherits all projections
# (in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, conv1d, A_log, dt_bias,
# norm, out_proj) and adds two new projections for the erase and write gates.
# The forward method replaces the GDN-1 delta-rule recurrence with the GDN-2
# recurrence.


class Qwen3_5GDN2(nn.Module):
    """GDN-2 attention: drop-in replacement for Qwen3_5GatedDeltaNet.

    Copies weights from a source GDN-1 module and adds two new gate projections.
    The existing beta gate (in_proj_b) is retained but unused — the erase/write
    gates supersede it.
    """

    def __init__(self, gdn1_module: nn.Module):
        super().__init__()
        # Copy all existing parameters and buffers
        self.hidden_size = gdn1_module.hidden_size
        self.num_v_heads = gdn1_module.num_v_heads
        self.num_k_heads = gdn1_module.num_k_heads
        self.head_k_dim = gdn1_module.head_k_dim
        self.head_v_dim = gdn1_module.head_v_dim
        self.key_dim = gdn1_module.key_dim
        self.value_dim = gdn1_module.value_dim
        self.conv_kernel_size = gdn1_module.conv_kernel_size
        self.layer_idx = gdn1_module.layer_idx
        self.layer_norm_epsilon = gdn1_module.layer_norm_epsilon

        # Copy shared modules (same references — weights come from checkpoint)
        self.conv1d = gdn1_module.conv1d
        self.dt_bias = gdn1_module.dt_bias
        self.A_log = gdn1_module.A_log
        self.norm = gdn1_module.norm
        self.out_proj = gdn1_module.out_proj
        self.in_proj_qkv = gdn1_module.in_proj_qkv
        self.in_proj_z = gdn1_module.in_proj_z
        self.in_proj_a = gdn1_module.in_proj_a
        self.in_proj_b = gdn1_module.in_proj_b  # kept but unused

        # New GDN-2 gate projections
        # Erase gate: per key channel  [hidden_size → key_dim]
        self.in_proj_erase_gate = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        # Write gate: per value channel [hidden_size → value_dim]
        self.in_proj_write_gate = nn.Linear(self.hidden_size, self.value_dim, bias=False)

        # Match the model's dtype (bf16) so projections don't cause dtype errors
        model_dtype = next(gdn1_module.parameters()).dtype
        self.in_proj_erase_gate = self.in_proj_erase_gate.to(model_dtype)
        self.in_proj_write_gate = self.in_proj_write_gate.to(model_dtype)

        # Initialize so gates start near 0.5 (sigmoid of small values).
        # With gain=0.1, the Xavier init produces small weights, so the gate
        # logits are near 0 and sigmoid output ≈ 0.5.  Adaptation then
        # adjusts the gates to learn the optimal erase/write split.
        with torch.no_grad():
            nn.init.xavier_uniform_(self.in_proj_erase_gate.weight, gain=0.1)
            nn.init.xavier_uniform_(self.in_proj_write_gate.weight, gain=0.1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params=None,
        attention_mask=None,
        **kwargs,
    ):
        batch_size, seq_len, _ = hidden_states.shape

        # ── Projections (same as GDN-1) ──
        mixed_qkv = self.in_proj_qkv(hidden_states)  # [B, T, key_dim*2 + value_dim]
        mixed_qkv_t = mixed_qkv.transpose(1, 2)  # [B, conv_dim, T]

        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        a = self.in_proj_a(hidden_states)  # [B, T, num_v_heads]

        # ── Causal Conv1d (depthwise, SiLU activation) ──
        conv_out = self.conv1d(mixed_qkv_t)  # [B, conv_dim, T + k - 1]
        conv_out = conv_out[:, :, :seq_len]  # causal slice
        conv_out = F.silu(conv_out)
        mixed_qkv = conv_out.transpose(1, 2)  # [B, T, conv_dim]

        # ── Split QKV ──
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        # ── Compute gates ──
        # Decay gate (same as GDN-1)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        # GDN-2 erase and write gates
        b_raw = self.in_proj_erase_gate(hidden_states)  # [B, T, key_dim]
        w_raw = self.in_proj_write_gate(hidden_states)  # [B, T, value_dim]
        b_gate = torch.sigmoid(b_raw)
        w_gate = torch.sigmoid(w_raw)

        # Reshape gates to per-head
        b_gate = b_gate.reshape(batch_size, seq_len, -1, self.head_k_dim)  # [B, T, H, K]
        w_gate = w_gate.reshape(batch_size, seq_len, -1, self.head_v_dim)  # [B, T, H, V]

        # Handle num_v_heads > num_k_heads (repeat key-grouped heads)
        if self.num_v_heads // self.num_k_heads > 1:
            rep = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(rep, dim=2)
            key = key.repeat_interleave(rep, dim=2)
            b_gate = b_gate.repeat_interleave(rep, dim=2)

        # Transpose to [B, H, T, dim] for recurrence
        query = query.transpose(1, 2)  # [B, H, T, K]
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        g.unsqueeze(-1)  # [B, T, H, 1] → need [B, H, T]
        # g is [B, T, H], reshape for recurrence: [B, H, T]
        g_for_rec = g.permute(0, 2, 1).contiguous()  # [B, H, T]
        b_gate = b_gate.transpose(1, 2)  # [B, H, T, K]
        w_gate = w_gate.transpose(1, 2)  # [B, H, T, V]

        # ── GDN-2 recurrence ──
        core_attn_out = gdn2_recurrent(
            query,
            key,
            value,
            g_for_rec,
            b_gate,
            w_gate,
            use_qk_l2norm=True,
        )  # [B, T, H, V]

        # ── Gated norm + output projection (same as GDN-1) ──
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z_flat = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z_flat)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        output = self.out_proj(core_attn_out)
        return output


# ── Layer swap ────────────────────────────────────────────────────────────────


def swap_gdn1_to_gdn2(model, layer_indices):
    """Replace GDN-1 modules at the given layer indices with GDN-2 modules."""
    tm = model.model  # Qwen3_5TextModel
    swapped = []
    for idx in layer_indices:
        old_module = tm.layers[idx].linear_attn
        new_module = Qwen3_5GDN2(old_module)
        tm.layers[idx].linear_attn = new_module
        swapped.append(idx)
    return swapped


def count_parameters(module):
    """Count trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# ── Loss evaluation ───────────────────────────────────────────────────────────

ADAPTATION_TEXT = (
    "The future of efficient sequence modeling lies in linear attention mechanisms "
    "that avoid the quadratic cost of full self-attention. Gated DeltaNet and its "
    "successor GDN-2 represent the state of the art in this direction, using a "
    "compact recurrent state matrix that grows linearly with model dimension but "
    "remains constant in sequence length. The key innovation of GDN-2 is the "
    "separation of the erase and write gates, allowing the network to selectively "
    "forget stale information while selectively writing new content. This decoupling "
    "gives GDN-2 strictly more expressive power than the original Gated DeltaNet, "
    "whose single input gate must simultaneously control both forgetting and writing. "
    "On resource-constrained edge devices like the RK3588, where memory bandwidth "
    "rather than compute is the bottleneck, the state read-modify-write cost is "
    "identical between GDN-1 and GDN-2, making the upgrade essentially free."
)


def evaluate_loss(model, input_ids, labels):
    """Run forward pass and return cross-entropy loss."""
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels, use_cache=False)
    return outputs.loss.item()


# ── Manifest ──────────────────────────────────────────────────────────────────


def capture_manifest():
    """Capture provenance metadata (mirrors bench/manifest.py)."""
    import platform

    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
            ).strip()
        )
    except Exception:
        sha, dirty = "unknown", False

    return {
        "git_sha": sha,
        "git_dirty": dirty,
        "device": platform.node(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "timestamp": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
    }


# ── Main experiment ───────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="GDN-2 layer swap experiment (ob-68l)")
    parser.add_argument("--model", default="models/Qwen3.5-0.8B", help="Model path")
    parser.add_argument(
        "--layers",
        default="0,1,2",
        help="Comma-separated GDN-1 layer indices to swap to GDN-2",
    )
    parser.add_argument("--steps", type=int, default=30, help="Adaptation steps")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--csv", action="store_true", help="Write CSV results")
    args = parser.parse_args()

    layer_indices = [int(x) for x in args.layers.split(",")]
    print("=== GDN-2 Layer Swap Experiment (ob-68l) ===", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Layers to swap: {layer_indices}", flush=True)
    print(f"Adaptation steps: {args.steps}, seq_len: {args.seq_len}, lr: {args.lr}", flush=True)

    # ── Load model ──
    print("\n--- Loading model ---", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cpu",
    )
    model.eval()

    # ── Prepare data ──
    text_ids = tokenizer(
        ADAPTATION_TEXT, return_tensors="pt", truncation=True, max_length=args.seq_len
    )["input_ids"]
    actual_seq_len = text_ids.shape[1]
    labels = text_ids.clone()
    print(f"Input tokens: {actual_seq_len}", flush=True)

    # ── Baseline loss (pure GDN-1) ──
    print("\n--- Baseline (GDN-1) ---", flush=True)
    t0 = time.time()
    baseline_loss = evaluate_loss(model, text_ids, labels)
    baseline_time = time.time() - t0
    print(f"Baseline loss: {baseline_loss:.4f}  ({baseline_time:.1f}s)", flush=True)

    # ── Capture GDN-1 layer I/O for isolated training ──
    # We capture the input (hidden_states) and output of the first swapped
    # layer's linear_attn, so we can train the GDN-2 replacement in isolation
    # without backpropagating through the entire 800M-parameter model.
    print("\n--- Capturing GDN-1 reference I/O ---", flush=True)
    captured = {}

    def make_pre_hook(layer_idx):
        def pre_hook_fn(module, args, kwargs):
            # hidden_states is first arg or in kwargs
            hs = args[0] if args else kwargs.get("hidden_states")
            captured.setdefault(layer_idx, {})["input"] = hs.detach().clone()

        return pre_hook_fn

    def make_post_hook(layer_idx):
        def post_hook_fn(module, args, kwargs, output):
            captured.setdefault(layer_idx, {})["output"] = output.detach().clone()

        return post_hook_fn

    handles = []
    for idx in layer_indices:
        h1 = model.model.layers[idx].linear_attn.register_forward_pre_hook(
            make_pre_hook(idx), with_kwargs=True
        )
        h2 = model.model.layers[idx].linear_attn.register_forward_hook(
            make_post_hook(idx), with_kwargs=True
        )
        handles.extend([h1, h2])

    with torch.no_grad():
        model(input_ids=text_ids, use_cache=False)

    for h in handles:
        h.remove()
    print(
        f"Captured I/O for layers {layer_indices} "
        f"(input shape: {captured[layer_indices[0]]['input'].shape})",
        flush=True,
    )

    # ── Swap layers ──
    print(f"\n--- Swapping layers {layer_indices} to GDN-2 ---", flush=True)
    # Count new parameters
    total_new_params = 0
    for idx in layer_indices:
        old_module = model.model.layers[idx].linear_attn
        new_params = old_module.hidden_size * (old_module.key_dim + old_module.value_dim)
        total_new_params += new_params
    print(
        f"New GDN-2 gate parameters: {total_new_params} "
        f"({total_new_params * 4 / 1024 / 1024:.1f} MB in fp32)",
        flush=True,
    )

    swap_gdn1_to_gdn2(model, layer_indices)
    model.train()

    # ── Post-swap loss (before adaptation) ──
    print("\n--- Post-swap loss (before adaptation) ---", flush=True)
    post_swap_loss = evaluate_loss(model, text_ids, labels)
    print(f"Post-swap loss: {post_swap_loss:.4f}", flush=True)
    if math.isnan(post_swap_loss) or math.isinf(post_swap_loss):
        print("ERROR: Loss is NaN/Inf — swap produced broken model", flush=True)
        sys.exit(1)
    print(f"Loss increase from swap: {post_swap_loss - baseline_loss:+.4f}", flush=True)

    # ── Isolated layer adaptation ──
    # Full-model backprop takes ~436s/step on this CPU (800M params).
    # Instead, we train the GDN-2 module in isolation using the cached
    # GDN-1 reference output as a distillation target (MSE loss).
    # Backprop flows only through the ~4M new gate parameters.
    print(f"\n--- Isolated layer adaptation ({args.steps} steps) ---", flush=True)

    # Freeze everything except the new gate projections
    for p in model.parameters():
        p.requires_grad = False
    trainable_count = 0
    trainable_params = []
    for idx in layer_indices:
        mod = model.model.layers[idx].linear_attn
        for pname in ("in_proj_erase_gate", "in_proj_write_gate"):
            param = getattr(mod, pname)
            for p in param.parameters():
                p.requires_grad = True
                trainable_count += p.numel()
                trainable_params.append(p)
    print(
        f"Trainable parameters: {trainable_count} ({trainable_count * 4 / 1024 / 1024:.1f} MB)",
        flush=True,
    )
    print(
        "Strategy: MSE distillation against cached GDN-1 output (isolated, no full-model backprop)",
        flush=True,
    )

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    losses = []
    t0 = time.time()
    for step in range(args.steps):
        optimizer.zero_grad()
        step_loss = 0.0
        for idx in layer_indices:
            mod = model.model.layers[idx].linear_attn
            ref_input = captured[idx]["input"]
            ref_output = captured[idx]["output"]
            gdn2_output = mod(ref_input)
            loss = F.mse_loss(gdn2_output.float(), ref_output.float())
            loss.backward()
            step_loss += loss.item()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        losses.append(step_loss / len(layer_indices))
        if step == 0 or (step + 1) % 5 == 0 or step == args.steps - 1:
            elapsed = time.time() - t0
            print(
                f"  Step {step + 1:3d}/{args.steps}: mse={step_loss / len(layer_indices):.6f}  "
                f"grad_norm={grad_norm.item():.2f}  ({elapsed:.1f}s elapsed)",
                flush=True,
            )

    # ── Final evaluation (full-model CE loss, no_grad) ──
    print("\n--- Final evaluation (full-model CE loss) ---", flush=True)
    model.eval()
    final_loss = evaluate_loss(model, text_ids, labels)
    print(f"Final CE loss: {final_loss:.4f}", flush=True)
    print(
        f"CE recovery: {post_swap_loss:.4f} → {final_loss:.4f} "
        f"({final_loss - post_swap_loss:+.4f})",
        flush=True,
    )
    print(
        f"vs Baseline: {baseline_loss:.4f} → {final_loss:.4f} ({final_loss - baseline_loss:+.4f})",
        flush=True,
    )
    print(f"Isolated MSE (final step): {losses[-1]:.6f}", flush=True)

    # ── Results ──
    manifest = capture_manifest()
    results = {
        "experiment": "gdn2_layer_swap",
        "bead": "ob-68l",
        "model": args.model,
        "layers_swapped": layer_indices,
        "seq_len": actual_seq_len,
        "baseline_loss_gdn1": baseline_loss,
        "post_swap_loss_gdn2": post_swap_loss,
        "final_loss_after_adaptation": final_loss,
        "adaptation_steps": args.steps,
        "learning_rate": args.lr,
        "isolated_mse_curve": losses,
        "final_isolated_mse": losses[-1],
        "new_parameters": total_new_params,
        "trainable_parameters": trainable_count,
        "adaptation_strategy": "isolated_mse_distillation",
        "manifest": manifest,
    }

    # Print summary JSON
    print("\n=== RESULTS JSON ===", flush=True)
    print(json.dumps(results, indent=2), flush=True)

    # Write CSV
    if args.csv:
        csv_dir = "results/raw"
        os.makedirs(csv_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, "gdn2_swap_t3.csv")
        with open(csv_path, "w") as f:
            f.write("step,mse_loss\n")
            for i, loss in enumerate(losses):
                f.write(f"{i + 1},{loss:.6f}\n")
        print(f"\nCSV written to {csv_path}", flush=True)

        # Write manifest
        manifest_dir = "results/manifests"
        os.makedirs(manifest_dir, exist_ok=True)
        manifest_path = os.path.join(manifest_dir, "gdn2_swap_t3.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Manifest written to {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
