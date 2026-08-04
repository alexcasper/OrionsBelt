#!/usr/bin/env python3
"""Emit minimal per-operator ONNX graphs to probe CIX NOE Compiler coverage of GDN ops.

Bead ``ob-t3b.2``, feeding the op-coverage audit ``ob-t3b.1``.

WHY HAND-AUTHORED GRAPHS INSTEAD OF A TORCH EXPORT
--------------------------------------------------
The whole point of the audit is to learn *which* operator the NOE Compiler cannot
handle. ``torch.onnx.export`` fuses, decomposes, and renames ops on the way out, so a
rejected export tells you a model failed, not which primitive caused it. Each graph here
contains one operator family and nothing else, so a ``cixbuild`` failure is attributable
to exactly one thing. It also keeps the probe dependency-free apart from ``onnx`` itself.

SHAPES
------
Taken from Qwen3.5's verified linear-attention config (docs/CLAIM_VERIFICATION.md §2.3):
``linear_conv_kernel_dim=4``, ``linear_key_head_dim=128``, ``linear_value_head_dim=128``,
``linear_num_key_heads=16``. Small batch/sequence values keep the graphs tiny; coverage is
a function of the operator set, not of tensor size. If a probe passes at these shapes but
fails at production shapes, that is itself a finding worth recording.

USAGE
-----
    python3 scripts/npu_op_probe.py --out artifacts/npu_op_probe

Then, once the NOE SDK is installed in its Python 3.10 environment, run the generated
``run_cixbuild.sh`` (see the emitted README) and record per-probe outcomes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
from onnx import TensorProto, helper

# Verified Qwen3.5 linear-attention shapes; see module docstring.
CONV_KERNEL = 4
KEY_HEAD_DIM = 128
VALUE_HEAD_DIM = 128
NUM_KEY_HEADS = 16

BATCH = 1
SEQ = 32
CHUNK = 8  # chunkwise formulation: SEQ / CHUNK chunks


def _tensor(name: str, dims):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, list(dims))


def _model(graph: onnx.GraphProto, opset: int = 17) -> onnx.ModelProto:
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", opset)],
        producer_name="orionsbelt-npu-op-probe",
    )
    onnx.checker.check_model(model, full_check=True)
    return model


# ---------------------------------------------------------------------------
# Probe 1 — causal depthwise Conv1D
# ---------------------------------------------------------------------------


def probe_causal_conv1d() -> tuple[onnx.ModelProto, str]:
    """Depthwise Conv1D with left-only padding, as GDN's short convolution.

    Causality comes from asymmetric padding (kernel-1 on the left, 0 on the right).
    Some NPU compilers only accept symmetric padding, which would force us to pad
    explicitly in a preceding Pad node -- a difference worth knowing about early.
    """
    channels = NUM_KEY_HEADS * KEY_HEAD_DIM
    x = _tensor("x", [BATCH, channels, SEQ])
    y = _tensor("y", [BATCH, channels, SEQ])
    w = helper.make_tensor(
        "w",
        TensorProto.FLOAT,
        [channels, 1, CONV_KERNEL],
        [0.1] * (channels * CONV_KERNEL),
    )
    node = helper.make_node(
        "Conv",
        ["x", "w"],
        ["y"],
        kernel_shape=[CONV_KERNEL],
        pads=[CONV_KERNEL - 1, 0],  # causal: left pad only
        group=channels,  # depthwise
        name="causal_depthwise_conv1d",
    )
    graph = helper.make_graph([node], "causal_conv1d", [x], [y], initializer=[w])
    return _model(graph), (
        "GDN's causal short convolution as a depthwise Conv1D with left-only padding "
        "(kernel=4, groups=channels). Tests both depthwise grouping and asymmetric pads."
    )


# ---------------------------------------------------------------------------
# Probe 2 — gated decay via Log -> CumSum -> Exp
# ---------------------------------------------------------------------------


def probe_decay_cumprod() -> tuple[onnx.ModelProto, str]:
    """Cumulative decay product along the sequence axis.

    ONNX has no CumProd operator, so the standard expression of a cumulative decay is
    ``exp(cumsum(log(a)))``. That decomposition is itself the coverage question: CumSum
    is a scan-shaped op and is a plausible gap on a matmul-oriented NPU. If CumSum is
    unsupported, the gated decay cannot be expressed at all without control flow, which
    would be a significant finding for the mapping ADR.
    """
    a = _tensor("a", [BATCH, NUM_KEY_HEADS, SEQ])
    out = _tensor("decay", [BATCH, NUM_KEY_HEADS, SEQ])
    axis = helper.make_tensor("axis", TensorProto.INT64, [], [2])
    nodes = [
        helper.make_node("Log", ["a"], ["log_a"], name="log"),
        helper.make_node("CumSum", ["log_a", "axis"], ["cs"], name="cumsum_seq"),
        helper.make_node("Exp", ["cs"], ["decay"], name="exp"),
    ]
    graph = helper.make_graph(nodes, "decay_cumprod", [a], [out], initializer=[axis])
    return _model(graph), (
        "Gated decay as exp(cumsum(log(a))) along the sequence axis. ONNX has no CumProd, "
        "so this decomposition is the only standard route -- and CumSum support is the "
        "real question on a matmul-oriented NPU."
    )


# ---------------------------------------------------------------------------
# Probe 3 — delta-rule state update
# ---------------------------------------------------------------------------


def probe_delta_rule_update() -> tuple[onnx.ModelProto, str]:
    """One delta-rule state update step: S <- (I - k k^T) S + k v^T.

    This is the arithmetic core of Gated DeltaNet, and the part that *should* map well
    to an NPU since it is all matmuls and outer products. If this probe fails, the
    NPU is essentially unusable for GDN and everything routes to GPU/CPU.
    """
    s_in = _tensor("s_in", [BATCH, NUM_KEY_HEADS, KEY_HEAD_DIM, VALUE_HEAD_DIM])
    k = _tensor("k", [BATCH, NUM_KEY_HEADS, KEY_HEAD_DIM, 1])
    v = _tensor("v", [BATCH, NUM_KEY_HEADS, 1, VALUE_HEAD_DIM])
    s_out = _tensor("s_out", [BATCH, NUM_KEY_HEADS, KEY_HEAD_DIM, VALUE_HEAD_DIM])
    nodes = [
        # k k^T S  (apply the rank-1 erase projection to the running state)
        helper.make_node("Transpose", ["k"], ["kt"], perm=[0, 1, 3, 2], name="k_T"),
        helper.make_node("MatMul", ["kt", "s_in"], ["kt_s"], name="kT_S"),
        helper.make_node("MatMul", ["k", "kt_s"], ["kkt_s"], name="k_kT_S"),
        helper.make_node("Sub", ["s_in", "kkt_s"], ["erased"], name="erase"),
        # + k v^T  (write)
        helper.make_node("MatMul", ["k", "v"], ["kv"], name="outer_kv"),
        helper.make_node("Add", ["erased", "kv"], ["s_out"], name="write"),
    ]
    graph = helper.make_graph(nodes, "delta_rule_update", [s_in, k, v], [s_out])
    return _model(graph), (
        "One delta-rule step S <- (I - k k^T) S + k v^T as batched MatMul/Sub/Add. "
        "Pure dense linear algebra, so this is the probe most likely to pass -- and if "
        "it does not, the NPU cannot host GDN layers at all."
    )


# ---------------------------------------------------------------------------
# Probe 4 — elementwise gate chain
# ---------------------------------------------------------------------------


def probe_gate_chain() -> tuple[onnx.ModelProto, str]:
    """Sigmoid / Softplus / Mul elementwise gating, as used for the decay and beta gates."""
    x = _tensor("x", [BATCH, NUM_KEY_HEADS, SEQ, KEY_HEAD_DIM])
    out = _tensor("gated", [BATCH, NUM_KEY_HEADS, SEQ, KEY_HEAD_DIM])
    nodes = [
        helper.make_node("Sigmoid", ["x"], ["sig"], name="sigmoid_gate"),
        helper.make_node("Softplus", ["x"], ["sp"], name="softplus_gate"),
        helper.make_node("Neg", ["sp"], ["neg_sp"], name="neg"),
        helper.make_node("Exp", ["neg_sp"], ["decay"], name="exp_neg_softplus"),
        helper.make_node("Mul", ["sig", "decay"], ["gated"], name="combine"),
    ]
    graph = helper.make_graph(nodes, "gate_chain", [x], [out])
    return _model(graph), (
        "Elementwise gate chain (Sigmoid, Softplus, Neg, Exp, Mul). Expected to be "
        "well supported; included so a failure here isolates cheaply."
    )


# ---------------------------------------------------------------------------
# Probe 5 — control flow: Scan and Loop  (the crux)
# ---------------------------------------------------------------------------


def probe_scan_recurrence() -> tuple[onnx.ModelProto, str]:
    """Chunkwise recurrence expressed with ONNX Scan -- the most likely failure point.

    The chunk-to-chunk dependency in GDN is inherently sequential. Expressing it in a
    single graph requires ONNX control flow (Scan or Loop), and NPU compilers built for
    feed-forward dense networks frequently support neither. If this probe fails while
    probe 3 passes, that is the empirical case for the planned split: dense per-chunk
    math on the NPU, the sequential scan driven from GPU/CPU.
    """
    num_chunks = SEQ // CHUNK
    state_in = _tensor("state_in", [BATCH, KEY_HEAD_DIM])
    chunks = _tensor("chunks", [num_chunks, BATCH, KEY_HEAD_DIM])
    state_out = _tensor("state_out", [BATCH, KEY_HEAD_DIM])
    ys = _tensor("ys", [num_chunks, BATCH, KEY_HEAD_DIM])

    body_state = _tensor("b_state", [BATCH, KEY_HEAD_DIM])
    body_chunk = _tensor("b_chunk", [BATCH, KEY_HEAD_DIM])
    body_next = _tensor("b_next", [BATCH, KEY_HEAD_DIM])
    body_y = _tensor("b_y", [BATCH, KEY_HEAD_DIM])
    decay = helper.make_tensor("decay_k", TensorProto.FLOAT, [], [0.9])
    body = helper.make_graph(
        [
            helper.make_node("Mul", ["b_state", "decay_k"], ["decayed"], name="decay_state"),
            helper.make_node("Add", ["decayed", "b_chunk"], ["b_next"], name="accumulate"),
            helper.make_node("Identity", ["b_next"], ["b_y"], name="emit"),
        ],
        "scan_body",
        [body_state, body_chunk],
        [body_next, body_y],
        initializer=[decay],
    )
    node = helper.make_node(
        "Scan",
        ["state_in", "chunks"],
        ["state_out", "ys"],
        body=body,
        num_scan_inputs=1,
        name="chunkwise_scan",
    )
    graph = helper.make_graph([node], "scan_recurrence", [state_in, chunks], [state_out, ys])
    return _model(graph), (
        "Chunk-to-chunk recurrence as an ONNX Scan (gated accumulate over "
        f"{num_chunks} chunks). THE CRUX PROBE: NPU compilers often lack control flow "
        "entirely. Failure here plus success on the delta-rule probe is the empirical "
        "justification for routing the sequential scan off the NPU."
    )


def probe_loop_recurrence() -> tuple[onnx.ModelProto, str]:
    """Same recurrence via ONNX Loop, in case the compiler supports one form but not the other."""
    state_in = _tensor("state_in", [BATCH, KEY_HEAD_DIM])
    state_out = _tensor("state_out", [BATCH, KEY_HEAD_DIM])
    trip = helper.make_tensor("trip_count", TensorProto.INT64, [], [SEQ // CHUNK])
    cond = helper.make_tensor("cond", TensorProto.BOOL, [], [True])

    it = helper.make_tensor_value_info("iter", TensorProto.INT64, [])
    keep_in = helper.make_tensor_value_info("keep_in", TensorProto.BOOL, [])
    body_state = _tensor("l_state", [BATCH, KEY_HEAD_DIM])
    keep_out = helper.make_tensor_value_info("keep_out", TensorProto.BOOL, [])
    body_next = _tensor("l_next", [BATCH, KEY_HEAD_DIM])
    decay = helper.make_tensor("l_decay", TensorProto.FLOAT, [], [0.9])
    body = helper.make_graph(
        [
            helper.make_node("Identity", ["keep_in"], ["keep_out"], name="keep"),
            helper.make_node("Mul", ["l_state", "l_decay"], ["l_next"], name="decay_state"),
        ],
        "loop_body",
        [it, keep_in, body_state],
        [keep_out, body_next],
        initializer=[decay],
    )
    node = helper.make_node(
        "Loop",
        ["trip_count", "cond", "state_in"],
        ["state_out"],
        body=body,
        name="chunkwise_loop",
    )
    graph = helper.make_graph(
        [node], "loop_recurrence", [state_in], [state_out], initializer=[trip, cond]
    )
    return _model(graph), (
        "The same recurrence via ONNX Loop rather than Scan. Compilers sometimes accept "
        "one control-flow form and not the other, so both are probed."
    )


def probe_loop_dynamic_trip() -> tuple[onnx.ModelProto, str]:
    """Probe 06 with a RUNTIME trip count — the negative control for static unrolling.

    Probe 06 compiles, which initially looked like Loop being supported. It is not:
    the IR showed the body had been statically unrolled, which is only possible
    because its trip_count is a constant initializer. This probe is identical except
    ``trip_count`` is a graph INPUT, so no unrolling is available and the compiler has
    to express the loop as control flow or refuse.

    It refuses ("Graph is not DAG"), which is what makes the operator-coverage finding
    load-bearing: the NPU can do GDN's per-chunk arithmetic but cannot drive the
    chunk-to-chunk recurrence. Without this probe, probe 06's rc=0 reads as success.
    """
    # trip_count as an input, not an initializer — that single change is the whole probe.
    trip_in = helper.make_tensor_value_info("trip_count", TensorProto.INT64, [])
    state_in = _tensor("state_in", [BATCH, KEY_HEAD_DIM])
    state_out = _tensor("state_out", [BATCH, KEY_HEAD_DIM])
    cond = helper.make_tensor("cond", TensorProto.BOOL, [], [True])

    it = helper.make_tensor_value_info("iter", TensorProto.INT64, [])
    keep_in = helper.make_tensor_value_info("keep_in", TensorProto.BOOL, [])
    body_state = _tensor("l_state", [BATCH, KEY_HEAD_DIM])
    keep_out = helper.make_tensor_value_info("keep_out", TensorProto.BOOL, [])
    body_next = _tensor("l_next", [BATCH, KEY_HEAD_DIM])
    decay = helper.make_tensor("l_decay", TensorProto.FLOAT, [], [0.9])
    body = helper.make_graph(
        [
            helper.make_node("Identity", ["keep_in"], ["keep_out"], name="keep"),
            helper.make_node("Mul", ["l_state", "l_decay"], ["l_next"], name="decay_state"),
        ],
        "loop_body",
        [it, keep_in, body_state],
        [keep_out, body_next],
        initializer=[decay],
    )
    node = helper.make_node(
        "Loop",
        ["trip_count", "cond", "state_in"],
        ["state_out"],
        body=body,
        name="dynamic_loop",
    )
    graph = helper.make_graph(
        [node], "loop_dynamic_trip", [trip_in, state_in], [state_out], initializer=[cond]
    )
    # Description text is reproduced verbatim from the hand-authored artifact this
    # function replaces, including the shape-inference detail — it is an observation
    # from the actual cixbuild run and is not re-derivable from the graph.
    return _model(graph), (
        "Follow-up added after probe 06 turned out to be statically unrolled: identical "
        "recurrence but trip_count is a graph INPUT rather than an initializer. Decides "
        "whether Loop is genuinely supported as control flow or only ever unrolled. "
        "Result: rejected ('Graph is not DAG', shape inference unreliable for non-const "
        "max_count), proving unrolling is the only mechanism."
    )


PROBES = {
    "01_causal_conv1d": probe_causal_conv1d,
    "02_decay_cumprod": probe_decay_cumprod,
    "03_delta_rule_update": probe_delta_rule_update,
    "04_gate_chain": probe_gate_chain,
    "05_scan_recurrence": probe_scan_recurrence,
    "06_loop_recurrence": probe_loop_recurrence,
    # 07 was hand-authored during the audit and never added here, so regenerating
    # silently dropped it from the manifest — and it is the probe the central finding
    # rests on. See docs/FINDINGS.md section 1.
    "07_loop_dynamic_trip": probe_loop_dynamic_trip,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="artifacts/npu_op_probe", help="output directory")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest = []
    for name, fn in PROBES.items():
        model, description = fn()
        path = out / f"{name}.onnx"
        onnx.save(model, str(path))
        ops = sorted({n.op_type for n in model.graph.node})
        # Include ops nested inside control-flow bodies, which is the whole point of 05/06.
        for node in model.graph.node:
            for attr in node.attribute:
                if attr.g.node:
                    ops = sorted(set(ops) | {n.op_type for n in attr.g.node})
        manifest.append(
            {
                "probe": name,
                "file": path.name,
                "ops": ops,
                "bytes": path.stat().st_size,
                "description": description,
            }
        )
        print(f"  {name}: ops={','.join(ops)} ({path.stat().st_size} bytes)")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\n{len(manifest)} probes written to {out}/ (+ manifest.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
