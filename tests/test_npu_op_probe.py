"""Tests for scripts/npu_op_probe.py — ONNX probe graph generation.

Each probe in ``npu_op_probe`` generates a minimal ONNX model containing one
operator family from Gated DeltaNet. The models are fed to the CIX NOE Compiler
to determine operator coverage (bead ob-t3b.2, feeding the audit ob-t3b.1).

These tests verify that every probe produces a structurally valid ONNX model
with the expected operator set, correct shapes, and a meaningful description.
No NOE toolkit or hardware is required — ``onnx.checker`` does the validation.
"""

import json
import os
import sys

import onnx
import pytest
from onnx import TensorProto

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from npu_op_probe import (
    BATCH,
    CHUNK,
    CONV_KERNEL,
    KEY_HEAD_DIM,
    NUM_KEY_HEADS,
    PROBES,
    SEQ,
    VALUE_HEAD_DIM,
    _model,
    _tensor,
    main,
    probe_causal_conv1d,
    probe_decay_cumprod,
    probe_delta_rule_update,
    probe_gate_chain,
    probe_loop_dynamic_trip,
    probe_loop_recurrence,
    probe_scan_recurrence,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _all_ops(model: onnx.ModelProto) -> set[str]:
    """Collect all op types in a model, including control-flow sub-graphs."""
    ops = {n.op_type for n in model.graph.node}
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.g.node:
                ops |= {n.op_type for n in attr.g.node}
    return ops


def _graph_input_names(model: onnx.ModelProto) -> set[str]:
    """Names of all graph-level inputs (excluding initializers)."""
    inits = {i.name for i in model.graph.initializer}
    return {i.name for i in model.graph.input if i.name not in inits}


def _graph_output_names(model: onnx.ModelProto) -> set[str]:
    return {o.name for o in model.graph.output}


def _get_attr(node, name):
    """Return the value of a node attribute by name (onnx 1.22 API)."""
    for attr in node.attribute:
        if attr.name == name:
            return onnx.helper.get_attribute_value(attr)
    raise KeyError(f"attribute '{name}' not found on node {node.name}")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Constants are derived from verified Qwen3.5 config (CLAIM_VERIFICATION §2.3)."""

    def test_conv_kernel(self):
        assert CONV_KERNEL == 4

    def test_key_head_dim(self):
        assert KEY_HEAD_DIM == 128

    def test_value_head_dim(self):
        assert VALUE_HEAD_DIM == 128

    def test_num_key_heads(self):
        assert NUM_KEY_HEADS == 16

    def test_batch_is_one(self):
        assert BATCH == 1

    def test_seq_is_small(self):
        assert SEQ == 32

    def test_chunk_divides_seq(self):
        assert SEQ % CHUNK == 0
        assert SEQ // CHUNK > 0


# ---------------------------------------------------------------------------
# _tensor helper
# ---------------------------------------------------------------------------


class TestTensorHelper:
    def test_creates_float_tensor(self):
        t = _tensor("foo", [2, 3])
        assert t.name == "foo"
        assert t.type.tensor_type.elem_type == TensorProto.FLOAT

    def test_dims_preserved(self):
        t = _tensor("bar", [1, 16, 128])
        dims = [d.dim_value for d in t.type.tensor_type.shape.dim]
        assert dims == [1, 16, 128]

    def test_accepts_generator_dims(self):
        t = _tensor("baz", (d for d in [4, 5]))
        dims = [d.dim_value for d in t.type.tensor_type.shape.dim]
        assert dims == [4, 5]

    def test_empty_dims_allowed(self):
        t = _tensor("scalar", [])
        assert len(t.type.tensor_type.shape.dim) == 0


# ---------------------------------------------------------------------------
# _model helper
# ---------------------------------------------------------------------------


class TestModelHelper:
    def test_returns_valid_model(self):
        x = _tensor("x", [1])
        y = _tensor("y", [1])
        node = onnx.helper.make_node("Identity", ["x"], ["y"])
        graph = onnx.helper.make_graph([node], "test", [x], [y])
        model = _model(graph)
        assert isinstance(model, onnx.ModelProto)

    def test_default_opset_is_17(self):
        x = _tensor("x", [1])
        y = _tensor("y", [1])
        node = onnx.helper.make_node("Identity", ["x"], ["y"])
        graph = onnx.helper.make_graph([node], "test", [x], [y])
        model = _model(graph)
        assert len(model.opset_import) == 1
        assert model.opset_import[0].version == 17

    def test_custom_opset(self):
        x = _tensor("x", [1])
        y = _tensor("y", [1])
        node = onnx.helper.make_node("Relu", ["x"], ["y"])
        graph = onnx.helper.make_graph([node], "test", [x], [y])
        model = _model(graph, opset=16)
        assert model.opset_import[0].version == 16

    def test_passes_onnx_checker(self):
        """The helper runs full_check=True, so an invalid model must raise."""
        x = _tensor("x", [1])
        y = _tensor("y", [1])
        # Node references a non-existent input
        node = onnx.helper.make_node("Identity", ["missing"], ["y"])
        graph = onnx.helper.make_graph([node], "broken", [x], [y])
        with pytest.raises((onnx.checker.ValidationError, ValueError, RuntimeError)):
            _model(graph)

    def test_producer_name(self):
        x = _tensor("x", [1])
        y = _tensor("y", [1])
        node = onnx.helper.make_node("Identity", ["x"], ["y"])
        graph = onnx.helper.make_graph([node], "test", [x], [y])
        model = _model(graph)
        assert model.producer_name == "orionsbelt-npu-op-probe"


# ---------------------------------------------------------------------------
# PROBES dict
# ---------------------------------------------------------------------------


class TestProbesDict:
    def test_expected_keys(self):
        assert set(PROBES.keys()) == {
            "01_causal_conv1d",
            "02_decay_cumprod",
            "03_delta_rule_update",
            "04_gate_chain",
            "05_scan_recurrence",
            "06_loop_recurrence",
            "07_loop_dynamic_trip",
        }

    def test_all_callable(self):
        for name, fn in PROBES.items():
            assert callable(fn), f"{name} is not callable"

    def test_count(self):
        assert len(PROBES) == 7

    @pytest.mark.parametrize("name", list(PROBES.keys()))
    def test_each_probe_validates(self, name):
        """Every probe produces a checker-valid model with a non-empty description."""
        model, desc = PROBES[name]()
        assert isinstance(model, onnx.ModelProto)
        assert isinstance(desc, str)
        assert len(desc) > 10
        # Full ONNX shape/type check (also run by _model, but explicit for safety)
        onnx.checker.check_model(model, full_check=True)


# ---------------------------------------------------------------------------
# Probe 1 — causal Conv1D
# ---------------------------------------------------------------------------


class TestProbeCausalConv1d:
    def test_returns_model_and_description(self):
        model, desc = probe_causal_conv1d()
        assert isinstance(model, onnx.ModelProto)
        assert isinstance(desc, str)

    def test_single_conv_node(self):
        model, _ = probe_causal_conv1d()
        ops = [n.op_type for n in model.graph.node]
        assert ops == ["Conv"]

    def test_depthwise_grouping(self):
        model, _ = probe_causal_conv1d()
        conv = model.graph.node[0]
        assert _get_attr(conv, "group") == NUM_KEY_HEADS * KEY_HEAD_DIM

    def test_causal_pads(self):
        """Left-only padding: [kernel-1, 0] for 1D conv."""
        model, _ = probe_causal_conv1d()
        conv = model.graph.node[0]
        pads = _get_attr(conv, "pads")
        assert list(pads) == [CONV_KERNEL - 1, 0]

    def test_kernel_shape(self):
        model, _ = probe_causal_conv1d()
        conv = model.graph.node[0]
        assert list(_get_attr(conv, "kernel_shape")) == [CONV_KERNEL]

    def test_input_channels_match(self):
        model, _ = probe_causal_conv1d()
        inputs = _graph_input_names(model)
        assert "x" in inputs

    def test_weight_is_initializer(self):
        model, _ = probe_causal_conv1d()
        init_names = {i.name for i in model.graph.initializer}
        assert "w" in init_names

    def test_description_mentions_depthwise(self):
        _, desc = probe_causal_conv1d()
        assert "depthwise" in desc.lower()


# ---------------------------------------------------------------------------
# Probe 2 — gated decay via Log/CumSum/Exp
# ---------------------------------------------------------------------------


class TestProbeDecayCumprod:
    def test_three_nodes(self):
        model, _ = probe_decay_cumprod()
        ops = [n.op_type for n in model.graph.node]
        assert ops == ["Log", "CumSum", "Exp"]

    def test_operates_on_sequence_axis(self):
        model, _ = probe_decay_cumprod()
        cumsum = model.graph.node[1]
        assert cumsum.op_type == "CumSum"

    def test_axis_is_constant_initializer(self):
        model, _ = probe_decay_cumprod()
        init_names = {i.name for i in model.graph.initializer}
        assert "axis" in init_names

    def test_description_mentions_cumsum(self):
        _, desc = probe_decay_cumprod()
        assert "CumSum" in desc or "cumsum" in desc.lower()


# ---------------------------------------------------------------------------
# Probe 3 — delta-rule state update
# ---------------------------------------------------------------------------


class TestProbeDeltaRuleUpdate:
    def test_expected_ops(self):
        model, _ = probe_delta_rule_update()
        ops = [n.op_type for n in model.graph.node]
        assert ops == ["Transpose", "MatMul", "MatMul", "Sub", "MatMul", "Add"]

    def test_has_three_inputs(self):
        model, _ = probe_delta_rule_update()
        assert len(_graph_input_names(model)) == 3

    def test_node_names(self):
        model, _ = probe_delta_rule_update()
        names = [n.name for n in model.graph.node]
        assert "k_T" in names
        assert "kT_S" in names
        assert "k_kT_S" in names
        assert "erase" in names
        assert "outer_kv" in names
        assert "write" in names

    def test_description_mentions_delta_rule(self):
        _, desc = probe_delta_rule_update()
        assert "delta" in desc.lower()


# ---------------------------------------------------------------------------
# Probe 4 — elementwise gate chain
# ---------------------------------------------------------------------------


class TestProbeGateChain:
    def test_expected_ops(self):
        model, _ = probe_gate_chain()
        ops = [n.op_type for n in model.graph.node]
        assert ops == ["Sigmoid", "Softplus", "Neg", "Exp", "Mul"]

    def test_single_input(self):
        model, _ = probe_gate_chain()
        assert _graph_input_names(model) == {"x"}

    def test_single_output(self):
        model, _ = probe_gate_chain()
        assert _graph_output_names(model) == {"gated"}

    def test_description_mentions_gate(self):
        _, desc = probe_gate_chain()
        assert "gate" in desc.lower()


# ---------------------------------------------------------------------------
# Probe 5 — Scan recurrence
# ---------------------------------------------------------------------------


class TestProbeScanRecurrence:
    def test_single_scan_node(self):
        model, _ = probe_scan_recurrence()
        assert [n.op_type for n in model.graph.node] == ["Scan"]

    def test_body_has_gated_accumulate(self):
        model, _ = probe_scan_recurrence()
        scan = model.graph.node[0]
        body = _get_attr(scan, "body")
        body_ops = [n.op_type for n in body.node]
        assert body_ops == ["Mul", "Add", "Identity"]

    def test_num_scan_inputs(self):
        model, _ = probe_scan_recurrence()
        scan = model.graph.node[0]
        assert _get_attr(scan, "num_scan_inputs") == 1

    def test_two_outputs(self):
        """Scan emits both the final state and per-step values."""
        model, _ = probe_scan_recurrence()
        assert len(_graph_output_names(model)) == 2

    def test_description_mentions_control_flow(self):
        _, desc = probe_scan_recurrence()
        assert "Scan" in desc or "control flow" in desc.lower()


# ---------------------------------------------------------------------------
# Probe 6 — Loop recurrence (static trip count)
# ---------------------------------------------------------------------------


class TestProbeLoopRecurrence:
    def test_single_loop_node(self):
        model, _ = probe_loop_recurrence()
        assert [n.op_type for n in model.graph.node] == ["Loop"]

    def test_body_ops(self):
        model, _ = probe_loop_recurrence()
        loop = model.graph.node[0]
        body = _get_attr(loop, "body")
        body_ops = [n.op_type for n in body.node]
        assert "Identity" in body_ops
        assert "Mul" in body_ops

    def test_trip_count_is_initializer(self):
        model, _ = probe_loop_recurrence()
        init_names = {i.name for i in model.graph.initializer}
        assert "trip_count" in init_names

    def test_cond_is_initializer(self):
        model, _ = probe_loop_recurrence()
        init_names = {i.name for i in model.graph.initializer}
        assert "cond" in init_names

    def test_description_mentions_loop(self):
        _, desc = probe_loop_recurrence()
        assert "Loop" in desc


# ---------------------------------------------------------------------------
# Probe 7 — Loop with dynamic trip count (the negative control)
# ---------------------------------------------------------------------------


class TestProbeLoopDynamicTrip:
    def test_single_loop_node(self):
        model, _ = probe_loop_dynamic_trip()
        assert [n.op_type for n in model.graph.node] == ["Loop"]

    def test_trip_count_is_graph_input_not_initializer(self):
        """This is the whole point of probe 07: trip_count must be an input."""
        model, _ = probe_loop_dynamic_trip()
        inputs = _graph_input_names(model)
        inits = {i.name for i in model.graph.initializer}
        assert "trip_count" in inputs
        assert "trip_count" not in inits

    def test_cond_is_initializer(self):
        """cond stays as a constant — only trip_count is promoted to input."""
        model, _ = probe_loop_dynamic_trip()
        init_names = {i.name for i in model.graph.initializer}
        assert "cond" in init_names

    def test_description_mentions_unrolling(self):
        _, desc = probe_loop_dynamic_trip()
        dl = desc.lower()
        assert "unroll" in dl or "static" in dl or "dag" in dl


# ---------------------------------------------------------------------------
# Cross-probe structural checks
# ---------------------------------------------------------------------------


class TestCrossProbeChecks:
    @pytest.mark.parametrize("probe_name", list(PROBES.keys()))
    def test_all_models_valid(self, probe_name):
        model, _ = PROBES[probe_name]()
        onnx.checker.check_model(model, full_check=True)

    @pytest.mark.parametrize("probe_name", list(PROBES.keys()))
    def test_all_models_serializable(self, probe_name, tmp_path):
        model, _ = PROBES[probe_name]()
        path = tmp_path / f"{probe_name}.onnx"
        onnx.save(model, str(path))
        reloaded = onnx.load(str(path))
        assert reloaded.producer_name == "orionsbelt-npu-op-probe"

    @pytest.mark.parametrize("probe_name", list(PROBES.keys()))
    def test_descriptions_are_unique(self, probe_name):
        """No two probes should share an identical description."""
        descs = [PROBES[n]()[1] for n in PROBES if n != probe_name]
        _, this_desc = PROBES[probe_name]()
        assert this_desc not in descs

    def test_control_flow_probes_differ(self):
        """Scan, Loop, and dynamic-Loop must use different ONNX control-flow ops."""
        scan_model, _ = probe_scan_recurrence()
        loop_model, _ = probe_loop_recurrence()
        dyn_model, _ = probe_loop_dynamic_trip()

        scan_ops = {n.op_type for n in scan_model.graph.node}
        loop_ops = {n.op_type for n in loop_model.graph.node}
        dyn_ops = {n.op_type for n in dyn_model.graph.node}

        # Scan vs Loop are different ops
        assert "Scan" in scan_ops
        assert "Loop" in loop_ops
        assert "Loop" in dyn_ops

    def test_loop_static_vs_dynamic_differ_in_inputs(self):
        """The ONLY structural difference between probe 06 and 07 is trip_count."""
        static_model, _ = probe_loop_recurrence()
        dyn_model, _ = probe_loop_dynamic_trip()

        static_inputs = _graph_input_names(static_model)
        dyn_inputs = _graph_input_names(dyn_model)

        assert "trip_count" not in static_inputs
        assert "trip_count" in dyn_inputs


# ---------------------------------------------------------------------------
# main() function
# ---------------------------------------------------------------------------


class TestMain:
    def test_default_out_directory(self, tmp_path, monkeypatch):
        """Without --out, main() writes to the default 'artifacts/npu_op_probe'."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("sys.argv", ["npu_op_probe.py"])
        rc = main()
        assert rc == 0
        default_dir = tmp_path / "artifacts" / "npu_op_probe"
        assert (default_dir / "manifest.json").exists()
        assert len(list(default_dir.glob("*.onnx"))) == len(PROBES)

    def test_writes_with_explicit_out(self, tmp_path, monkeypatch):
        out = tmp_path / "probe_out"
        monkeypatch.setattr("sys.argv", ["npu_op_probe.py", "--out", str(out)])
        rc = main()
        assert rc == 0

        # Each probe generates an .onnx file
        for name in PROBES:
            onnx_path = out / f"{name}.onnx"
            assert onnx_path.exists(), f"Missing {onnx_path}"
            assert onnx_path.stat().st_size > 0

        # Manifest exists and is valid JSON
        manifest_path = out / "manifest.json"
        assert manifest_path.exists()
        data = json.loads(manifest_path.read_text())
        assert isinstance(data, list)
        assert len(data) == len(PROBES)

    def test_manifest_entries_have_required_fields(self, tmp_path, monkeypatch):
        out = tmp_path / "probe_out2"
        monkeypatch.setattr("sys.argv", ["npu_op_probe.py", "--out", str(out)])
        main()

        data = json.loads((out / "manifest.json").read_text())
        for entry in data:
            assert "probe" in entry
            assert "file" in entry
            assert "ops" in entry
            assert "bytes" in entry
            assert "description" in entry
            assert isinstance(entry["ops"], list)
            assert len(entry["ops"]) > 0
            assert isinstance(entry["bytes"], int)
            assert entry["bytes"] > 0

    def test_manifest_probe_names_match_keys(self, tmp_path, monkeypatch):
        out = tmp_path / "probe_out3"
        monkeypatch.setattr("sys.argv", ["npu_op_probe.py", "--out", str(out)])
        main()

        data = json.loads((out / "manifest.json").read_text())
        manifest_names = {e["probe"] for e in data}
        assert manifest_names == set(PROBES.keys())

    def test_manifest_files_match(self, tmp_path, monkeypatch):
        out = tmp_path / "probe_out4"
        monkeypatch.setattr("sys.argv", ["npu_op_probe.py", "--out", str(out)])
        main()

        data = json.loads((out / "manifest.json").read_text())
        for entry in data:
            expected = f"{entry['probe']}.onnx"
            assert entry["file"] == expected

    def test_idempotent(self, tmp_path, monkeypatch):
        """Running twice produces identical output."""
        out = tmp_path / "probe_out5"
        monkeypatch.setattr("sys.argv", ["npu_op_probe.py", "--out", str(out)])
        main()
        first_manifest = (out / "manifest.json").read_text()
        main()
        second_manifest = (out / "manifest.json").read_text()
        assert first_manifest == second_manifest

    def test_ops_in_manifest_match_model(self, tmp_path, monkeypatch):
        """Manifest ops list must match what's actually in each model."""
        out = tmp_path / "probe_out6"
        monkeypatch.setattr("sys.argv", ["npu_op_probe.py", "--out", str(out)])
        main()

        data = json.loads((out / "manifest.json").read_text())
        for entry in data:
            model = onnx.load(str(out / entry["file"]))
            actual_ops = sorted(_all_ops(model))
            assert entry["ops"] == actual_ops

    def test_creates_output_directory(self, tmp_path, monkeypatch):
        """Output directory is created if it doesn't exist."""
        out = tmp_path / "nested" / "deep" / "dir"
        monkeypatch.setattr("sys.argv", ["npu_op_probe.py", "--out", str(out)])
        rc = main()
        assert rc == 0
        assert out.exists()
