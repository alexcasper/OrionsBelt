# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the GDN layer-info module (ob-37v).

These test the figures that downstream beads (memory instrumentation,
quantization policy, NOE op-coverage) will rely on — all verified against
config.json and modeling source in the audit doc.
"""

import pytest

from orionsbelt.model.gdn_layer_info import LAYER_INFO


class TestQwen35_4B:
    """Primary checkpoint per ADR 0003."""

    info = LAYER_INFO["4B"]

    # --- layer counts ---
    def test_layer_counts(self):
        assert self.info.num_hidden_layers == 32
        assert self.info.num_gdn_layers == 24
        assert self.info.num_full_attention_layers == 8
        assert self.info.num_gdn_layers + self.info.num_full_attention_layers == 32

    def test_layer_pattern(self):
        assert self.info.layer_types[3] == "full_attention"
        assert self.info.layer_types[7] == "full_attention"
        assert self.info.layer_types[31] == "full_attention"
        assert self.info.layer_types[0] == "linear_attention"
        assert self.info.full_attention_layer_indices == [3, 7, 11, 15, 19, 23, 27, 31]

    # --- GDN dims ---
    def test_key_value_dims(self):
        assert self.info.key_dim == 2048  # 128 * 16
        assert self.info.value_dim == 4096  # 128 * 32
        assert self.info.conv_dim == 8192  # 2048*2 + 4096
        assert self.info.kv_head_ratio == 2  # 32 // 16

    def test_recurrent_state_shape(self):
        assert self.info.recurrent_state_shape == (32, 128, 128)

    def test_recurrent_state_per_layer(self):
        assert self.info.recurrent_state_elements_per_layer == 524_288

    def test_recurrent_state_total_fp32(self):
        """24 layers * 524288 * 4 bytes = 48 MiB."""
        assert self.info.recurrent_state_total_mib(dtype_size=4) == pytest.approx(48.0, abs=0.1)

    def test_recurrent_state_total_fp16(self):
        assert self.info.recurrent_state_total_mib(dtype_size=2) == pytest.approx(24.0, abs=0.1)

    # --- conv state ---
    def test_conv_state_per_layer(self):
        assert self.info.conv_state_elements_per_layer == 32_768  # 8192 * 4

    # --- KV cache ---
    def test_kv_cache_per_token_bytes(self):
        # 4 heads * 256 head_dim * 2 (K+V) * 2 bytes (FP16)
        assert self.info.kv_cache_bytes_per_token(dtype_size=2) == 4096

    def test_kv_cache_at_4k(self):
        """8 FA layers, 4KV heads, 256 head_dim, 2 (K+V), FP16."""
        assert self.info.kv_cache_mib_at_context(4096, dtype_size=2) == pytest.approx(
            128.0, abs=1.0
        )

    def test_kv_cache_at_262k(self):
        """At 262K context, KV cache is ~8 GiB in FP16."""
        assert self.info.kv_cache_mib_at_context(262144, dtype_size=2) == pytest.approx(
            8192.0, abs=10.0
        )

    def test_kv_vs_recurrent_ratio(self):
        """The central claim: at 262K, KV cache dwarfs recurrent state."""
        kv = self.info.kv_cache_mib_at_context(262144, dtype_size=2)
        gdn = self.info.recurrent_state_total_mib(dtype_size=4)
        ratio = kv / gdn
        assert ratio > 100  # expected ~170x


class TestQwen35_0_8B:
    """Fallback checkpoint per ADR 0003."""

    info = LAYER_INFO["0.8B"]

    def test_layer_counts(self):
        assert self.info.num_hidden_layers == 24
        assert self.info.num_gdn_layers == 18
        assert self.info.num_full_attention_layers == 6

    def test_recurrent_state_shape(self):
        assert self.info.recurrent_state_shape == (16, 128, 128)

    def test_recurrent_state_per_layer(self):
        assert self.info.recurrent_state_elements_per_layer == 262_144

    def test_recurrent_state_total_fp32(self):
        """18 layers * 262144 * 4 bytes = 18 MiB."""
        assert self.info.recurrent_state_total_mib(dtype_size=4) == pytest.approx(18.0, abs=0.1)

    def test_kv_head_ratio_is_one(self):
        """0.8B has equal key and value heads — no replication."""
        assert self.info.kv_head_ratio == 1

    def test_kv_cache_at_4k(self):
        """6 FA layers x (K+V) x 2 KV heads x 256 head_dim x 4096 tokens x 2 bytes = 48 MiB.

        Corrected 2026-08-03: the expectation was 24.0 MiB, which omits the factor of 2
        for storing both K and V. Verified independently against the arithmetic above;
        the implementation's 48.0 MiB is correct and the test was wrong.
        """
        assert self.info.kv_cache_mib_at_context(4096, dtype_size=2) == pytest.approx(48.0, abs=1.0)
