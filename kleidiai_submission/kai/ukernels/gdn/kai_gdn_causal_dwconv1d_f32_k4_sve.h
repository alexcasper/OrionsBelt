// SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
// SPDX-License-Identifier: Apache-2.0
//
// Origin: src/orionsbelt/engines/cpu/kernels/gdn_sve.c (OrionsBelt repository)
//
// KleidiAI micro-kernel: Causal depthwise Conv1D, kernel width 4.
//
//   out[t][c] = w[0][c]*h0[c] + w[1][c]*h1[c] + w[2][c]*h2[c] + w[3][c]*in[t][c]
//
// KleidiAI's existing depthwise convolution is SME2-only; this kernel provides
// a portable SVE/NEON/scalar path that runs on every AArch64 core.

#ifndef KAI_GDN_CAUSAL_DWCONV1D_F32_K4_SVE_H
#define KAI_GDN_CAUSAL_DWCONV1D_F32_K4_SVE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Causal depthwise Conv1D with kernel width 4.
///
/// For each channel c and timestep t, computes:
///   out[t][c] = w[0][c]*h0 + w[1][c]*h1 + w[2][c]*h2 + w[3][c]*in[t][c]
/// where h0, h1, h2 are the three previous input timesteps for that channel.
///
/// The hist[] buffer holds the K-1 = 3 previous timesteps per channel
/// (3 * channels floats, layout [3][channels]), carried across calls so decode
/// can advance one token at a time — the conv analogue of a KV cache.
///
/// @param in       Input activations, layout [seq][channels], row-major.
/// @param w        Depthwise weights, layout [4][channels], row-major.
/// @param out      Output, layout [seq][channels], row-major.
/// @param hist     History buffer (3 * channels floats), read on entry and
///                 updated on exit to the last 3 input timesteps.
/// @param seq      Number of timesteps (sequence length).
/// @param channels Number of channels (vectorization axis).
void kai_run_gdn_causal_dwconv1d_f32_k4_sve(const float *in, const float *w,
                                            float *out, float *hist, size_t seq,
                                            size_t channels);

#ifdef __cplusplus
}
#endif

#endif /* KAI_GDN_CAUSAL_DWCONV1D_F32_K4_SVE_H */
