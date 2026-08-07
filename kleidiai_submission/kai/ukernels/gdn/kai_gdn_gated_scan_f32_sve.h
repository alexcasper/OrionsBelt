// SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
// SPDX-License-Identifier: Apache-2.0
//
// Origin: src/orionsbelt/engines/cpu/kernels/gdn_sve.c (OrionsBelt repository)
//
// KleidiAI micro-kernel: Chunkwise gated delta-rule scan.
//
// Computes:  s[t][c] = g[t][c] * s[t-1][c] + x[t][c]
//
// The recurrence the NPU cannot express.  state[] carries across calls
// (cross-invocation continuity for single-token decode).

#ifndef KAI_GDN_GATED_SCAN_F32_SVE_H
#define KAI_GDN_GATED_SCAN_F32_SVE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Chunkwise gated scan (first-order linear recurrence).
///
/// For each channel c, computes s[t][c] = g[t][c] * s[t-1][c] + x[t][c]
/// for t = 0 .. seq-1.  The initial state s[-1] is read from state[], and the
/// final state is written back to state[], enabling cross-call continuity.
///
/// @param g        Gate factors,    layout [seq][channels], row-major.
/// @param x        Input values,    layout [seq][channels], row-major.
/// @param s        Output state,    layout [seq][channels], row-major.
/// @param state    Persistent recurrent state (channels elements), read on
///                 entry as s[-1] and updated on exit to the final s[seq-1].
/// @param seq      Number of timesteps (sequence length).
/// @param channels Number of channels (vectorization axis).
void kai_run_gdn_gated_scan_f32_sve(const float *g, const float *x, float *s,
                                    float *state, size_t seq, size_t channels);

#ifdef __cplusplus
}
#endif

#endif /* KAI_GDN_GATED_SCAN_F32_SVE_H */
