// SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
// SPDX-License-Identifier: Apache-2.0
//
// Origin: src/orionsbelt/engines/cpu/kernels/gdn_sve.c (OrionsBelt repository)
//
// KleidiAI micro-kernel: Gated cumulative decay (inclusive prefix product).
//
// Computes:  decay[t][c] = prod_{i=0}^{t} a[i][c]
//
// Vectorized across the channel axis with a running product; the sequence
// axis is walked with a plain loop.  SVE1 baseline (predicated tails via
// svwhilelt), NEON double-width fallback (8 channels/iter), and a scalar
// fallback for portability.

#ifndef KAI_GDN_CUMDECAY_F32_SVE_H
#define KAI_GDN_CUMDECAY_F32_SVE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Gated cumulative decay (inclusive prefix product along the sequence axis).
///
/// For each channel c, computes the running product decay[t][c] = a[0][c] *
/// a[1][c] * ... * a[t][c] for t = 0 .. seq-1.
///
/// @param a        Input gate factors, layout [seq][channels], row-major.
/// @param decay    Output prefix products, layout [seq][channels], row-major.
/// @param seq      Number of timesteps (sequence length).
/// @param channels Number of channels (vectorization axis).
void kai_run_gdn_cumdecay_f32_sve(const float *a, float *decay, size_t seq,
                                  size_t channels);

#ifdef __cplusplus
}
#endif

#endif /* KAI_GDN_CUMDECAY_F32_SVE_H */
