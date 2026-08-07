// SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
// SPDX-License-Identifier: Apache-2.0
//
// Origin: src/orionsbelt/engines/cpu/kernels/gdn_sve.c (OrionsBelt repository)
//
// KleidiAI micro-kernel: Chunkwise gated delta-rule scan.
//
//   s[t][c] = g[t][c] * s[t-1][c] + x[t][c]
//
// This is the outer, sequential half of the chunkwise GDN formulation,
// deliberately kept separate from the per-chunk dense math so the mapping
// layer can move the inner matmuls to GPU/NPU while this stays on the CPU.
// state[] carries across calls — exactly the cross-invocation continuity
// the model serving runtime needs for single-token decode.
//
// The scan is a single FMA per element (acc = x + acc*g).  Vectorized across
// the channel axis: SVE1 baseline with predicated tails, NEON double-width
// unroll (8 channels/iter), scalar fallback.

#include <stddef.h>

#include "kai_gdn_gated_scan_f32_sve.h"

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

void kai_run_gdn_gated_scan_f32_sve(const float *g, const float *x, float *s,
                                    float *state, size_t seq, size_t channels) {
#ifdef __ARM_FEATURE_SVE
    {
        unsigned vl = (unsigned)svcntw();
        size_t n_vec = (channels + vl - 1) / vl;
        for (size_t vi = 0; vi < n_vec; ++vi) {
            size_t c = vi * vl;
            svbool_t pg = svwhilelt_b32((unsigned)c, (unsigned)channels);
            svfloat32_t acc = svld1_f32(pg, state + c);
            for (size_t t = 0; t < seq; ++t) {
                svfloat32_t gv = svld1_f32(pg, g + t * channels + c);
                svfloat32_t xv = svld1_f32(pg, x + t * channels + c);
                acc = svmla_f32_x(pg, xv, acc, gv); /* acc = x + acc*g, one FMA */
                svst1_f32(pg, s + t * channels + c, acc);
            }
            svst1_f32(pg, state + c, acc);
        }
    }
#elif defined(__ARM_NEON)
    /* Double-width unroll: 8 channels/iter (2 NEON register groups). */
    size_t n_vec8 = channels >> 3;
    for (size_t vi = 0; vi < n_vec8; ++vi) {
        size_t c = vi << 3;
        float32x4_t acc0 = vld1q_f32(state + c);
        float32x4_t acc1 = vld1q_f32(state + c + 4);
        for (size_t t = 0; t < seq; ++t) {
            acc0 = vfmaq_f32(vld1q_f32(x + t * channels + c), acc0,
                             vld1q_f32(g + t * channels + c));
            acc1 = vfmaq_f32(vld1q_f32(x + t * channels + c + 4), acc1,
                             vld1q_f32(g + t * channels + c + 4));
            vst1q_f32(s + t * channels + c, acc0);
            vst1q_f32(s + t * channels + c + 4, acc1);
        }
        vst1q_f32(state + c, acc0);
        vst1q_f32(state + c + 4, acc1);
    }
    /* Remaining groups of 4 (at most 1) */
    for (size_t c = n_vec8 << 3; c + 4 <= channels; c += 4) {
        float32x4_t acc = vld1q_f32(state + c);
        for (size_t t = 0; t < seq; ++t) {
            acc = vfmaq_f32(vld1q_f32(x + t * channels + c), acc,
                            vld1q_f32(g + t * channels + c));
            vst1q_f32(s + t * channels + c, acc);
        }
        vst1q_f32(state + c, acc);
    }
    for (size_t c = (n_vec8 << 3) + (((channels >> 2) & 1) << 2); c < channels; ++c) {
        float a = state[c];
        for (size_t t = 0; t < seq; ++t) {
            a = x[t * channels + c] + a * g[t * channels + c];
            s[t * channels + c] = a;
        }
        state[c] = a;
    }
#else
    for (size_t c = 0; c < channels; ++c) {
        float acc = state[c];
        for (size_t t = 0; t < seq; ++t) {
            acc = x[t * channels + c] + acc * g[t * channels + c];
            s[t * channels + c] = acc;
        }
        state[c] = acc;
    }
#endif
}
