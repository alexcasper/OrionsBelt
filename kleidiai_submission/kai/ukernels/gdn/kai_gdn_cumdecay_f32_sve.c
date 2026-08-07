// SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
// SPDX-License-Identifier: Apache-2.0
//
// Origin: src/orionsbelt/engines/cpu/kernels/gdn_sve.c (OrionsBelt repository)
//
// KleidiAI micro-kernel: Gated cumulative decay (inclusive prefix product).
//
// THE LAYOUT DECISION
// -------------------
// A prefix scan along the *vector lanes* needs a log-depth Hillis-Steele
// shuffle network.  We never do that.  GDN's sequence axis is inherently
// sequential, so we vectorize across the CHANNEL/HEAD axis and walk the
// sequence with a plain loop.  The kernel is then a sequence of independent
// lane-wise multiplies — no cross-lane communication anywhere.
//
// ISA FLOOR IS SVE1, NOT SVE2.  Every intrinsic used here — svcntw, svdup_f32,
// svld1_f32, svmul_f32_x, svst1_f32, svwhilelt_b32 — is base SVE, and the guard
// is __ARM_FEATURE_SVE.  Nothing in this fp32 kernel needs SVE2's integer/DSP
// additions.
//
// NEON PATH: double-width unroll — 8 channels/iter (two 4-wide register
// groups) hides MUL latency and doubles memory-level parallelism.

#include <stddef.h>

#include "kai_gdn_cumdecay_f32_sve.h"

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

void kai_run_gdn_cumdecay_f32_sve(const float *a, float *decay, size_t seq,
                                  size_t channels) {
#ifdef __ARM_FEATURE_SVE
    {
        unsigned vl = (unsigned)svcntw();
        size_t n_vec = (channels + vl - 1) / vl;
        for (size_t vi = 0; vi < n_vec; ++vi) {
            size_t c = vi * vl;
            svbool_t pg = svwhilelt_b32((unsigned)c, (unsigned)channels);
            svfloat32_t run = svdup_f32(1.0f);
            for (size_t t = 0; t < seq; ++t) {
                svfloat32_t av = svld1_f32(pg, a + t * channels + c);
                run = svmul_f32_x(pg, run, av);
                svst1_f32(pg, decay + t * channels + c, run);
            }
        }
    }
#elif defined(__ARM_NEON)
    /* Double-width unroll: 8 channels/iter (2 NEON register groups) hides
     * MUL latency and doubles memory-level parallelism. */
    size_t n_vec8 = channels >> 3;
    for (size_t vi = 0; vi < n_vec8; ++vi) {
        size_t c = vi << 3;
        float32x4_t run0 = vdupq_n_f32(1.0f);
        float32x4_t run1 = vdupq_n_f32(1.0f);
        for (size_t t = 0; t < seq; ++t) {
            run0 = vmulq_f32(run0, vld1q_f32(a + t * channels + c));
            run1 = vmulq_f32(run1, vld1q_f32(a + t * channels + c + 4));
            vst1q_f32(decay + t * channels + c, run0);
            vst1q_f32(decay + t * channels + c + 4, run1);
        }
    }
    /* Remaining groups of 4 (at most 1) */
    for (size_t c = n_vec8 << 3; c + 4 <= channels; c += 4) {
        float32x4_t run = vdupq_n_f32(1.0f);
        for (size_t t = 0; t < seq; ++t) {
            run = vmulq_f32(run, vld1q_f32(a + t * channels + c));
            vst1q_f32(decay + t * channels + c, run);
        }
    }
    for (size_t c = (n_vec8 << 3) + (((channels >> 2) & 1) << 2); c < channels; ++c) {
        float run = 1.0f;
        for (size_t t = 0; t < seq; ++t) {
            run *= a[t * channels + c];
            decay[t * channels + c] = run;
        }
    }
#else
    for (size_t c = 0; c < channels; ++c) {
        float run = 1.0f;
        for (size_t t = 0; t < seq; ++t) {
            run *= a[t * channels + c];
            decay[t * channels + c] = run;
        }
    }
#endif
}
