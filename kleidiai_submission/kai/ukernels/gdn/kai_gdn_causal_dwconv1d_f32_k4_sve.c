// SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
// SPDX-License-Identifier: Apache-2.0
//
// Origin: src/orionsbelt/engines/cpu/kernels/gdn_sve.c (OrionsBelt repository)
//
// KleidiAI micro-kernel: Causal depthwise Conv1D, kernel width 4.
//
//   out[t][c] = w[0][c]*h0 + w[1][c]*h1 + w[2][c]*h2 + w[3][c]*in[t][c]
//
// Causal by construction: output t reads no input beyond t.  hist[] holds the
// K-1 = 3 previous timesteps per channel so decode can advance one token at a
// time — the conv-state analogue of a KV cache (3 * channels floats).
//
// The conv's 4-deep FMA chain (acc = h0*w0 + h1*w1 + h2*w2 + cur*w3) benefits
// MORE from ILP than gated_scan's single FMA: in the NEON double-width path
// two independent chains let the OoO scheduler hide the full FMA latency.

#include <stddef.h>

#include "kai_gdn_causal_dwconv1d_f32_k4_sve.h"

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#define KAI_GDN_CONV_K 4

void kai_run_gdn_causal_dwconv1d_f32_k4_sve(const float *in, const float *w,
                                            float *out, float *hist, size_t seq,
                                            size_t channels) {
#ifdef __ARM_FEATURE_SVE
    {
        unsigned vl = (unsigned)svcntw();
        size_t n_vec = (channels + vl - 1) / vl;
        for (size_t vi = 0; vi < n_vec; ++vi) {
            size_t c = vi * vl;
            svbool_t pg = svwhilelt_b32((unsigned)c, (unsigned)channels);
            svfloat32_t h0 = svld1_f32(pg, hist + 0 * channels + c);
            svfloat32_t h1 = svld1_f32(pg, hist + 1 * channels + c);
            svfloat32_t h2 = svld1_f32(pg, hist + 2 * channels + c);
            svfloat32_t w0 = svld1_f32(pg, w + 0 * channels + c);
            svfloat32_t w1 = svld1_f32(pg, w + 1 * channels + c);
            svfloat32_t w2 = svld1_f32(pg, w + 2 * channels + c);
            svfloat32_t w3 = svld1_f32(pg, w + 3 * channels + c);
            for (size_t t = 0; t < seq; ++t) {
                svfloat32_t cur = svld1_f32(pg, in + t * channels + c);
                svfloat32_t acc = svmul_f32_x(pg, h0, w0);
                acc = svmla_f32_x(pg, acc, h1, w1);
                acc = svmla_f32_x(pg, acc, h2, w2);
                acc = svmla_f32_x(pg, acc, cur, w3);
                svst1_f32(pg, out + t * channels + c, acc);
                h0 = h1;
                h1 = h2;
                h2 = cur;
            }
            svst1_f32(pg, hist + 0 * channels + c, h0);
            svst1_f32(pg, hist + 1 * channels + c, h1);
            svst1_f32(pg, hist + 2 * channels + c, h2);
        }
    }
#elif defined(__ARM_NEON)
    /* Double-width unroll: 8 channels/iter (2 NEON register groups).
     * The conv's 4-deep FMA chain benefits MORE from ILP than gated_scan's
     * single FMA: two independent chains let the OoO scheduler hide the
     * 4-cycle FMA latency. */
    size_t n_vec8 = channels >> 3;
    for (size_t vi = 0; vi < n_vec8; ++vi) {
        size_t c = vi << 3;
        float32x4_t h0_0 = vld1q_f32(hist + 0 * channels + c);
        float32x4_t h0_1 = vld1q_f32(hist + 0 * channels + c + 4);
        float32x4_t h1_0 = vld1q_f32(hist + 1 * channels + c);
        float32x4_t h1_1 = vld1q_f32(hist + 1 * channels + c + 4);
        float32x4_t h2_0 = vld1q_f32(hist + 2 * channels + c);
        float32x4_t h2_1 = vld1q_f32(hist + 2 * channels + c + 4);
        float32x4_t w0_0 = vld1q_f32(w + 0 * channels + c);
        float32x4_t w0_1 = vld1q_f32(w + 0 * channels + c + 4);
        float32x4_t w1_0 = vld1q_f32(w + 1 * channels + c);
        float32x4_t w1_1 = vld1q_f32(w + 1 * channels + c + 4);
        float32x4_t w2_0 = vld1q_f32(w + 2 * channels + c);
        float32x4_t w2_1 = vld1q_f32(w + 2 * channels + c + 4);
        float32x4_t w3_0 = vld1q_f32(w + 3 * channels + c);
        float32x4_t w3_1 = vld1q_f32(w + 3 * channels + c + 4);
        for (size_t t = 0; t < seq; ++t) {
            float32x4_t cur0 = vld1q_f32(in + t * channels + c);
            float32x4_t cur1 = vld1q_f32(in + t * channels + c + 4);
            float32x4_t acc0 = vmulq_f32(h0_0, w0_0);
            float32x4_t acc1 = vmulq_f32(h0_1, w0_1);
            acc0 = vfmaq_f32(acc0, h1_0, w1_0);
            acc1 = vfmaq_f32(acc1, h1_1, w1_1);
            acc0 = vfmaq_f32(acc0, h2_0, w2_0);
            acc1 = vfmaq_f32(acc1, h2_1, w2_1);
            acc0 = vfmaq_f32(acc0, cur0, w3_0);
            acc1 = vfmaq_f32(acc1, cur1, w3_1);
            vst1q_f32(out + t * channels + c, acc0);
            vst1q_f32(out + t * channels + c + 4, acc1);
            h0_0 = h1_0; h1_0 = h2_0; h2_0 = cur0;
            h0_1 = h1_1; h1_1 = h2_1; h2_1 = cur1;
        }
        vst1q_f32(hist + 0 * channels + c, h0_0);
        vst1q_f32(hist + 0 * channels + c + 4, h0_1);
        vst1q_f32(hist + 1 * channels + c, h1_0);
        vst1q_f32(hist + 1 * channels + c + 4, h1_1);
        vst1q_f32(hist + 2 * channels + c, h2_0);
        vst1q_f32(hist + 2 * channels + c + 4, h2_1);
    }
    /* Remaining groups of 4 (at most 1) */
    for (size_t c = n_vec8 << 3; c + 4 <= channels; c += 4) {
        float32x4_t h0 = vld1q_f32(hist + 0 * channels + c);
        float32x4_t h1 = vld1q_f32(hist + 1 * channels + c);
        float32x4_t h2 = vld1q_f32(hist + 2 * channels + c);
        float32x4_t w0 = vld1q_f32(w + 0 * channels + c);
        float32x4_t w1 = vld1q_f32(w + 1 * channels + c);
        float32x4_t w2 = vld1q_f32(w + 2 * channels + c);
        float32x4_t w3 = vld1q_f32(w + 3 * channels + c);
        for (size_t t = 0; t < seq; ++t) {
            float32x4_t cur = vld1q_f32(in + t * channels + c);
            float32x4_t acc = vmulq_f32(h0, w0);
            acc = vfmaq_f32(acc, h1, w1);
            acc = vfmaq_f32(acc, h2, w2);
            acc = vfmaq_f32(acc, cur, w3);
            vst1q_f32(out + t * channels + c, acc);
            h0 = h1;
            h1 = h2;
            h2 = cur;
        }
        vst1q_f32(hist + 0 * channels + c, h0);
        vst1q_f32(hist + 1 * channels + c, h1);
        vst1q_f32(hist + 2 * channels + c, h2);
    }
    for (size_t c = (n_vec8 << 3) + (((channels >> 2) & 1) << 2); c < channels; ++c) {
        float hh[KAI_GDN_CONV_K - 1];
        for (int j = 0; j < KAI_GDN_CONV_K - 1; ++j)
            hh[j] = hist[(size_t)j * channels + c];
        for (size_t t = 0; t < seq; ++t) {
            float cur = in[t * channels + c];
            float acc = hh[0] * w[0 * channels + c] + hh[1] * w[1 * channels + c] +
                        hh[2] * w[2 * channels + c] + cur * w[3 * channels + c];
            out[t * channels + c] = acc;
            hh[0] = hh[1];
            hh[1] = hh[2];
            hh[2] = cur;
        }
        for (int j = 0; j < KAI_GDN_CONV_K - 1; ++j)
            hist[(size_t)j * channels + c] = hh[j];
    }
#else
    for (size_t c = 0; c < channels; ++c) {
        float h[KAI_GDN_CONV_K - 1];
        for (int j = 0; j < KAI_GDN_CONV_K - 1; ++j)
            h[j] = hist[(size_t)j * channels + c];
        for (size_t t = 0; t < seq; ++t) {
            float cur = in[t * channels + c];
            float acc = h[0] * w[0 * channels + c] + h[1] * w[1 * channels + c] +
                        h[2] * w[2 * channels + c] + cur * w[3 * channels + c];
            out[t * channels + c] = acc;
            h[0] = h[1];
            h[1] = h[2];
            h[2] = cur;
        }
        for (int j = 0; j < KAI_GDN_CONV_K - 1; ++j)
            hist[(size_t)j * channels + c] = h[j];
    }
#endif
}
