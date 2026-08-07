// SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
// SPDX-License-Identifier: Apache-2.0
//
// Origin: src/orionsbelt/engines/cpu/kernels/gdn_delta_matmul.c (OrionsBelt repository)
//
// KleidiAI micro-kernel: NEON-only GEMV for the M=1 decode-path delta-rule matmul.
//
//   C[j] = sum_k  A[k] * B[k][j]     (M=1 matrix-vector product)
//
// WHY NOT KLEIDIAI'S PACKED GEMM?
//   KleidiAI's packed-GEMM wins 3.1-3.6x at prefill (M>=64), even after
//   per-call RHS repack.  But at decode (M=1) the recurrent state S changes
//   every chunk — packing cannot be amortised — and the 7-126 us repack cost
//   exceeds the matmul itself.  Hand-NEON without packing wins 2.6-3.0x at M=1.
//   Measured break-even M is 3-6 on Cortex-A76 (RK3588).
//
// The kernel accumulates in a 4-wide fp32 NEON register across the K
// reduction dimension using FMA.  An SVE path (vector-length-agnostic,
// predicated tails) is included for cores with SVE; a scalar fallback ensures
// correctness on any architecture.

#include <stddef.h>
#include <string.h> /* memset */

#include "kai_gdn_gemv_f32_f32_f32_1x4_neon_dotprod.h"

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

void kai_run_gdn_gemv_f32_f32_f32_1x4_neon_dotprod(const float *a,
                                                    const float *b, float *c,
                                                    size_t k, size_t n) {
#ifdef __ARM_FEATURE_SVE
    {
        unsigned vl = (unsigned)svcntw();
        for (size_t j0 = 0; j0 < n; j0 += vl) {
            svbool_t pg = svwhilelt_b32((unsigned)j0, (unsigned)n);
            svfloat32_t acc = svdup_f32(0.0f);
            for (size_t kk = 0; kk < k; ++kk) {
                svfloat32_t a_scalar = svdup_f32(a[kk]);
                svfloat32_t b_vec = svld1_f32(pg, b + kk * n + j0);
                acc = svmla_f32_x(pg, acc, b_vec, a_scalar);
            }
            svst1_f32(pg, c + j0, acc);
        }
    }
#elif defined(__ARM_NEON)
    /* Zero the output, then accumulate A[k]*B[k][:] across the reduction
     * dimension with 4-wide NEON FMA.  No packing — at M=1 the repack cost
     * dominates, so a direct accumulation is faster (2.6-3.0x measured). */
    memset(c, 0, n * sizeof(float));
    for (size_t kk = 0; kk < k; ++kk) {
        float32x4_t a_vec = vdupq_n_f32(a[kk]);
        size_t j = 0;
        for (; j + 4 <= n; j += 4) {
            float32x4_t c_vec = vld1q_f32(c + j);
            float32x4_t b_vec = vld1q_f32(b + kk * n + j);
            vst1q_f32(c + j, vfmaq_f32(c_vec, a_vec, b_vec));
        }
        for (; j < n; ++j) {
            c[j] += a[kk] * b[kk * n + j];
        }
    }
#else
    for (size_t j = 0; j < n; ++j) {
        float acc = 0.0f;
        for (size_t kk = 0; kk < k; ++kk) {
            acc += a[kk] * b[kk * n + j];
        }
        c[j] = acc;
    }
#endif
}
