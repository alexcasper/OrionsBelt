/* Gated DeltaNet delta-rule matmul: beta = alpha . S
 *
 * Bead ob-8qt.1. The one piece of the delta-rule update the other three GDN
 * kernels in this directory (gdn_sve.c: cumdecay, gated_scan, causal_dwconv1d)
 * don't cover -- a small per-chunk dense matmul, K=head_dim=128,
 * N=head_dim*n_heads.
 *
 * DUAL-PATH DISPATCH, per the RK3588 A76 measurement in docs/FINDINGS.md
 * section 8 (bead ob-8qt.2):
 *
 *   - KleidiAI's packed-GEMM (kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla)
 *     wins 3.1-3.6x over hand-NEON at prefill (M>=64, measured), even after
 *     per-call repack cost, because the repack is <1% of the matmul time at
 *     that M.
 *   - Hand-written NEON without packing wins 2.6-3.0x at decode (M=1,
 *     measured), because S changes every chunk -- packing cannot be
 *     amortised across calls the way a static weight matrix would -- and
 *     KleidiAI's 7-126us repack cost exceeds the (tiny) matmul itself.
 *   - The measured break-even M is 3-6 (FINDINGS section 8, "Net comparison").
 *     Only M=1 and M=64 were actually measured; nothing in between. 5 is
 *     picked as the dispatch threshold below as the middle of that measured
 *     range, not as a third measured point -- a real GDN chunk size (64) is
 *     always in the KleidiAI regime and a real decode step (1) is always in
 *     the NEON regime, so the exact threshold placement inside [3,6] doesn't
 *     change behavior for either real workload; it only matters if some
 *     future partial-chunk or speculative-decode path calls this with
 *     2 <= M <= 8.
 *
 * KleidiAI ITSELF IS NOT VENDORED in this repo -- consistent with the
 * "evaluation phase, not yet a submodule" note in FINDINGS section 8's
 * Reproducing steps. The fast path here is therefore compile-time optional,
 * guarded by ORIONSBELT_WITH_KLEIDIAI: unless that macro and the KleidiAI
 * sources are supplied at build time (see the header comment in
 * bench/kleidiai_matmul_bench.c for the exact clone + compile invocation),
 * gdn_delta_rule_matmul degrades to the hand-NEON/SVE path unconditionally.
 * That fallback is CORRECT at every M -- just not fast at large M. No build
 * of this project silently loses correctness for lack of KleidiAI; it only
 * loses the prefill speedup.
 *
 * The kernel selected (f32, plain NEON MLA, no i8mm/dotprod) matches what was
 * actually measured: the RK3588 A76 test device predates i8mm (A78+) and SVE
 * (Neoverse V2+/A720+), and the delta-rule operands are fp32 per the
 * project's quantization policy (docs/QUANTIZATION_POLICY.md FP16 carve-outs
 * apply to the recurrent state, not to this matmul's operands). An int8/i8mm
 * path would need the delta-rule's K and S quantized first, which is a
 * separate, larger decision this bead does not make.
 */

#include "gdn_delta_matmul.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif

#ifdef ORIONSBELT_WITH_KLEIDIAI
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon.h"
#endif

/* Middle of the measured break-even range [3,6] (FINDINGS.md section 8). */
#define GDN_DELTA_MATMUL_KLEIDIAI_MIN_M 5

/* ---------------------------------------------------------------------------
 * Hand-written path: correct at every M, the one actually used at decode.
 * SVE (predicated, vector-length-agnostic) with NEON and scalar fallbacks --
 * same three-way portability posture as the other kernels in gdn_sve.c.
 * ------------------------------------------------------------------------- */
static void gdn_delta_matmul_neon(const float *restrict A, const float *restrict B,
                                  float *restrict C, size_t M, size_t K, size_t N) {
#if defined(__ARM_FEATURE_SVE)
    unsigned vl = (unsigned)svcntw();
    for (size_t i = 0; i < M; ++i) {
        for (size_t j0 = 0; j0 < N; j0 += vl) {
            svbool_t pg = svwhilelt_b32((unsigned)j0, (unsigned)N);
            svfloat32_t acc = svdup_f32(0.0f);
            for (size_t k = 0; k < K; ++k) {
                svfloat32_t a_scalar = svdup_f32(A[i * K + k]);
                svfloat32_t b_vec = svld1_f32(pg, B + k * N + j0);
                acc = svmla_f32_x(pg, acc, b_vec, a_scalar);
            }
            svst1_f32(pg, C + i * N + j0, acc);
        }
    }
#elif defined(__ARM_NEON)
    for (size_t i = 0; i < M; ++i) {
        memset(C + i * N, 0, N * sizeof(float));
        for (size_t k = 0; k < K; ++k) {
            float32x4_t a_vec = vdupq_n_f32(A[i * K + k]);
            size_t j = 0;
            for (; j + 4 <= N; j += 4) {
                float32x4_t c_vec = vld1q_f32(C + i * N + j);
                float32x4_t b_vec = vld1q_f32(B + k * N + j);
                vst1q_f32(C + i * N + j, vfmaq_f32(c_vec, a_vec, b_vec));
            }
            for (; j < N; ++j) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
#else
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (size_t k = 0; k < K; ++k) acc += A[i * K + k] * B[k * N + j];
            C[i * N + j] = acc;
        }
    }
#endif
}

#ifdef ORIONSBELT_WITH_KLEIDIAI
/* KleidiAI path: repack B every call (S changes every chunk -- see the file
 * header, this cost is exactly what FINDINGS section 8 measured and it is
 * why this path is only worth it at M>=GDN_DELTA_MATMUL_KLEIDIAI_MIN_M). */
static void gdn_delta_matmul_kleidiai(const float *restrict A, const float *restrict B,
                                      float *restrict C, size_t M, size_t K, size_t N) {
    const size_t nr = kai_get_nr_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla();
    const size_t kr = kai_get_kr_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla();
    const size_t sr = kai_get_sr_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla();
    const size_t packed_size =
        kai_get_rhs_packed_size_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon(N, K);

    void *packed = aligned_alloc(64, packed_size);
    float *zero_bias = calloc(N, sizeof(float)); /* delta-rule matmul has no bias */

    kai_run_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon(
        /*num_groups=*/1, N, K, nr, kr, sr, N * sizeof(float), B, zero_bias,
        /*scale=*/NULL, packed, /*extra_bytes=*/0, /*params=*/NULL);

    kai_run_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla(
        M, N, K, A, K * sizeof(float), packed, C, N * sizeof(float), sizeof(float),
        -INFINITY, INFINITY);

    free(zero_bias);
    free(packed);
}
#endif

void gdn_delta_rule_matmul(const float *restrict A, const float *restrict B,
                           float *restrict C, size_t M, size_t K, size_t N) {
#ifdef ORIONSBELT_WITH_KLEIDIAI
    if (M >= GDN_DELTA_MATMUL_KLEIDIAI_MIN_M) {
        gdn_delta_matmul_kleidiai(A, B, C, M, K, N);
        return;
    }
#endif
    gdn_delta_matmul_neon(A, B, C, M, K, N);
}
