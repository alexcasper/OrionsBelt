/* Correctness test for INT8 weight quantization + GEMV (gdn_e2e_decode.c).
 *
 * The INT8 weight-only quantization path (-DINT8_WEIGHTS) stores projection
 * weights as int8 + per-column float scale, cutting memory traffic 4×.  This
 * test verifies the quantization logic and NEON GEMV against an FP32 reference
 * at the shapes actually used in the decode loop (K=head_dim, N=output dim).
 *
 * What is tested:
 *   1. Per-column scale = max_abs / 127 (symmetric quantization)
 *   2. INT8 GEMV output within theoretical error bound:
 *        |c_int8[n] - c_fp32[n]| <= scale[n] * 0.5 * sum_k(|a[k]|)
 *   3. Mean relative error < 2% over well-conditioned outputs (|c| > 1.0)
 *   4. Scalar tail handling for non-multiple-of-8 N
 *
 * Build:
 *   gcc -O3 -DINT8_WEIGHTS gdn_sve.c gdn_delta_matmul.c test_gdn_e2e_int8.c \
 *       -I. -o test_gdn_e2e_int8 -lm
 *
 * The static functions are accessed by including the e2e decode source with
 * main() renamed out of the way.
 */
#define main main_disabled_e2e
#include "gdn_e2e_decode.c"
#undef main

#include <math.h>

int main(void) {
    int failures = 0;
    size_t K = 128, N = 256;

    printf("INT8 weight quantization + GEMV correctness test\n");
    printf("  K=%zu (head_dim), N=%zu (output dim)\n\n", K, N);

    /* Generate test weight matrix */
    float *W = malloc(K * N * sizeof(float));
    unsigned seed = 42;
    for (size_t i = 0; i < K * N; ++i)
        W[i] = ((float)(rand_r(&seed) % 2000) - 1000) / 1000.0f;

    /* Quantize */
    int8_t *q;
    float *s;
    quantize_weight(W, &q, &s, K, N);

    /* ---- Test 1: per-column scale correctness ---- */
    printf("--- Test 1: per-column scale correctness ---\n");
    {
        int ok = 1;
        for (size_t n = 0; n < N; ++n) {
            float max_abs = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                float v = fabsf(W[k * N + n]);
                if (v > max_abs) max_abs = v;
            }
            float expected = (max_abs > 0.0f) ? max_abs / 127.0f : 1.0f;
            if (fabsf(s[n] - expected) > 1e-6f * fmaxf(expected, 1e-30f)) {
                printf("  FAIL: col %zu scale=%.6e expected=%.6e\n", n, s[n], expected);
                ok = 0;
                break;
            }
        }
        printf("  %s\n\n", ok ? "PASS" : "FAIL");
        failures += !ok;
    }

    /* ---- Test 2: INT8 GEMV within theoretical error bound ---- */
    printf("--- Test 2: INT8 GEMV vs FP32 GEMV (theoretical bound) ---\n");
    {
        float *a = malloc(K * sizeof(float));
        for (size_t k = 0; k < K; ++k)
            a[k] = ((float)(rand_r(&seed) % 2000) - 1000) / 1000.0f;

        float *c_fp32 = malloc(N * sizeof(float));
        float *c_int8 = malloc(N * sizeof(float));

        gemv_neon(a, W, c_fp32, K, N);
        gemv_int8_neon(a, q, s, c_int8, K, N);

        /* Bound: error[n] <= s[n] * 0.5 * sum_k(|a[k]|)
         * Each quantized weight element has at most 0.5 ULP error in q-space,
         * and the GEMV sums K of them with the same scale[n]. */
        double a_abs_sum = 0.0;
        for (size_t k = 0; k < K; ++k)
            a_abs_sum += fabs((double)a[k]);

        int ok = 1;
        double worst_ratio = 0.0;
        for (size_t n = 0; n < N; ++n) {
            double actual = fabs((double)c_int8[n] - c_fp32[n]);
            double bound = (double)s[n] * 0.5 * a_abs_sum;
            /* Allow 2× margin for rounding accumulation */
            double ratio = (bound > 1e-20) ? actual / bound : 0.0;
            if (ratio > worst_ratio) worst_ratio = ratio;
            if (actual > bound * 2.0) {
                printf("  FAIL: col %zu actual=%.4e bound=%.4e ratio=%.2f\n",
                       n, actual, bound, ratio);
                ok = 0;
            }
        }
        printf("  worst actual/bound ratio: %.3f (should be < 2.0)\n", worst_ratio);
        printf("  %s\n\n", ok ? "PASS" : "FAIL");
        failures += !ok;

        /* ---- Test 3: mean relative error over well-conditioned outputs ---- */
        printf("--- Test 3: mean relative error (|c_fp32| > 1.0) ---\n");
        {
            double sum_rel = 0.0;
            size_t n_rel = 0;
            double max_rel = 0.0;
            for (size_t n = 0; n < N; ++n) {
                if (fabs((double)c_fp32[n]) > 1.0) {
                    double r = fabs((double)c_int8[n] - c_fp32[n])
                             / fabs((double)c_fp32[n]);
                    sum_rel += r;
                    if (r > max_rel) max_rel = r;
                    n_rel++;
                }
            }
            double mean_rel = (n_rel > 0) ? sum_rel / (double)n_rel : 0.0;
            printf("  mean_rel=%.4f%%  max_rel=%.4f%%  (n=%zu)\n",
                   mean_rel * 100.0, max_rel * 100.0, n_rel);
            int rel_ok = mean_rel < 0.02;  /* <2% mean relative error */
            printf("  %s\n\n", rel_ok ? "PASS" : "FAIL");
            failures += !rel_ok;
        }

        free(a); free(c_fp32); free(c_int8);
    }

    /* ---- Test 4: scalar tail handling (non-multiple-of-8 N) ---- */
    printf("--- Test 4: non-multiple-of-8 N (tail handling) ---\n");
    {
        size_t K2 = 64, N2 = 130;  /* 130 = 16*8 + 2 */
        float *W2 = malloc(K2 * N2 * sizeof(float));
        float *a2 = malloc(K2 * sizeof(float));
        unsigned s2 = 77;
        for (size_t i = 0; i < K2 * N2; ++i)
            W2[i] = ((float)(rand_r(&s2) % 2000) - 1000) / 1000.0f;
        for (size_t k = 0; k < K2; ++k)
            a2[k] = ((float)(rand_r(&s2) % 2000) - 1000) / 1000.0f;

        int8_t *q2; float *sc2;
        quantize_weight(W2, &q2, &sc2, K2, N2);
        float *cf2 = malloc(N2 * sizeof(float));
        float *ci2 = malloc(N2 * sizeof(float));
        gemv_neon(a2, W2, cf2, K2, N2);
        gemv_int8_neon(a2, q2, sc2, ci2, K2, N2);

        /* Check last 2 elements (scalar tail) have bounded error */
        double a2_sum = 0.0;
        for (size_t k = 0; k < K2; ++k) a2_sum += fabs((double)a2[k]);

        int tail_ok = 1;
        for (size_t n = 0; n < N2; ++n) {
            double actual = fabs((double)ci2[n] - cf2[n]);
            double bound = (double)sc2[n] * 0.5 * a2_sum * 2.0;
            if (actual > bound) {
                printf("  FAIL: col %zu actual=%.4e bound=%.4e\n", n, actual, bound);
                tail_ok = 0;
            }
        }
        printf("  N=%zu (tail=%zu): %s\n\n", N2, N2 % 8, tail_ok ? "PASS" : "FAIL");
        failures += !tail_ok;

        free(W2); free(a2); free(q2); free(sc2); free(cf2); free(ci2);
    }

    /* ---- Test 5: SDOT vs NEON INT8 GEMV correctness (ob-8qt.14) ---- */
    /* When __ARM_FEATURE_DOTPROD is active, the SDOT kernel keeps the entire
     * dot-product in int8×int8→int32 and only converts to float once at the
     * end. The NEON kernel dequantizes every weight to float32 before the FMA.
     * The SDOT path also quantizes the input vector to int8 (single scale),
     * introducing a small additional error vs NEON. This test verifies both
     * kernels agree within that bound. */
#if defined(INT8_WEIGHTS) && defined(__ARM_FEATURE_DOTPROD)
    printf("--- Test 5: SDOT vs NEON INT8 GEMV (dotprod) ---\n");
    {
        /* Use real-ish shapes: K=head_dim, N=larger to exercise vectorized path */
        size_t K5 = 128, N5 = 512;
        float *W5 = malloc(K5 * N5 * sizeof(float));
        unsigned s5 = 99;
        for (size_t i = 0; i < K5 * N5; ++i)
            W5[i] = ((float)(rand_r(&s5) % 2000) - 1000) / 1000.0f;

        int8_t *q5; float *sc5;
        quantize_weight(W5, &q5, &sc5, K5, N5);

        /* Repack for SDOT */
        int8_t *q5_sdot = repack_int8_k_interleaved(q5, K5, N5);

        float *a5 = malloc(K5 * sizeof(float));
        for (size_t k = 0; k < K5; ++k)
            a5[k] = ((float)(rand_r(&s5) % 2000) - 1000) / 1000.0f;

        float *c_neon = malloc(N5 * sizeof(float));
        float *c_sdot = malloc(N5 * sizeof(float));
        float *c_fp32 = malloc(N5 * sizeof(float));

        gemv_neon(a5, W5, c_fp32, K5, N5);
        gemv_int8_neon(a5, q5, sc5, c_neon, K5, N5);
        gemv_int8_sdot(a5, q5_sdot, sc5, c_sdot, K5, N5);

        /* SDOT vs NEON: bound accounts for input-quantization error.
         * Each aq[k] has |a[k] - aq[k]*a_scale| <= 0.5*a_scale, and each
         * int8 weight has |w| <= 127. Over K elements:
         *   |c_neon[n] - c_sdot[n]| <= sc5[n] * 0.5 * a_scale * 128 * K5
         * (very conservative — actual errors are far smaller). */
        float a_scale5;
        {
            float max_abs = 0.0f;
            for (size_t k = 0; k < K5; ++k) {
                float v = fabsf(a5[k]);
                if (v > max_abs) max_abs = v;
            }
            a_scale5 = (max_abs > 0.0f) ? max_abs / 127.0f : 1.0f;
        }
        double neon_sdot_bound = (double)a_scale5 * 0.5 * 128.0 * K5;

        int sdot_ok = 1;
        double worst_ns = 0.0;
        for (size_t n = 0; n < N5; ++n) {
            double actual = fabs((double)c_neon[n] - c_sdot[n]);
            double bound = (double)sc5[n] * neon_sdot_bound;
            if (actual > bound) {
                if (worst_ns == 0.0)  /* only print first failure */
                    printf("  FAIL: col %zu neon=%.6e sdot=%.6e diff=%.4e bound=%.4e\n",
                           n, c_neon[n], c_sdot[n], actual, bound);
                sdot_ok = 0;
            }
            double ratio = (bound > 1e-30) ? actual / bound : 0.0;
            if (ratio > worst_ns) worst_ns = ratio;
        }
        printf("  worst SDOT-vs-NEON actual/bound: %.4f (should be < 1.0)\n", worst_ns);

        /* Mean relative error SDOT vs NEON over well-conditioned outputs */
        double sum_rel_ns = 0.0; size_t n_ns = 0;
        for (size_t n = 0; n < N5; ++n) {
            double ref = fabs((double)c_neon[n]);
            if (ref > 1.0) {
                sum_rel_ns += fabs((double)c_sdot[n] - c_neon[n]) / ref;
                n_ns++;
            }
        }
        double mean_rel_ns = (n_ns > 0) ? sum_rel_ns / n_ns : 0.0;
        printf("  SDOT-vs-NEON mean_rel=%.4f%% (n=%zu)\n", mean_rel_ns * 100.0, n_ns);

        /* SDOT vs FP32: should be within same theoretical bound as NEON vs FP32,
         * plus the input-quantization overhead (still dominated by weight quant). */
        double worst_sdot_fp32 = 0.0;
        for (size_t n = 0; n < N5; ++n) {
            double actual = fabs((double)c_sdot[n] - c_fp32[n]);
            double ratio = actual / (1.0 + fabs((double)c_fp32[n]));
            if (ratio > worst_sdot_fp32) worst_sdot_fp32 = ratio;
        }
        printf("  SDOT-vs-FP32 max_rel=%.4f%%\n", worst_sdot_fp32 * 100.0);

        int rel_ok = mean_rel_ns < 0.03;  /* <3% mean SDOT-vs-NEON error */
        printf("  %s\n\n", (sdot_ok && rel_ok) ? "PASS" : "FAIL");
        failures += !(sdot_ok && rel_ok);

        free(W5); free(q5); free(sc5); free(q5_sdot);
        free(a5); free(c_neon); free(c_sdot); free(c_fp32);
    }
#endif /* INT8_WEIGHTS && __ARM_FEATURE_DOTPROD */

    free(W); free(q); free(s);

    printf("%s\n", failures == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return failures;
}
