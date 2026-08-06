/* Correctness oracle for gdn_delta_rule_matmul (bead ob-8qt.1).
 *
 * Naive triple-loop fp32 reference vs. the dispatcher, at the four GDN
 * delta-rule shapes actually measured on real silicon in docs/FINDINGS.md
 * section 8: decode (M=1) and prefill (M=64), both single-head (N=128) and
 * all-16-heads-batched (N=2048). Exact bit-identity is not the bar here --
 * unlike the scan/decay/conv kernels, a matmul's reduction order legitimately
 * differs between the naive scalar loop and a vectorized accumulator, so
 * this checks a tight relative-error tolerance instead and exits nonzero on
 * any failure, matching this script family's fail-fast convention
 * (scripts/verify_cpu_kernels.sh runs with set -euo pipefail).
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "gdn_delta_matmul.h"

static void naive_matmul(const float *A, const float *B, float *C, size_t M, size_t K,
                         size_t N) {
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (size_t k = 0; k < K; ++k) acc += A[i * K + k] * B[k * N + j];
            C[i * N + j] = acc;
        }
    }
}

static int check_shape(const char *label, size_t M, size_t K, size_t N) {
    float *A = malloc(M * K * sizeof(float));
    float *B = malloc(K * N * sizeof(float));
    float *C_ref = malloc(M * N * sizeof(float));
    float *C_test = malloc(M * N * sizeof(float));

    unsigned int seed = 42;
    for (size_t i = 0; i < M * K; ++i) A[i] = ((float)(rand_r(&seed) % 2000) - 1000) / 1000.0f;
    for (size_t i = 0; i < K * N; ++i) B[i] = ((float)(rand_r(&seed) % 2000) - 1000) / 1000.0f;

    naive_matmul(A, B, C_ref, M, K, N);
    gdn_delta_rule_matmul(A, B, C_test, M, K, N);

    double max_abs = 0.0, max_rel = 0.0;
    for (size_t i = 0; i < M * N; ++i) {
        double d = fabs((double)C_test[i] - (double)C_ref[i]);
        if (d > max_abs) max_abs = d;
        double refmag = fabs((double)C_ref[i]);
        if (refmag > 1e-6) {
            double r = d / refmag;
            if (r > max_rel) max_rel = r;
        }
    }

    /* K=128 accumulations in fp32: a few ULPs of reduction-order drift is
     * expected and fine; anything past 1e-4 relative is a real bug, not
     * float noise. */
    int pass = max_rel < 1e-4;
    printf("  %-24s M=%3zu K=%3zu N=%4zu  max_abs=%.3e  max_rel=%.3e  %s\n", label, M, K, N,
           max_abs, max_rel, pass ? "PASS" : "FAIL (EXCEEDS TOLERANCE)");

    free(A);
    free(B);
    free(C_ref);
    free(C_test);
    return pass;
}

int main(void) {
    printf("gdn_delta_rule_matmul vs naive fp32 reference:\n");
    int ok = 1;
    /* Shapes match docs/FINDINGS.md section 8 exactly: decode is M=1
     * (one token), prefill is M=64 (one chunk); N=128 is a single head,
     * N=2048 is all 16 heads batched (head_dim=128 * 16 heads). */
    ok &= check_shape("decode_1x128x128", 1, 128, 128);
    ok &= check_shape("prefill_64x128x128", 64, 128, 128);
    ok &= check_shape("decode_1x128x2048", 1, 128, 2048);
    ok &= check_shape("prefill_64x128x2048", 64, 128, 2048);

    /* A non-power-of-two, non-multiple-of-vector-width N to exercise the
     * SVE predicated tail / NEON scalar tail explicitly (mirrors why
     * test_gdn_sve.c uses C=2051 for the same reason). */
    ok &= check_shape("decode_1x128x130", 1, 128, 130);
    ok &= check_shape("prefill_64x128x130", 64, 128, 130);

    printf("\n%s\n", ok ? "ALL PASS" : "FAIL");
    return ok ? 0 : 1;
}
