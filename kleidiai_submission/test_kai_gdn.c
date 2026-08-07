// SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
// SPDX-License-Identifier: Apache-2.0
//
// Test harness for the four GDN KleidiAI micro-kernels.
//
// Each kernel is compared against a naive scalar C reference implementation.
// On success the program prints "ALL TESTS PASSED" and returns 0.
//
// Build:
//   gcc -O3 -march=armv8.2-a+simd \
//       -I kleidiai_submission \
//       kleidiai_submission/test_kai_gdn.c \
//       kleidiai_submission/kai/ukernels/gdn/*.c \
//       -lm -o test_kai_gdn
//
// For SVE targets:
//   aarch64-linux-gnu-gcc -O3 -march=armv8.2-a+sve \
//       -I kleidiai_submission \
//       kleidiai_submission/test_kai_gdn.c \
//       kleidiai_submission/kai/ukernels/gdn/*.c \
//       -lm -o test_kai_gdn

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kai/ukernels/gdn/kai_gdn_cumdecay_f32_sve.h"
#include "kai/ukernels/gdn/kai_gdn_gated_scan_f32_sve.h"
#include "kai/ukernels/gdn/kai_gdn_causal_dwconv1d_f32_k4_sve.h"
#include "kai/ukernels/gdn/kai_gdn_gemv_f32_f32_f32_1x4_neon_dotprod.h"

/* ----------------------------------------------------------------------- */
/* Helpers                                                                  */
/* ----------------------------------------------------------------------- */

static int g_tests_run = 0;
static int g_tests_failed = 0;

/* Deterministic PRNG (so results are reproducible across runs/platforms). */
static unsigned int g_rng_state = 0x5EED1234u;

static float rand_float(float lo, float hi) {
    g_rng_state = g_rng_state * 1103515245u + 12345u;
    float r = (float)((g_rng_state >> 8) & 0xFFFFFF) / (float)0xFFFFFF;
    return lo + r * (hi - lo);
}

static void fill_random(float *p, size_t n, float lo, float hi) {
    for (size_t i = 0; i < n; ++i) p[i] = rand_float(lo, hi);
}

static int approx_eq(float a, float b, float abs_tol, float rel_tol) {
    float diff = fabsf(a - b);
    if (diff <= abs_tol) return 1;
    float mag = fmaxf(fabsf(a), fabsf(b));
    return diff <= rel_tol * mag;
}

static int compare_arrays(const float *got, const float *ref, size_t n,
                          const char *label, float abs_tol, float rel_tol) {
    int ok = 1;
    for (size_t i = 0; i < n; ++i) {
        if (!approx_eq(got[i], ref[i], abs_tol, rel_tol)) {
            if (ok) {
                printf("  MISMATCH in %s (first at index %zu):\n", label, i);
            }
            printf("    got[%zu] = %.7e   ref[%zu] = %.7e   diff = %.3e\n",
                   i, got[i], i, ref[i], fabsf(got[i] - ref[i]));
            ok = 0;
            if (i > 10) {
                printf("    ... (suppressing further mismatches)\n");
                break;
            }
        }
    }
    return ok;
}

#define RUN_TEST(fn_name, test_body)                                         \
    do {                                                                      \
        g_tests_run++;                                                        \
        printf("[ RUN      ] %s\n", fn_name);                                 \
        test_body                                                             \
        printf("[       OK ] %s\n\n", fn_name);                              \
    } while (0)

#define FAIL_TEST(fn_name)                                                    \
    do {                                                                      \
        g_tests_failed++;                                                     \
        printf("[  FAILED  ] %s\n\n", fn_name);                              \
    } while (0)

/* ----------------------------------------------------------------------- */
/* Naive reference implementations                                          */
/* ----------------------------------------------------------------------- */

static void ref_cumdecay_f32(const float *a, float *decay, size_t seq,
                             size_t channels) {
    for (size_t c = 0; c < channels; ++c) {
        float run = 1.0f;
        for (size_t t = 0; t < seq; ++t) {
            run *= a[t * channels + c];
            decay[t * channels + c] = run;
        }
    }
}

static void ref_gated_scan_f32(const float *g, const float *x, float *s,
                               float *state, size_t seq, size_t channels) {
    for (size_t c = 0; c < channels; ++c) {
        float acc = state[c];
        for (size_t t = 0; t < seq; ++t) {
            acc = x[t * channels + c] + acc * g[t * channels + c];
            s[t * channels + c] = acc;
        }
        state[c] = acc;
    }
}

#define REF_CONV_K 4

static void ref_causal_dwconv1d_f32(const float *in, const float *w,
                                    float *out, float *hist, size_t seq,
                                    size_t channels) {
    for (size_t c = 0; c < channels; ++c) {
        float h[REF_CONV_K - 1];
        for (int j = 0; j < REF_CONV_K - 1; ++j)
            h[j] = hist[(size_t)j * channels + c];
        for (size_t t = 0; t < seq; ++t) {
            float cur = in[t * channels + c];
            float acc = h[0] * w[0 * channels + c] +
                        h[1] * w[1 * channels + c] +
                        h[2] * w[2 * channels + c] +
                        cur * w[3 * channels + c];
            out[t * channels + c] = acc;
            h[0] = h[1];
            h[1] = h[2];
            h[2] = cur;
        }
        for (int j = 0; j < REF_CONV_K - 1; ++j)
            hist[(size_t)j * channels + c] = h[j];
    }
}

static void ref_gemv_f32(const float *a, const float *b, float *c, size_t k,
                         size_t n) {
    for (size_t j = 0; j < n; ++j) {
        float acc = 0.0f;
        for (size_t kk = 0; kk < k; ++kk) {
            acc += a[kk] * b[kk * n + j];
        }
        c[j] = acc;
    }
}

/* ----------------------------------------------------------------------- */
/* Test cases                                                               */
/* ----------------------------------------------------------------------- */

/* Test sizes: include multiples of 8, 4, and odd sizes to exercise
 * NEON/SVE tail handling and the scalar epilogue. */
static const size_t TEST_SEQS[] = {1, 4, 16, 64};
static const size_t TEST_CHANNELS[] = {1, 3, 7, 8, 12, 16, 17, 32, 33, 128};

#define N_SEQS (sizeof(TEST_SEQS) / sizeof(TEST_SEQS[0]))
#define N_CHANS (sizeof(TEST_CHANNELS) / sizeof(TEST_CHANNELS[0]))

static void test_cumdecay(void) {
    int all_ok = 1;
    for (size_t si = 0; si < N_SEQS; ++si) {
        for (size_t ci = 0; ci < N_CHANS; ++ci) {
            size_t seq = TEST_SEQS[si];
            size_t ch = TEST_CHANNELS[ci];
            size_t total = seq * ch;

            float *a = malloc(total * sizeof(float));
            float *got = malloc(total * sizeof(float));
            float *ref = malloc(total * sizeof(float));

            fill_random(a, total, 0.5f, 1.5f); /* keep products in range */

            ref_cumdecay_f32(a, ref, seq, ch);
            kai_run_gdn_cumdecay_f32_sve(a, got, seq, ch);

            if (!compare_arrays(got, ref, total, "cumdecay", 1e-5f, 1e-4f))
                all_ok = 0;

            free(a); free(got); free(ref);
        }
    }
    if (all_ok) {} else FAIL_TEST("test_cumdecay");
}

static void test_gated_scan(void) {
    int all_ok = 1;
    for (size_t si = 0; si < N_SEQS; ++si) {
        for (size_t ci = 0; ci < N_CHANS; ++ci) {
            size_t seq = TEST_SEQS[si];
            size_t ch = TEST_CHANNELS[ci];
            size_t total = seq * ch;

            float *g = malloc(total * sizeof(float));
            float *x = malloc(total * sizeof(float));
            float *got_s = malloc(total * sizeof(float));
            float *got_state = malloc(ch * sizeof(float));
            float *ref_s = malloc(total * sizeof(float));
            float *ref_state = malloc(ch * sizeof(float));

            fill_random(g, total, -0.99f, 0.99f);
            fill_random(x, total, -2.0f, 2.0f);
            fill_random(got_state, ch, -1.0f, 1.0f);
            memcpy(ref_state, got_state, ch * sizeof(float));

            ref_gated_scan_f32(g, x, ref_s, ref_state, seq, ch);
            kai_run_gdn_gated_scan_f32_sve(g, x, got_s, got_state, seq, ch);

            if (!compare_arrays(got_s, ref_s, total, "gated_scan output", 1e-4f, 5e-4f))
                all_ok = 0;
            if (!compare_arrays(got_state, ref_state, ch,
                                "gated_scan state", 1e-4f, 5e-4f))
                all_ok = 0;

            free(g); free(x); free(got_s); free(got_state);
            free(ref_s); free(ref_state);
        }
    }
    if (all_ok) {} else FAIL_TEST("test_gated_scan");
}

static void test_causal_dwconv1d(void) {
    int all_ok = 1;
    for (size_t si = 0; si < N_SEQS; ++si) {
        for (size_t ci = 0; ci < N_CHANS; ++ci) {
            size_t seq = TEST_SEQS[si];
            size_t ch = TEST_CHANNELS[ci];
            size_t total = seq * ch;

            float *in = malloc(total * sizeof(float));
            float *w = malloc((size_t)REF_CONV_K * ch * sizeof(float));
            float *got_out = malloc(total * sizeof(float));
            float *got_hist = malloc((size_t)(REF_CONV_K - 1) * ch * sizeof(float));
            float *ref_out = malloc(total * sizeof(float));
            float *ref_hist = malloc((size_t)(REF_CONV_K - 1) * ch * sizeof(float));

            fill_random(in, total, -1.0f, 1.0f);
            fill_random(w, (size_t)REF_CONV_K * ch, -1.0f, 1.0f);
            fill_random(got_hist, (size_t)(REF_CONV_K - 1) * ch, -1.0f, 1.0f);
            memcpy(ref_hist, got_hist, (size_t)(REF_CONV_K - 1) * ch * sizeof(float));

            ref_causal_dwconv1d_f32(in, w, ref_out, ref_hist, seq, ch);
            kai_run_gdn_causal_dwconv1d_f32_k4_sve(in, w, got_out, got_hist, seq, ch);

            if (!compare_arrays(got_out, ref_out, total,
                                "dwconv1d output", 1e-4f, 5e-4f))
                all_ok = 0;
            if (!compare_arrays(got_hist, ref_hist,
                                (size_t)(REF_CONV_K - 1) * ch,
                                "dwconv1d history", 1e-4f, 5e-4f))
                all_ok = 0;

            free(in); free(w); free(got_out); free(got_hist);
            free(ref_out); free(ref_hist);
        }
    }
    if (all_ok) {} else FAIL_TEST("test_causal_dwconv1d");
}

static void test_gemv(void) {
    int all_ok = 1;
    /* GEMV test sizes: K (reduction) and N (output) */
    static const size_t TEST_K[] = {1, 4, 16, 64, 128};
    static const size_t TEST_N[] = {1, 3, 4, 7, 8, 16, 17, 32, 33, 128, 256};
    size_t nk = sizeof(TEST_K) / sizeof(TEST_K[0]);
    size_t nn = sizeof(TEST_N) / sizeof(TEST_N[0]);

    for (size_t ki = 0; ki < nk; ++ki) {
        for (size_t ni = 0; ni < nn; ++ni) {
            size_t k = TEST_K[ki];
            size_t n = TEST_N[ni];

            float *a = malloc(k * sizeof(float));
            float *b = malloc(k * n * sizeof(float));
            float *got_c = malloc(n * sizeof(float));
            float *ref_c = malloc(n * sizeof(float));

            fill_random(a, k, -1.0f, 1.0f);
            fill_random(b, k * n, -1.0f, 1.0f);

            ref_gemv_f32(a, b, ref_c, k, n);
            kai_run_gdn_gemv_f32_f32_f32_1x4_neon_dotprod(a, b, got_c, k, n);

            if (!compare_arrays(got_c, ref_c, n, "gemv", 1e-3f, 5e-3f))
                all_ok = 0;

            free(a); free(b); free(got_c); free(ref_c);
        }
    }
    if (all_ok) {} else FAIL_TEST("test_gemv");
}

/* Test cross-call state continuity for gated_scan: two sequential calls
 * should produce the same result as one call with the concatenated input. */
static void test_gated_scan_continuity(void) {
    size_t ch = 16;
    size_t seq1 = 8;
    size_t seq2 = 8;
    size_t seq_full = seq1 + seq2;

    float *g = malloc(seq_full * ch * sizeof(float));
    float *x = malloc(seq_full * ch * sizeof(float));
    float *ref_s = malloc(seq_full * ch * sizeof(float));
    float *ref_state = malloc(ch * sizeof(float));
    float *got_s = malloc(seq_full * ch * sizeof(float));
    float *got_state = malloc(ch * sizeof(float));

    fill_random(g, seq_full * ch, -0.99f, 0.99f);
    fill_random(x, seq_full * ch, -2.0f, 2.0f);
    fill_random(ref_state, ch, -1.0f, 1.0f);
    memcpy(got_state, ref_state, ch * sizeof(float));

    /* Reference: one big call */
    ref_gated_scan_f32(g, x, ref_s, ref_state, seq_full, ch);

    /* Kernel: two sequential calls */
    kai_run_gdn_gated_scan_f32_sve(g, x, got_s, got_state, seq1, ch);
    kai_run_gdn_gated_scan_f32_sve(g + seq1 * ch, x + seq1 * ch,
                                   got_s + seq1 * ch, got_state, seq2, ch);

    int ok = compare_arrays(got_s, ref_s, seq_full * ch,
                            "gated_scan continuity", 1e-4f, 5e-4f);
    ok = ok && compare_arrays(got_state, ref_state, ch,
                              "gated_scan continuity state", 1e-4f, 5e-4f);
    if (!ok) FAIL_TEST("test_gated_scan_continuity");

    free(g); free(x); free(ref_s); free(ref_state);
    free(got_s); free(got_state);
}

/* Test cross-call history continuity for dwconv1d: two sequential calls
 * should produce the same result as one call with concatenated input. */
static void test_dwconv1d_continuity(void) {
    size_t ch = 17; /* odd, to test tails */
    size_t seq1 = 8;
    size_t seq2 = 8;
    size_t seq_full = seq1 + seq2;

    float *in = malloc(seq_full * ch * sizeof(float));
    float *w = malloc((size_t)REF_CONV_K * ch * sizeof(float));
    float *ref_out = malloc(seq_full * ch * sizeof(float));
    float *ref_hist = malloc((size_t)(REF_CONV_K - 1) * ch * sizeof(float));
    float *got_out = malloc(seq_full * ch * sizeof(float));
    float *got_hist = malloc((size_t)(REF_CONV_K - 1) * ch * sizeof(float));

    fill_random(in, seq_full * ch, -1.0f, 1.0f);
    fill_random(w, (size_t)REF_CONV_K * ch, -1.0f, 1.0f);
    fill_random(ref_hist, (size_t)(REF_CONV_K - 1) * ch, -1.0f, 1.0f);
    memcpy(got_hist, ref_hist, (size_t)(REF_CONV_K - 1) * ch * sizeof(float));

    /* Reference: one big call */
    ref_causal_dwconv1d_f32(in, w, ref_out, ref_hist, seq_full, ch);

    /* Kernel: two sequential calls */
    kai_run_gdn_causal_dwconv1d_f32_k4_sve(in, w, got_out, got_hist, seq1, ch);
    kai_run_gdn_causal_dwconv1d_f32_k4_sve(in + seq1 * ch, w,
                                           got_out + seq1 * ch, got_hist, seq2, ch);

    int ok = compare_arrays(got_out, ref_out, seq_full * ch,
                            "dwconv1d continuity", 1e-4f, 5e-4f);
    ok = ok && compare_arrays(got_hist, ref_hist,
                              (size_t)(REF_CONV_K - 1) * ch,
                              "dwconv1d continuity hist", 1e-4f, 5e-4f);
    if (!ok) FAIL_TEST("test_dwconv1d_continuity");

    free(in); free(w); free(ref_out); free(ref_hist);
    free(got_out); free(got_hist);
}

/* ----------------------------------------------------------------------- */
/* Main                                                                     */
/* ----------------------------------------------------------------------- */

int main(void) {
    printf("=== KleidiAI GDN Micro-Kernel Test Suite ===\n\n");

    RUN_TEST("test_cumdecay", test_cumdecay(););
    RUN_TEST("test_gated_scan", test_gated_scan(););
    RUN_TEST("test_causal_dwconv1d", test_causal_dwconv1d(););
    RUN_TEST("test_gemv", test_gemv(););
    RUN_TEST("test_gated_scan_continuity", test_gated_scan_continuity(););
    RUN_TEST("test_dwconv1d_continuity", test_dwconv1d_continuity(););

    printf("=== Summary ===\n");
    printf("Tests run:    %d\n", g_tests_run);
    printf("Tests failed: %d\n", g_tests_failed);

    if (g_tests_failed == 0) {
        printf("\nALL TESTS PASSED\n");
        return 0;
    } else {
        printf("\nSOME TESTS FAILED\n");
        return 1;
    }
}
