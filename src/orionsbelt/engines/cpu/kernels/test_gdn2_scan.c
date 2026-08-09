// SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
// SPDX-License-Identifier: Apache-2.0

/* Correctness oracle for gdn2_gated_scan_f32 (bead ob-y3f / ob-82b).
 *
 * The GDN-2 decoupled-gating scan kernel is implemented in gdn_sve.c and
 * benchmarked in bench_gdn.c (K_SCAN2), with results in FINDINGS.md §10.
 * But unlike the other three GDN kernels (scan, decay, conv), which all have
 * precision-matched reference gates in test_gdn_sve.c, this kernel had NO
 * correctness test. This file fills that gap.
 *
 * GDN-2 recurrence:
 *   GDN-1: s[t] = x[t] + g[t] * s[t-1]
 *   GDN-2: s[t] = w[t]*x[t] + g[t]*b[t]*s[t-1]
 *
 * Why not bit-identical: the GDN-1 kernels ARE bit-identical to a scalar
 * reference because both compile to a single FMA (x + acc*g → vfmaq). The
 * GDN-2 kernel has two extra multiplications (g*b, w*x) before the FMA.
 * Even when the compiler contracts the final add, the intermediate products
 * round separately, and the compiler's auto-vectorization of the scalar
 * reference may choose a different contraction than the hand-written NEON.
 * The result is ~1-2 ULP drift — expected, not a bug. We use a tight relative
 * tolerance (1e-5) instead, which catches real logic errors while accepting
 * legitimate float noise.
 *
 * Tests:
 *   1. Tight tolerance vs precision-matched float reference
 *   2. Carried state correctness
 *   3. GDN-2→GDN-1 reduction: with b_gate=1, w_gate=1, gdn2 ≈ gdn1
 *   4. State carry across two chunks (recurrent continuity)
 *   5. Multi-chunk stability (8 chunks, confirms no accumulation pathology)
 *
 * Build (same pattern as the other test files):
 *   aarch64-linux-gnu-gcc -O3 -march=armv9.2-a+sve2 -static \
 *       gdn_sve.c test_gdn2_scan.c -o /tmp/test_gdn2 -lm
 *   QEMU_CPU=max,sve128=on qemu-aarch64 /tmp/test_gdn2
 */
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gdn_sve.h"

static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) { fprintf(stderr, "out of memory\n"); exit(1); }
    return p;
}


/* GDN-1 kernel for the reduction test — already declared in gdn_sve.h */

/* ---- Precision-matched scalar references (float accumulators, like the kernel) ---- */

static void refF_gdn2(const float *g, const float *b, const float *w, const float *x,
                      float *s, float *st, size_t T, size_t C) {
    for (size_t c = 0; c < C; c++) {
        float acc = st[c];
        for (size_t t = 0; t < T; t++) {
            size_t off = t * C + c;
            acc = w[off] * x[off] + acc * (g[off] * b[off]);
            s[off] = acc;
        }
        st[c] = acc;
    }
}

/* Double-precision reference for the numerical-quality view */
static void refD_gdn2(const float *g, const float *b, const float *w, const float *x,
                      float *s, float *st, size_t T, size_t C) {
    for (size_t c = 0; c < C; c++) {
        double acc = st[c];
        for (size_t t = 0; t < T; t++) {
            size_t off = t * C + c;
            acc = (double)w[off] * x[off] + acc * ((double)g[off] * b[off]);
            s[off] = (float)acc;
        }
        st[c] = (float)acc;
    }
}

static void report(const char *label, const float *a, const float *b, size_t N) {
    double mabs = 0, mrel = 0;
    size_t n_rel = 0;
    for (size_t i = 0; i < N; i++) {
        double d = fabs((double)a[i] - b[i]);
        if (d > mabs) mabs = d;
        if (fabs((double)b[i]) > 1e-2) {
            n_rel++;
            double r = d / fabs((double)b[i]);
            if (r > mrel) mrel = r;
        }
    }
    printf("  %-40s max_abs=%.3e  max_rel(|ref|>1e-2, n=%zu)=%.3e\n",
           label, mabs, n_rel, mrel);
}

/* Relative tolerance for GDN-2 vs scalar reference.
 *
 * The kernel has 2 extra multiplications vs GDN-1 (g*b and w*x before the
 * FMA), so it cannot be bit-identical. The accumulation over 64 steps means
 * at most ~64 * 2 ULP ≈ 1e-5 relative drift. Real logic errors produce
 * orders of magnitude more.
 */
#define GDN2_REL_TOL 1e-5

static int check_rel(const char *label, const float *test, const float *ref,
                     size_t N, double threshold) {
    double max_rel = 0;
    size_t n = 0;
    for (size_t i = 0; i < N; i++) {
        if (fabs((double)ref[i]) > 1e-2) {
            double r = fabs((double)test[i] - ref[i]) / fabs((double)ref[i]);
            if (r > max_rel) max_rel = r;
            n++;
        }
    }
    int pass = max_rel < threshold;
    printf("  %-40s max_rel(n=%zu)=%.3e  %s\n", label, n, max_rel,
           pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

int main(void) {
    size_t T = 64, C = 2051, N = T * C;  /* 2051 exercises the predicated tail */
    int failures = 0;

    float *g   = xmalloc(N * sizeof(float));
    float *b   = xmalloc(N * sizeof(float));
    float *w   = xmalloc(N * sizeof(float));
    float *x   = xmalloc(N * sizeof(float));
    float *sK  = xmalloc(N * sizeof(float));  /* kernel output */
    float *sR  = xmalloc(N * sizeof(float));  /* float reference */
    float *sD  = xmalloc(N * sizeof(float));  /* double reference */
    float *stK = xmalloc(C * sizeof(float));
    float *stR = xmalloc(C * sizeof(float));
    float *stD = xmalloc(C * sizeof(float));

    if (!g || !b || !w || !x || !sK || !sR || !sD || !stK || !stR || !stD) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }

    /* Representative GDN-2 gate ranges:
     *   g (decay gate):   (0.50, 0.90) — controls state decay
     *   b (erase gate):   (0.80, 0.99) — near-1, so g*b ≈ g (erase is gentle)
     *   w (write gate):   (0.10, 0.90) — modulates input magnitude
     *   x (input):        (-1.0, 1.0)  — typical residual-stream values
     */
    srand(42);
    for (size_t i = 0; i < N; i++) {
        g[i] = 0.50f + 0.40f * (rand() / (float)RAND_MAX);
        b[i] = 0.80f + 0.19f * (rand() / (float)RAND_MAX);
        w[i] = 0.10f + 0.80f * (rand() / (float)RAND_MAX);
        x[i] = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }
    for (size_t i = 0; i < C; i++) {
        float v = (rand() / (float)RAND_MAX) - 0.5f;
        stK[i] = stR[i] = stD[i] = v;
    }

    /* ================================================================
     * TEST 1: Tight tolerance vs precision-matched float reference
     * ================================================================ */
    printf("GDN-2 gated_scan vs PRECISION-MATCHED float reference:\n");

    gdn2_gated_scan_f32(g, b, w, x, sK, stK, T, C);
    refF_gdn2(g, b, w, x, sR, stR, T, C);

    report("gdn2_gated_scan output", sK, sR, N);
    report("gdn2_gated_scan carried state", stK, stR, C);
    failures += check_rel("output tolerance check", sK, sR, N, GDN2_REL_TOL);
    failures += check_rel("state tolerance check", stK, stR, C, GDN2_REL_TOL);

    /* ================================================================
     * TEST 2: vs double reference (fp32 accumulation quality)
     * ================================================================ */
    printf("\nGDN-2 gated_scan vs DOUBLE reference (fp32 accumulation quality):\n");
    refD_gdn2(g, b, w, x, sD, stD, T, C);
    report("gdn2_gated_scan output", sK, sD, N);
    failures += check_rel("vs double tolerance check", sK, sD, N, 1e-4);

    /* ================================================================
     * TEST 3: GDN-2 → GDN-1 reduction
     *
     * With b_gate ≡ 1 and w_gate ≡ 1:
     *   GDN-2: s[t] = 1*x[t] + g[t]*1*s[t-1] = x[t] + g[t]*s[t-1]
     *   GDN-1: s[t] = x[t] + g[t]*s[t-1]
     *
     * Mathematically identical. Numerically near-identical: multiplying by
     * 1.0f is exact, but the extra vmul instructions may prevent the same
     * FMA contraction. Expect < 1e-5 relative error.
     * ================================================================ */
    printf("\nGDN-2->GDN-1 reduction (b_gate=1, w_gate=1):\n");

    float *ones_b = xmalloc(N * sizeof(float));
    float *ones_w = xmalloc(N * sizeof(float));
    float *sGdn1 = xmalloc(N * sizeof(float));
    float *sGdn2_red = xmalloc(N * sizeof(float));
    float *st1 = xmalloc(C * sizeof(float));
    float *st2 = xmalloc(C * sizeof(float));

    for (size_t i = 0; i < N; i++) { ones_b[i] = 1.0f; ones_w[i] = 1.0f; }
    for (size_t i = 0; i < C; i++) {
        float v = (rand() / (float)RAND_MAX) - 0.5f;
        st1[i] = st2[i] = v;
    }

    gdn_gated_scan_f32(g, x, sGdn1, st1, T, C);
    gdn2_gated_scan_f32(g, ones_b, ones_w, x, sGdn2_red, st2, T, C);

    report("gdn2(b=1,w=1) vs gdn1", sGdn2_red, sGdn1, N);
    failures += check_rel("reduction tolerance check", sGdn2_red, sGdn1, N, GDN2_REL_TOL);

    free(ones_b); free(ones_w); free(sGdn1); free(sGdn2_red); free(st1); free(st2);

    /* ================================================================
     * TEST 4: State carry across two chunks
     *
     * Run two consecutive chunks and verify the state threads correctly.
     * The kernel must write back the final state, and the next call must
     * read it as initial state — this is the cross-invocation continuity
     * the NOE toolchain has no mechanism for.
     * ================================================================ */
    printf("\nState carry across two chunks:\n");

    srand(99);
    size_t T2 = 32, C2 = 128, N2 = T2 * C2;
    float *g2  = xmalloc(N2 * sizeof(float));
    float *b2  = xmalloc(N2 * sizeof(float));
    float *w2  = xmalloc(N2 * sizeof(float));
    float *x2  = xmalloc(N2 * sizeof(float));
    float *sK2 = xmalloc(N2 * sizeof(float));
    float *sR2 = xmalloc(N2 * sizeof(float));
    float *stK2 = xmalloc(C2 * sizeof(float));
    float *stR2 = xmalloc(C2 * sizeof(float));

    for (size_t i = 0; i < N2; i++) {
        g2[i] = 0.50f + 0.40f * (rand() / (float)RAND_MAX);
        b2[i] = 0.80f + 0.19f * (rand() / (float)RAND_MAX);
        w2[i] = 0.10f + 0.80f * (rand() / (float)RAND_MAX);
        x2[i] = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }
    for (size_t i = 0; i < C2; i++) {
        float v = (rand() / (float)RAND_MAX) - 0.5f;
        stK2[i] = stR2[i] = v;
    }

    /* Chunk 1 */
    gdn2_gated_scan_f32(g2, b2, w2, x2, sK2, stK2, T2, C2);
    refF_gdn2(g2, b2, w2, x2, sR2, stR2, T2, C2);
    report("chunk 1 output", sK2, sR2, N2);
    failures += check_rel("chunk 1 tolerance check", sK2, sR2, N2, GDN2_REL_TOL);

    /* Chunk 2: carry state forward */
    float *g2b  = xmalloc(N2 * sizeof(float));
    float *b2b  = xmalloc(N2 * sizeof(float));
    float *w2b  = xmalloc(N2 * sizeof(float));
    float *x2b  = xmalloc(N2 * sizeof(float));
    float *sK2b = xmalloc(N2 * sizeof(float));
    float *sR2b = xmalloc(N2 * sizeof(float));
    for (size_t i = 0; i < N2; i++) {
        g2b[i] = 0.50f + 0.40f * (rand() / (float)RAND_MAX);
        b2b[i] = 0.80f + 0.19f * (rand() / (float)RAND_MAX);
        w2b[i] = 0.10f + 0.80f * (rand() / (float)RAND_MAX);
        x2b[i] = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }

    gdn2_gated_scan_f32(g2b, b2b, w2b, x2b, sK2b, stK2, T2, C2);
    refF_gdn2(g2b, b2b, w2b, x2b, sR2b, stR2, T2, C2);
    report("chunk 2 output (state carried)", sK2b, sR2b, N2);
    report("chunk 2 carried state", stK2, stR2, C2);
    failures += check_rel("chunk 2 tolerance check", sK2b, sR2b, N2, GDN2_REL_TOL);

    free(g2); free(b2); free(w2); free(x2); free(sK2); free(sR2);
    free(stK2); free(stR2); free(g2b); free(b2b); free(w2b); free(x2b);
    free(sK2b); free(sR2b);

    /* ================================================================
     * TEST 5: Multi-chunk stability (8 chunks, same data per chunk)
     *
     * Confirms the kernel's output doesn't drift relative to the reference
     * over many chunks — if the kernel had an internal-state bug (e.g.
     * failing to write back the final state), the drift would compound.
     * ================================================================ */
    printf("\nMulti-chunk stability (8 chunks, same data per chunk):\n");

    /* Reset state */
    srand(7);
    for (size_t i = 0; i < C; i++) {
        float v = (rand() / (float)RAND_MAX) - 0.5f;
        stK[i] = stR[i] = v;
    }

    double worst_rel = 0;
    for (int chunk = 0; chunk < 8; chunk++) {
        gdn2_gated_scan_f32(g, b, w, x, sK, stK, T, C);
        refF_gdn2(g, b, w, x, sR, stR, T, C);
        double mrel = 0;
        for (size_t i = 0; i < N; i++) {
            if (fabs((double)sR[i]) > 1e-2) {
                double r = fabs((double)sK[i] - sR[i]) / fabs((double)sR[i]);
                if (r > mrel) mrel = r;
            }
        }
        printf("  chunk %d: max_rel=%.3e\n", chunk + 1, mrel);
        if (mrel > worst_rel) worst_rel = mrel;
    }
    printf("  worst over 8 chunks: %.3e %s\n", (double)worst_rel,
           worst_rel < GDN2_REL_TOL ? "(stable)" : "(CHECK)");
    if (worst_rel >= GDN2_REL_TOL) failures++;

    /* ================================================================
     * TEST 6: Determinism — same inputs produce same outputs
     * ================================================================ */
    printf("\nDeterminism check (two runs, identical inputs):\n");

    /* Reset state for both runs */
    srand(13);
    for (size_t i = 0; i < C; i++) {
        float v = (rand() / (float)RAND_MAX) - 0.5f;
        stK[i] = v;
    }
    float *stK_save = xmalloc(C * sizeof(float));
    memcpy(stK_save, stK, C * sizeof(float));

    gdn2_gated_scan_f32(g, b, w, x, sK, stK, T, C);
    /* Save outputs */
    float *run1_s = xmalloc(N * sizeof(float));
    float *run1_st = xmalloc(C * sizeof(float));
    memcpy(run1_s, sK, N * sizeof(float));
    memcpy(run1_st, stK, C * sizeof(float));

    /* Second run with identical state */
    memcpy(stK, stK_save, C * sizeof(float));
    gdn2_gated_scan_f32(g, b, w, x, sK, stK, T, C);

    int nondet = 0;
    for (size_t i = 0; i < N; i++)
        if (run1_s[i] != sK[i]) nondet++;
    for (size_t i = 0; i < C; i++)
        if (run1_st[i] != stK[i]) nondet++;

    printf("  %s\n", nondet == 0 ? "PASS (bit-identical)" : "FAIL (nondeterministic)");
    if (nondet) failures++;

    free(run1_s); free(run1_st); free(stK_save);

    /* ================================================================ */
    printf("\n%s\n", failures == 0
        ? "ALL TESTS PASSED"
        : "SOME TESTS FAILED");

    free(g); free(b); free(w); free(x); free(sK); free(sR); free(sD);
    free(stK); free(stR); free(stD);
    return failures;
}
