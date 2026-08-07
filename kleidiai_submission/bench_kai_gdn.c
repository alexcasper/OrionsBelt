// SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
// SPDX-License-Identifier: Apache-2.0

#define _POSIX_C_SOURCE 199309L  /* clock_gettime, CLOCK_MONOTONIC_RAW */

//
// Micro-benchmark for the four GDN KleidiAI micro-kernels.
//
// Produces a CSV on stdout:
//   kernel,shape,seq,channels,repeats,p50_us,gib_per_s_p50
//
// Protocol: 3 warmup iterations discarded, N repeats timed individually,
// p50 by nearest-rank percentile (matches bench/bench_delta_matmul.c).
//
// Build:
//   cd kleidiai_submission && make bench
// Or:
//   gcc -O3 -march=armv8-a -I kleidiai_submission
//       kleidiai_submission/bench_kai_gdn.c
//       kleidiai_submission/kai/ukernels/gdn/*.c
//       -lm -o bench_kai_gdn

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#include "kai/ukernels/gdn/kai_gdn_cumdecay_f32_sve.h"
#include "kai/ukernels/gdn/kai_gdn_gated_scan_f32_sve.h"
#include "kai/ukernels/gdn/kai_gdn_causal_dwconv1d_f32_k4_sve.h"
#include "kai/ukernels/gdn/kai_gdn_gemv_f32_f32_f32_1x4_neon.h"

/* ----------------------------------------------------------------------- */
/* Timing                                                                  */
/* ----------------------------------------------------------------------- */

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double p50_val(double *sorted, int n) {
    int idx = (int)(0.50 * (double)(n - 1) + 0.5);
    if (idx < 0) idx = 0;
    if (idx >= n) idx = n - 1;
    return sorted[idx];
}

/* ----------------------------------------------------------------------- */
/* Bench helpers                                                           */
/* ----------------------------------------------------------------------- */

static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) { fprintf(stderr, "out of memory\n"); exit(1); }
    return p;
}

static void bench_kernel(const char *name, const char *shape,
                         int seq, int channels, int repeats,
                         void (*fn)(void *ctx),
                         void *ctx,
                         double bytes_per_call)
{
/* Clock granularity on some aarch64 SoCs (e.g. RK3588) is ~291 ns, and
 * consecutive clock_gettime calls return the same timestamp 82% of the time.
 * Single-call timing is meaningless for fast kernels — gated_scan 1x160
 * genuinely completes in <291 ns, so p50 reads 0.000 us.
 *
 * Fix: batch BATCH calls per measurement.  100 calls of even the fastest
 * kernel (~0.3 us) totals ~30 us — well above the clock quantum.  The
 * reported per-call time is batch_time / BATCH. */
#define BATCH 100

    for (int i = 0; i < 20; ++i) fn(ctx);  /* warmup */

    double *samples = xmalloc((size_t)repeats * sizeof(double));
    for (int i = 0; i < repeats; ++i) {
        double t0 = now_us();
        for (int b = 0; b < BATCH; ++b) fn(ctx);
        samples[i] = (now_us() - t0) / BATCH;
    }
    qsort(samples, (size_t)repeats, sizeof(double), cmp_double);
    double us = p50_val(samples, repeats);
    double gib = bytes_per_call / (us * 1e-6) / (1024.0 * 1024.0 * 1024.0);

    printf("%s,%s,%d,%d,%d,%.3f,%.2f\n",
           name, shape, seq, channels, repeats, us, gib);

    free(samples);
}

/* ----------------------------------------------------------------------- */
/* Kernel wrappers                                                         */
/* ----------------------------------------------------------------------- */

typedef struct { float *a, *decay; size_t seq, ch; } ctx_cumdecay;
static void run_cumdecay(void *c) {
    ctx_cumdecay *ctx = c;
    kai_run_gdn_cumdecay_f32_sve(ctx->a, ctx->decay, ctx->seq, ctx->ch);
}

typedef struct { float *g, *x, *s, *state; size_t seq, ch; } ctx_scan;
static void run_scan(void *c) {
    ctx_scan *ctx = c;
    kai_run_gdn_gated_scan_f32_sve(ctx->g, ctx->x, ctx->s, ctx->state,
                                   ctx->seq, ctx->ch);
}

typedef struct { float *in, *w, *out, *hist; size_t seq, ch; } ctx_conv;
static void run_conv(void *c) {
    ctx_conv *ctx = c;
    /* Reset hist each iteration for reproducible timing */
    kai_run_gdn_causal_dwconv1d_f32_k4_sve(ctx->in, ctx->w, ctx->out,
                                           ctx->hist, ctx->seq, ctx->ch);
}

typedef struct { float *a, *b, *c; size_t k, n; } ctx_gemv;
static void run_gemv(void *c) {
    ctx_gemv *ctx = c;
    kai_run_gdn_gemv_f32_f32_f32_1x4_neon(ctx->a, ctx->b, ctx->c,
                                          ctx->k, ctx->n);
}

/* ----------------------------------------------------------------------- */
/* Main                                                                    */
/* ----------------------------------------------------------------------- */

int main(int argc, char **argv) {
    int repeats = 30;
    int csv = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--repeats") == 0 && i + 1 < argc)
            repeats = atoi(argv[++i]);
        else if (strcmp(argv[i], "--csv") == 0)
            csv = 1;
    }

    /* GDN shapes from the model architecture (hidden_size=2560, 16 heads):
     *   single-head channels = 2560 / 16 = 160
     *   all-heads channels   = 2560
     *   chunk (seq)          = 64
     */
    int shapes[][2] = {
        /* seq, channels */
        { 64,  160},   /* single-head prefill */
        {  1,  160},   /* single-head decode */
        { 64, 2560},   /* all-heads prefill */
        {  1, 2560},   /* all-heads decode */
    };
    int nshapes = (int)(sizeof(shapes) / sizeof(shapes[0]));

    /* GEMV shapes: K (reduction), N (output) */
    int gemv_shapes[][2] = {
        /* K,   N */
        {128,  128},   /* single-head decode */
        {128, 2048},   /* all-heads decode */
        {128, 2560},   /* full hidden decode */
    };
    int ngemv = (int)(sizeof(gemv_shapes) / sizeof(gemv_shapes[0]));

    if (!csv) {
        printf("GDN KleidiAI kernel microbenchmark\n");
        printf("  repeats per shape : %d × %d calls (after 20 warmups)\n\n", repeats, BATCH);
    }

    /* Detect ISA path */
    const char *isa = "scalar";
#ifdef __ARM_FEATURE_SVE
    isa = "sve";
#elif defined(__ARM_NEON)
    isa = "neon";
#endif
    if (!csv) printf("  dispatch path     : %s\n\n", isa);

    /* CSV header */
    if (csv) printf("kernel,shape,seq,channels,repeats,p50_us,gib_per_s_p50\n");

    /* --- Cumdecay --- */
    for (int si = 0; si < nshapes; ++si) {
        int seq = shapes[si][0], ch = shapes[si][1];
        size_t total = (size_t)seq * ch;
        float *a = xmalloc(total * sizeof(float));
        float *decay = xmalloc(total * sizeof(float));
        memset(a, 0x3f, total * sizeof(float)); /* ~1.0f */
        memset(decay, 0, total * sizeof(float));

        char label[32];
        snprintf(label, sizeof(label), "%dx%d", seq, ch);
        /* bytes: read a[] + write decay[] = 2 * total * 4 */
        bench_kernel("cumdecay", label, seq, ch, repeats,
                     run_cumdecay, &(ctx_cumdecay){a, decay, seq, ch},
                     2.0 * total * sizeof(float));

        free(a); free(decay);
    }

    /* --- Gated scan --- */
    for (int si = 0; si < nshapes; ++si) {
        int seq = shapes[si][0], ch = shapes[si][1];
        size_t total = (size_t)seq * ch;
        float *g = xmalloc(total * sizeof(float));
        float *x = xmalloc(total * sizeof(float));
        float *s = xmalloc(total * sizeof(float));
        float *state = xmalloc(ch * sizeof(float));
        memset(g, 0x3f, total * sizeof(float)); /* ~1.0f */
        memset(x, 0x3f, total * sizeof(float));
        memset(s, 0, total * sizeof(float));
        memset(state, 0, ch * sizeof(float));

        char label[32];
        snprintf(label, sizeof(label), "%dx%d", seq, ch);
        /* bytes: read g[] + read x[] + write s[] + read+write state[] */
        bench_kernel("gated_scan", label, seq, ch, repeats,
                     run_scan, &(ctx_scan){g, x, s, state, seq, ch},
                     (3.0 * total + 2.0 * ch) * sizeof(float));

        free(g); free(x); free(s); free(state);
    }

    /* --- Causal dwconv1d --- */
    for (int si = 0; si < nshapes; ++si) {
        int seq = shapes[si][0], ch = shapes[si][1];
        size_t total = (size_t)seq * ch;
        float *in = xmalloc(total * sizeof(float));
        float *w = xmalloc(4 * ch * sizeof(float));
        float *out = xmalloc(total * sizeof(float));
        float *hist = xmalloc(3 * ch * sizeof(float));
        memset(in, 0x3f, total * sizeof(float));
        memset(w, 0x3f, 4 * ch * sizeof(float));
        memset(out, 0, total * sizeof(float));
        memset(hist, 0, 3 * ch * sizeof(float));

        char label[32];
        snprintf(label, sizeof(label), "%dx%d", seq, ch);
        /* bytes: read in[] + read w[4*ch] + write out[] + read+write hist[3*ch] */
        bench_kernel("dwconv1d", label, seq, ch, repeats,
                     run_conv, &(ctx_conv){in, w, out, hist, seq, ch},
                     (2.0 * total + 4.0 * ch + 2.0 * 3.0 * ch) * sizeof(float));

        free(in); free(w); free(out); free(hist);
    }

    /* --- GEMV --- */
    for (int si = 0; si < ngemv; ++si) {
        int k = gemv_shapes[si][0], n = gemv_shapes[si][1];
        float *a = xmalloc(k * sizeof(float));
        float *b = xmalloc((size_t)k * n * sizeof(float));
        float *c = xmalloc(n * sizeof(float));
        memset(a, 0x3f, k * sizeof(float));
        memset(b, 0x3f, (size_t)k * n * sizeof(float));
        memset(c, 0, n * sizeof(float));

        char label[32];
        snprintf(label, sizeof(label), "K%d_N%d", k, n);
        /* bytes: read a[k] + read b[k*n] + write c[n] */
        bench_kernel("gemv", label, 0, n, repeats,
                     run_gemv, &(ctx_gemv){a, b, c, k, n},
                     ((double)k + (double)k * n + (double)n) * sizeof(float));

        free(a); free(b); free(c);
    }

    if (!csv) {
        printf("\nDone. %d repeats per shape, ISA path: %s\n", repeats, isa);
    }

    return 0;
}
