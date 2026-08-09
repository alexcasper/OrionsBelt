// SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
// SPDX-License-Identifier: Apache-2.0

/*
 * bench_kleidiai_matmul.c — Evaluate KleidiAI matmul micro-kernels at GDN shapes
 *
 * Bead ob-8qt.2: "Reuse KleidiAI i8mm/dotprod matmul micro-kernels for the
 * delta-rule updates." KleidiAI is tuned for larger GEMMs; this benchmark
 * tests whether reuse actually wins at the small per-chunk matmul sizes
 * that GDN's delta-rule produces.
 *
 * The delta-rule update S ← (I − k kᵀ) S + k vᵀ has two main GEMM shapes:
 *   Prefill: K @ S  = [C×d] × [d×d]   (C=64, d=128)
 *   Decode:  k @ S  = [1×d] × [d×d]   (d=128)
 *
 * We compare three implementations:
 *   1. Naive C triple loop (compiler-auto-vectorized at -O3)
 *   2. Hand-written NEON intrinsics (4-wide fp32 FMA)
 *   3. KleidiAI packed GEMM (f32_f32_f32p8x1biasf32_6x8x4_neon_mla)
 *
 * Build (on RK3588 aarch64):
 *   gcc -O3 -march=armv8.2-a+simd -I/tmp/kleidiai \
 *     bench/kleidiai_matmul_bench.c \
 *     /tmp/kleidiai/kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla.c \
 *     /tmp/kleidiai/kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon.c \
 *     -lm -o dist/bench_kleidiai
 */

#define _POSIX_C_SOURCE 200112L  /* clock_gettime, CLOCK_MONOTONIC_RAW, rand_r */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdint.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

/* KleidiAI headers */
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla.h"


/* Safe alloc wrappers — exit on OOM instead of dereferencing NULL. */
__attribute__((unused))
static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) { fprintf(stderr, "out of memory (%zu bytes)\n", n); exit(1); }
    return p;
}
__attribute__((unused))
static void *xcalloc(size_t nmemb, size_t size) {
    void *p = calloc(nmemb, size);
    if (!p) { fprintf(stderr, "out of memory (%zu * %zu bytes)\n", nmemb, size); exit(1); }
    return p;
}
__attribute__((unused))
static void *xaligned_alloc(size_t alignment, size_t size) {
    void *p = aligned_alloc(alignment, size);
    if (!p) { fprintf(stderr, "out of memory (aligned %zu, %zu bytes)\n", alignment, size); exit(1); }
    return p;
}

/* ----------------------------------------------------------------------- */
/* Timing                                                                  */
/* ----------------------------------------------------------------------- */

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

/* ----------------------------------------------------------------------- */
/* Implementation 1: Naive C matmul (compiler will auto-vectorize at -O3)  */
/* ----------------------------------------------------------------------- */

static void naive_matmul(
    const float *A, const float *B, float *C,
    size_t M, size_t K, size_t N)
{
    /* A is row-major [M×K], B is row-major [K×N], C is row-major [M×N] */
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            float acc = 0.0f;
            for (size_t k = 0; k < K; k++) {
                acc += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = acc;
        }
    }
}

/* ----------------------------------------------------------------------- */
/* Implementation 2: Hand-written NEON intrinsics                          */
/* ----------------------------------------------------------------------- */

#ifdef __ARM_NEON
static void neon_matmul(
    const float *A, const float *B, float *C,
    size_t M, size_t K, size_t N)
{
    /* A is row-major [M×K], B is row-major [K×N], C is row-major [M×N] */
    for (size_t i = 0; i < M; i++) {
        /* Zero accumulator row */
        memset(C + i * N, 0, N * sizeof(float));

        for (size_t k = 0; k < K; k++) {
            float a_ik = A[i * K + k];
            float32x4_t a_vec = vdupq_n_f32(a_ik);
            size_t j = 0;
            for (; j + 4 <= N; j += 4) {
                float32x4_t c_vec = vld1q_f32(C + i * N + j);
                float32x4_t b_vec = vld1q_f32(B + k * N + j);
                c_vec = vfmaq_f32(c_vec, a_vec, b_vec);
                vst1q_f32(C + i * N + j, c_vec);
            }
            /* Scalar tail */
            for (; j < N; j++) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}
#endif

/* ----------------------------------------------------------------------- */
/* Implementation 3: KleidiAI packed GEMM                                  */
/* ----------------------------------------------------------------------- */

typedef struct {
    void *rhs_packed;
    size_t rhs_packed_size;
    size_t n_step, nr, kr, sr;
} kleidiai_state;

static void kleidiai_init(kleidiai_state *st, const float *B, const float *bias,
                          size_t K, size_t N)
{
    st->n_step = kai_get_n_step_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon();
    st->nr     = kai_get_nr_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla();
    st->kr     = kai_get_kr_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla();
    st->sr     = kai_get_sr_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla();

    st->rhs_packed_size = kai_get_rhs_packed_size_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon(N, K);
    st->rhs_packed = xaligned_alloc(64, st->rhs_packed_size);
    if (!st->rhs_packed) { fprintf(stderr, "OOM in kleidiai_init\n"); exit(1); }

    /* Pack RHS (B is [K×N] row-major, same as KleidiAI expects) */
    kai_run_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon(
        1,          /* num_groups */
        N,          /* n */
        K,          /* k */
        st->nr,     /* nr */
        st->kr,     /* kr */
        st->sr,     /* sr */
        N * sizeof(float),  /* rhs_stride (row stride in bytes) */
        B,          /* rhs */
        bias,       /* bias */
        NULL,       /* scale (unused for fp32) */
        st->rhs_packed,
        0,          /* extra_bytes */
        NULL        /* params */
    );
}

static void kleidiai_matmul(
    kleidiai_state *st, const float *A, float *C,
    size_t M, size_t K, size_t N)
{
    kai_run_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla(
        M, N, K,
        A, K * sizeof(float),
        st->rhs_packed,
        C, N * sizeof(float), sizeof(float),
        -INFINITY, INFINITY);
}

static void kleidiai_free(kleidiai_state *st) {
    free(st->rhs_packed);
}

/* ----------------------------------------------------------------------- */
/* Benchmark harness                                                       */
/* ----------------------------------------------------------------------- */

typedef void (*matmul_fn)(const float *, const float *, float *, size_t, size_t, size_t);

static void bench(const char *name, matmul_fn fn,
                  const float *A, const float *B, float *C,
                  size_t M, size_t K, size_t N, int repeats)
{
    /* Warmup */
    fn(A, B, C, M, K, N);

    double t0 = now_us();
    for (int r = 0; r < repeats; r++) {
        fn(A, B, C, M, K, N);
    }
    double elapsed = now_us() - t0;

    double per_call_us = elapsed / repeats;
    /* Bytes touched: read A (M*K), read B (K*N), write C (M*N) — all float32 */
    double bytes = (M * K + K * N + M * N) * sizeof(float);
    double gibs = bytes / per_call_us * 1e6 / (1024*1024*1024);
    double gflops = (2.0 * M * K * N) / per_call_us * 1e6 / 1e9;

    printf("  %-20s %10.1f us/call  %7.2f GiB/s  %7.2f GFLOP/s\n",
           name, per_call_us, gibs, gflops);
}

static void bench_kleidiai(
    const char *name,
    const float *A, const float *B, float *C,
    size_t M, size_t K, size_t N, int repeats)
{
    /* Pack RHS once */
    float *bias = xcalloc(N, sizeof(float));  /* zero bias */
    if (!bias) { fprintf(stderr, "OOM in bench_kleidiai\n"); exit(1); }
    kleidiai_state st;
    kleidiai_init(&st, B, bias, K, N);

    /* Warmup */
    kleidiai_matmul(&st, A, C, M, K, N);

    double t0 = now_us();
    for (int r = 0; r < repeats; r++) {
        kleidiai_matmul(&st, A, C, M, K, N);
    }
    double elapsed = now_us() - t0;

    double per_call_us = elapsed / repeats;
    double bytes = (M * K + K * N + M * N) * sizeof(float);
    double gibs = bytes / per_call_us * 1e6 / (1024*1024*1024);
    double gflops = (2.0 * M * K * N) / per_call_us * 1e6 / 1e9;

    printf("  %-20s %10.1f us/call  %7.2f GiB/s  %7.2f GFLOP/s\n",
           name, per_call_us, gibs, gflops);

    kleidiai_free(&st);
    free(bias);
}

/* ----------------------------------------------------------------------- */
/* Correctness check                                                       */
/* ----------------------------------------------------------------------- */

static float max_abs_diff(const float *a, const float *b, size_t n) {
    float max_diff = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > max_diff) max_diff = d;
    }
    return max_diff;
}

/* ----------------------------------------------------------------------- */
/* Main                                                                    */
/* ----------------------------------------------------------------------- */

int main(int argc, char **argv) {
    int repeats = 1000;
    int csv = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--repeats")) repeats = atoi(argv[++i]);
        if (!strcmp(argv[i], "--csv")) csv = 1;
    }

    /* GDN delta-rule shapes:
     *   Decode:  M=1,  K=128, N=128  (single token × state matrix)
     *   Prefill: M=64, K=128, N=128  (chunk of 64 × state matrix)
     *   Head parallel: M=1, K=128, N=128 × 16 heads (batched)
     */
    struct { const char *label; size_t M, K, N; } shapes[] = {
        {"decode_1x128x128",   1,  128, 128},
        {"prefill_64x128x128", 64, 128, 128},
        {"decode_1x128x2048",  1,  128, 2048},  /* all 16 heads at once */
        {"prefill_64x128x2048",64, 128, 2048},  /* all 16 heads at once */
    };

    if (csv) {
        printf("shape,impl,M,K,N,us_per_call,GiB_s,GFLOP_s\n");
    }

    for (size_t s = 0; s < sizeof(shapes)/sizeof(shapes[0]); s++) {
        size_t M = shapes[s].M, K = shapes[s].K, N = shapes[s].N;

        /* Allocate and fill */
        float *A = xmalloc(M * K * sizeof(float));
        float *B = xmalloc(K * N * sizeof(float));
        float *C_ref = xmalloc(M * N * sizeof(float));
        float *C_test = xmalloc(M * N * sizeof(float));
        if (!A || !B || !C_ref || !C_test) {
            fprintf(stderr, "OOM in main allocation\n");
            exit(1);
        }

        /* Deterministic pseudo-random data in [-1, 1] */
        unsigned int seed = 42;
        for (size_t i = 0; i < M * K; i++) A[i] = ((float)(rand_r(&seed) % 2000) - 1000) / 1000.0f;
        for (size_t i = 0; i < K * N; i++) B[i] = ((float)(rand_r(&seed) % 2000) - 1000) / 1000.0f;

        /* Reference: naive */
        naive_matmul(A, B, C_ref, M, K, N);

        if (!csv) {
            printf("\n=== %s (M=%zu, K=%zu, N=%zu) ===\n", shapes[s].label, M, K, N);
        }

        /* Correctness: NEON vs naive */
        neon_matmul(A, B, C_test, M, K, N);
        float neon_err = max_abs_diff(C_ref, C_test, M * N);

        /* Correctness: KleidiAI vs naive */
        float *bias = xcalloc(N, sizeof(float));
        if (!bias) { fprintf(stderr, "OOM in KleidiAI correctness bias\n"); exit(1); }
        kleidiai_state st;
        kleidiai_init(&st, B, bias, K, N);
        kleidiai_matmul(&st, A, C_test, M, K, N);
        float kiai_err = max_abs_diff(C_ref, C_test, M * N);
        kleidiai_free(&st);
        free(bias);

        if (!csv) {
            printf("  Correctness: NEON max_abs_diff=%.2e, KleidiAI max_abs_diff=%.2e\n\n",
                   neon_err, kiai_err);
        }

        /* Benchmarks */
        if (csv) {
            /* For CSV mode, time each implementation */
            double t0, elapsed;

            /* Naive */
            naive_matmul(A, B, C_test, M, K, N); /* warmup */
            t0 = now_us();
            for (int r = 0; r < repeats; r++) naive_matmul(A, B, C_test, M, K, N);
            elapsed = now_us() - t0;
            printf("%s,naive,%zu,%zu,%zu,%.3f,%.2f,%.2f\n",
                   shapes[s].label, M, K, N, elapsed/repeats,
                   (M*K+K*N+M*N)*4/(elapsed/repeats)*1e6/(1024*1024*1024),
                   2.0*M*K*N/(elapsed/repeats)*1e6/1e9);

#ifdef __ARM_NEON
            /* NEON */
            neon_matmul(A, B, C_test, M, K, N);
            t0 = now_us();
            for (int r = 0; r < repeats; r++) neon_matmul(A, B, C_test, M, K, N);
            elapsed = now_us() - t0;
            printf("%s,neon,%zu,%zu,%zu,%.3f,%.2f,%.2f\n",
                   shapes[s].label, M, K, N, elapsed/repeats,
                   (M*K+K*N+M*N)*4/(elapsed/repeats)*1e6/(1024*1024*1024),
                   2.0*M*K*N/(elapsed/repeats)*1e6/1e9);
#endif

            /* KleidiAI */
            bias = xcalloc(N, sizeof(float));
            if (!bias) { fprintf(stderr, "OOM in CSV KleidiAI bias\n"); exit(1); }
            kleidiai_init(&st, B, bias, K, N);
            kleidiai_matmul(&st, A, C_test, M, K, N);
            t0 = now_us();
            for (int r = 0; r < repeats; r++) kleidiai_matmul(&st, A, C_test, M, K, N);
            elapsed = now_us() - t0;
            printf("%s,kleidiai,%zu,%zu,%zu,%.3f,%.2f,%.2f\n",
                   shapes[s].label, M, K, N, elapsed/repeats,
                   (M*K+K*N+M*N)*4/(elapsed/repeats)*1e6/(1024*1024*1024),
                   2.0*M*K*N/(elapsed/repeats)*1e6/1e9);
            kleidiai_free(&st);
            free(bias);
        } else {
            bench("naive_C", naive_matmul, A, B, C_test, M, K, N, repeats);
#ifdef __ARM_NEON
            bench("neon_intrinsics", neon_matmul, A, B, C_test, M, K, N, repeats);
#endif
            bench_kleidiai("kleidiai_packed", A, B, C_test, M, K, N, repeats);
        }

        free(A); free(B); free(C_ref); free(C_test);
    }

    return 0;
}
