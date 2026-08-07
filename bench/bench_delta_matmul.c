/*
 * bench_delta_matmul.c — Microbenchmark for the GDN delta-rule matmul kernel
 *
 * Bead ob-8qt.1. Produces CSV in the delta_matmul schema:
 *   kernel,M,K,N,repeats,p50_us,p95_us,gib_per_s_p50
 *
 * This harness exists so every fleet device can produce a delta_matmul CSV
 * with the same shapes and methodology, without ad-hoc one-liners. The
 * kernel under test (gdn_delta_rule_matmul) has NEON/SVE/scalar paths;
 * the build flags select which path is compiled in.
 *
 * Shapes (matching docs/FINDINGS.md section 8 and test_gdn_delta_matmul.c):
 *   decode:   M=1  K=128 N=128   (single head, single token)
 *   prefill:  M=64 K=128 N=128   (single head, one chunk)
 *   decode:   M=1  K=128 N=2048  (all 16 heads batched, single token)
 *   prefill:  M=64 K=128 N=2048  (all 16 heads batched, one chunk)
 *
 * Bandwidth accounting: bytes read+written = 4*(M*K + K*N + M*N) per call
 *   (all operands are fp32; A[M×K] read, B[K×N] read, C[M×N] written).
 *
 * Protocol: docs/METRICS.md — warmups discarded, N repeats timed individually,
 * p50/p95 by nearest-rank percentile.
 *
 * Build (cross or native, same flags as build_device_bench.sh):
 *   gcc -O3 -static -mcpu=cortex-a57 \
 *     bench/bench_delta_matmul.c \
 *     src/orionsbelt/engines/cpu/kernels/gdn_delta_matmul.c \
 *     -I src/orionsbelt/engines/cpu/kernels -lm -o dist/bench_delta_matmul_a57
 *
 *   ./dist/bench_delta_matmul_a57 --repeats 30            # human-readable
 *   ./dist/bench_delta_matmul_a57 --repeats 30 --csv      # CSV to stdout
 */
#define _POSIX_C_SOURCE 200112L  /* clock_gettime, CLOCK_MONOTONIC, rand_r */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "gdn_delta_matmul.h"

/* ----------------------------------------------------------------------- */
/* Timing utilities (identical methodology to bench_gdn.c)                 */
/* ----------------------------------------------------------------------- */

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double pct(double *sorted, int n, double p) {
    if (n <= 0) return 0.0;
    int idx = (int)(p * (double)(n - 1) + 0.5);
    if (idx < 0) idx = 0;
    if (idx >= n) idx = n - 1;
    return sorted[idx];
}

typedef struct {
    double p50, p95;
    int repeats;
} stats_t;

static stats_t summarize(double *samples, int n) {
    qsort(samples, (size_t)n, sizeof(double), cmp_double);
    stats_t s;
    s.repeats = n;
    s.p50 = pct(samples, n, 0.50);
    s.p95 = pct(samples, n, 0.95);
    return s;
}

/* ----------------------------------------------------------------------- */
/* Dispatch path label — mirrors gdn_delta_matmul.c guard order            */
/* ----------------------------------------------------------------------- */

#if defined(__ARM_FEATURE_SVE)
#define DISPATCH_PATH "sve"
#elif defined(__ARM_NEON)
#define DISPATCH_PATH "neon"
#else
#define DISPATCH_PATH "scalar"
#endif

/* ----------------------------------------------------------------------- */
/* Benchmark shapes                                                        */
/* ----------------------------------------------------------------------- */

typedef struct {
    size_t M, K, N;
    const char *label;
} shape_t;

static const shape_t SHAPES[] = {
    {1,  128, 128,  "decode_1x128x128"},
    {64, 128, 128,  "prefill_64x128x128"},
    {1,  128, 2048, "decode_1x128x2048"},
    {64, 128, 2048, "prefill_64x128x2048"},
};
#define NUM_SHAPES (int)(sizeof(SHAPES) / sizeof(SHAPES[0]))

#define WARMUPS 3
#define MAX_REPEATS 256

/* Bytes of traffic per call: read A[M×K] + read B[K×N] + write C[M×N], all fp32 */
static double bytes_per_call(size_t M, size_t K, size_t N) {
    return (double)sizeof(float) * ((double)(M * K) + (double)(K * N) + (double)(M * N));
}

/* ----------------------------------------------------------------------- */
/* Main                                                                    */
/* ----------------------------------------------------------------------- */

int main(int argc, char **argv) {
    int csv = 0;
    int repeats = 30;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--csv")) csv = 1;
        else if (!strcmp(argv[i], "--repeats") && i + 1 < argc) {
            repeats = atoi(argv[++i]);
            if (repeats < 1) repeats = 1;
            if (repeats > MAX_REPEATS) repeats = MAX_REPEATS;
        }
    }

    if (!csv) {
        printf("GDN delta-rule matmul microbenchmark\n");
        printf("  dispatch path              : %s\n", DISPATCH_PATH);
        printf("  shapes                     : %d (decode + prefill, single-head + 16-heads)\n", NUM_SHAPES);
        printf("  repeats per shape          : %d (after %d warmups)\n", repeats, WARMUPS);
        printf("  protocol                   : docs/METRICS.md (p50/p95, nearest-rank)\n\n");
    } else {
        printf("kernel,M,K,N,repeats,p50_us,p95_us,gib_per_s_p50\n");
    }

    for (int si = 0; si < NUM_SHAPES; si++) {
        size_t M = SHAPES[si].M;
        size_t K = SHAPES[si].K;
        size_t N = SHAPES[si].N;

        float *A = malloc(M * K * sizeof(float));
        float *B = malloc(K * N * sizeof(float));
        float *C = malloc(M * N * sizeof(float));
        if (!A || !B || !C) {
            fprintf(stderr, "alloc failure for %s\n", SHAPES[si].label);
            return 1;
        }

        /* Deterministic fill (not timing-critical) */
        unsigned int seed = 42;
        for (size_t i = 0; i < M * K; i++)
            A[i] = ((float)(rand_r(&seed) % 2000) - 1000) / 1000.0f;
        for (size_t i = 0; i < K * N; i++)
            B[i] = ((float)(rand_r(&seed) % 2000) - 1000) / 1000.0f;

        /* Warmups — discarded, per METRICS.md */
        for (int w = 0; w < WARMUPS; w++)
            gdn_delta_rule_matmul(A, B, C, M, K, N);

        /* Timed repeats */
        double samples[MAX_REPEATS];
        for (int r = 0; r < repeats; r++) {
            double t0 = now_s();
            gdn_delta_rule_matmul(A, B, C, M, K, N);
            double t1 = now_s();
            samples[r] = t1 - t0;
        }

        stats_t s = summarize(samples, repeats);
        double bps = bytes_per_call(M, K, N);
        double gibs = s.p50 > 0.0 ? bps / s.p50 / 1073741824.0 : 0.0;

        if (csv) {
            printf("gdn_delta_rule_matmul,%zu,%zu,%zu,%d,%.3f,%.3f,%.2f\n",
                   M, K, N, repeats, s.p50 * 1e6, s.p95 * 1e6, gibs);
        } else {
            printf("  %-22s M=%3zu K=%3zu N=%4zu  "
                   "p50 %10.3f us  p95 %10.3f us  %.2f GiB/s\n",
                   SHAPES[si].label, M, K, N,
                   s.p50 * 1e6, s.p95 * 1e6, gibs);
        }

        free(A);
        free(B);
        free(C);
    }

    return 0;
}
