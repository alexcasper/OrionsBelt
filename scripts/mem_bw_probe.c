/*
 * mem_bw_probe.c — STREAM-like sequential read bandwidth probe.
 *
 * Measures achievable DRAM read bandwidth at varying thread counts.
 * Used to resolve the FINDINGS.md §6 bandwidth-ceiling claim for RK3588.
 *
 * Result on RK3588 (Turing RK1, t3): saturates at ~25 GiB/s (26.8 GB/s)
 * with 3+ A76 threads, confirming ~79% of the 33.8 GB/s theoretical spec
 * at 2112 MHz DMC (64-bit LPDDR4x bus).
 *
 * Build: gcc -O3 -fopenmp -o /tmp/mem_bw_probe scripts/mem_bw_probe.c -lm
 * Usage: /tmp/mem_bw_probe
 *
 * SPDX-License-Identifier: Apache-2.0
 * Origin: t3 characterization measurement, 2026-08-07
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#define REPEATS 20
#define ARRAY_MB 256

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void) {
    size_t bytes = (size_t)ARRAY_MB * 1024 * 1024;
    size_t count = bytes / sizeof(float);
    float *buf = aligned_alloc(64, bytes);
    if (!buf) { perror("alloc"); return 1; }
    memset(buf, 1, bytes);

    printf("=== Memory Bandwidth Probe (%d MB array) ===\n", ARRAY_MB);

    for (int nthreads = 1; nthreads <= 4; nthreads++) {
        omp_set_num_threads(nthreads);
        double best_bw = 0.0;

        for (int r = 0; r < REPEATS; r++) {
            volatile float sink;
            double t0 = now_sec();

            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                size_t chunk = count / nthreads;
                size_t start = (size_t)tid * chunk;
                size_t end = (tid == nthreads - 1) ? count : start + chunk;

                /* 8 independent accumulators to break latency chain */
                float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
                float a4 = 0, a5 = 0, a6 = 0, a7 = 0;
                size_t i = start;
                for (; i + 7 < end; i += 8) {
                    a0 += buf[i+0]; a1 += buf[i+1];
                    a2 += buf[i+2]; a3 += buf[i+3];
                    a4 += buf[i+4]; a5 += buf[i+5];
                    a6 += buf[i+6]; a7 += buf[i+7];
                }
                for (; i < end; i++) a0 += buf[i];
                sink = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
                (void)sink;
            }

            double t1 = now_sec();
            double secs = t1 - t0;
            if (secs > 0) {
                double gbs = (double)bytes / secs / 1e9;
                if (gbs > best_bw) best_bw = gbs;
            }
        }
        double gib_s = best_bw / 1.073741824;
        printf("  %d thread(s):  %.1f GiB/s  (%.1f GB/s)\n", nthreads, gib_s, best_bw);
    }

    free(buf);
    return 0;
}
