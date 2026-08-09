// SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
// SPDX-License-Identifier: Apache-2.0

/*
 * Engine boundary-crossing cost micro-benchmark (bead ob-t3b.6).
 *
 * Measures host↔device transfer latency for small tensors matching the
 * payloads that cross engine boundaries in a heterogeneous GDN deployment.
 * In a 3:1 hybrid (24 GDN layers on CPU, 8 attention on GPU/NPU), the
 * hidden state (~5KB at hidden_size=2560 fp16) crosses the CPU↔GPU boundary
 * at each layer handoff — up to 16 times per decoded token.
 *
 * This benchmark pre-validates the measurement methodology on whatever GPU
 * is available (Mali-G610 on RK3588 as a development proxy per ADR 0005),
 * so ob-t3b.3 on the Orion O6 is a re-run, not a from-scratch effort.
 *
 * Build:
 *   gcc -O2 -o boundary_crossing_bench boundary_crossing_bench.c -lOpenCL -lm
 *
 * Run:
 *   RUSTICL_ENABLE=panfrost ./gpu/boundary_crossing_bench
 *   RUSTICL_ENABLE=panfrost ./gpu/boundary_crossing_bench --csv
 *   RUSTICL_ENABLE=panfrost ./gpu/boundary_crossing_bench --repeats 100
 */

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ------------------------------------------------------------------ */
/* Globals                                                             */
/* ------------------------------------------------------------------ */

static cl_platform_id   g_platform;
static cl_device_id     g_device;
static cl_context       g_ctx;
static cl_command_queue g_queue;
static int              g_use_profiling = 0;
static char             g_dev_name[256];
static char             g_dev_version[256];

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

static void cl_check(cl_int err, const char *what) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "OpenCL error %d in %s\n", err, what);
        exit(1);
    }
}

static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) { fprintf(stderr, "out of memory (%zu bytes)\n", n); exit(1); }
    return p;
}

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

static double event_us(cl_event ev) {
    cl_ulong t_start, t_end;
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START,
                            sizeof(t_start), &t_start, NULL);
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,
                            sizeof(t_end), &t_end, NULL);
    clReleaseEvent(ev);
    return (double)(t_end - t_start) * 1e-3;  /* ns → µs */
}

/* ------------------------------------------------------------------ */
/* OpenCL init (minimal — no kernel program needed)                    */
/* ------------------------------------------------------------------ */

static void init_opencl(void) {
    cl_int err;

    cl_uint nplat;
    err = clGetPlatformIDs(1, &g_platform, &nplat);
    cl_check(err, "clGetPlatformIDs");
    if (nplat == 0) { fprintf(stderr, "No OpenCL platforms found.\n"); exit(1); }

    err = clGetDeviceIDs(g_platform, CL_DEVICE_TYPE_GPU, 1, &g_device, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "No GPU device, trying ALL...\n");
        err = clGetDeviceIDs(g_platform, CL_DEVICE_TYPE_ALL, 1, &g_device, NULL);
    }
    cl_check(err, "clGetDeviceIDs");

    size_t sz;
    clGetDeviceInfo(g_device, CL_DEVICE_NAME, sizeof(g_dev_name), g_dev_name, &sz);
    g_dev_name[sz] = '\0';
    clGetDeviceInfo(g_device, CL_DEVICE_VERSION, sizeof(g_dev_version),
                    g_dev_version, &sz);
    g_dev_version[sz] = '\0';

    g_ctx = clCreateContext(NULL, 1, &g_device, NULL, NULL, &err);
    cl_check(err, "clCreateContext");

    /* Profiling queue — fall back to wall-clock if driver rejects it. */
    const cl_queue_properties qprops[] = {CL_QUEUE_PROPERTIES,
        CL_QUEUE_PROFILING_ENABLE, 0};
    g_queue = clCreateCommandQueueWithProperties(g_ctx, g_device, qprops, &err);
    if (err != CL_SUCCESS) {
        g_use_profiling = 0;
        g_queue = clCreateCommandQueueWithProperties(g_ctx, g_device, NULL, &err);
        cl_check(err, "clCreateCommandQueueWithProperties (no profiling)");
        fprintf(stderr, "  [info] GPU profiling not supported; using wall-clock timing\n");
    } else {
        g_use_profiling = 1;
    }

    fprintf(stderr, "OpenCL device: %s (%s)\n", g_dev_name, g_dev_version);
    fprintf(stderr, "Profiling: %s\n\n", g_use_profiling ? "enabled" : "wall-clock");
}

/* ------------------------------------------------------------------ */
/* Transfer measurements                                               */
/* ------------------------------------------------------------------ */

#define WARMUPS 5
#define MAX_REPEATS 512

/*
 * Measure blocking write (host→device).
 * clEnqueueWriteBuffer with CL_TRUE blocks until the write completes.
 */
static double meas_write_blocking(size_t nbytes, int repeats) {
    void *host = xmalloc(nbytes);
    memset(host, 0xAA, nbytes);

    cl_int err;
    cl_mem buf = clCreateBuffer(g_ctx, CL_MEM_READ_ONLY, nbytes, NULL, &err);
    cl_check(err, "clCreateBuffer");

    /* warmup */
    for (int i = 0; i < WARMUPS; i++)
        clEnqueueWriteBuffer(g_queue, buf, CL_TRUE, 0, nbytes, host, 0, NULL, NULL);

    double *samples = xmalloc(repeats * sizeof(double));
    for (int i = 0; i < repeats; i++) {
        double t0 = now_s();
        clEnqueueWriteBuffer(g_queue, buf, CL_TRUE, 0, nbytes, host, 0, NULL, NULL);
        double t1 = now_s();
        samples[i] = (t1 - t0) * 1e6;  /* s → µs */
    }
    qsort(samples, repeats, sizeof(double), cmp_double);
    double p50 = pct(samples, repeats, 0.50);
    free(samples);

    clReleaseMemObject(buf);
    free(host);
    return p50;
}

/*
 * Measure non-blocking write with profiling event (device-side timing).
 * This captures the actual transfer time on the GPU, not including
 * the host-side overhead of enqueuing.
 */
static double meas_write_event(size_t nbytes, int repeats) {
    if (!g_use_profiling) return -1.0;

    void *host = xmalloc(nbytes);
    memset(host, 0xBB, nbytes);

    cl_int err;
    cl_mem buf = clCreateBuffer(g_ctx, CL_MEM_READ_ONLY, nbytes, NULL, &err);
    cl_check(err, "clCreateBuffer");

    for (int i = 0; i < WARMUPS; i++) {
        cl_event ev;
        clEnqueueWriteBuffer(g_queue, buf, CL_FALSE, 0, nbytes, host, 0, NULL, &ev);
        clFinish(g_queue);
        clReleaseEvent(ev);
    }

    double *samples = xmalloc(repeats * sizeof(double));
    for (int i = 0; i < repeats; i++) {
        cl_event ev;
        clEnqueueWriteBuffer(g_queue, buf, CL_FALSE, 0, nbytes, host, 0, NULL, &ev);
        clFinish(g_queue);
        samples[i] = event_us(ev);
    }
    qsort(samples, repeats, sizeof(double), cmp_double);
    double p50 = pct(samples, repeats, 0.50);
    free(samples);

    clReleaseMemObject(buf);
    free(host);
    return p50;
}

/*
 * Measure blocking read (device→host).
 */
static double meas_read_blocking(size_t nbytes, int repeats) {
    cl_int err;
    cl_mem buf = clCreateBuffer(g_ctx, CL_MEM_WRITE_ONLY, nbytes, NULL, &err);
    cl_check(err, "clCreateBuffer");

    void *host = xmalloc(nbytes);

    for (int i = 0; i < WARMUPS; i++)
        clEnqueueReadBuffer(g_queue, buf, CL_TRUE, 0, nbytes, host, 0, NULL, NULL);

    double *samples = xmalloc(repeats * sizeof(double));
    for (int i = 0; i < repeats; i++) {
        double t0 = now_s();
        clEnqueueReadBuffer(g_queue, buf, CL_TRUE, 0, nbytes, host, 0, NULL, NULL);
        double t1 = now_s();
        samples[i] = (t1 - t0) * 1e6;
    }
    qsort(samples, repeats, sizeof(double), cmp_double);
    double p50 = pct(samples, repeats, 0.50);
    free(samples);

    clReleaseMemObject(buf);
    free(host);
    return p50;
}

/*
 * Measure round-trip: write + read (one boundary crossing each direction).
 * In the 3:1 hybrid, a boundary crossing is a write to the other engine
 * followed eventually by a read-back. Round-trip captures the full overhead
 * of handing a tensor off and getting the result back.
 */
static double meas_roundtrip_blocking(size_t nbytes, int repeats) {
    cl_int err;
    cl_mem buf = clCreateBuffer(g_ctx, CL_MEM_READ_WRITE, nbytes, NULL, &err);
    cl_check(err, "clCreateBuffer");

    void *host_w = xmalloc(nbytes);
    void *host_r = xmalloc(nbytes);
    memset(host_w, 0xCC, nbytes);

    for (int i = 0; i < WARMUPS; i++) {
        clEnqueueWriteBuffer(g_queue, buf, CL_TRUE, 0, nbytes, host_w, 0, NULL, NULL);
        clEnqueueReadBuffer(g_queue, buf, CL_TRUE, 0, nbytes, host_r, 0, NULL, NULL);
    }

    double *samples = xmalloc(repeats * sizeof(double));
    for (int i = 0; i < repeats; i++) {
        double t0 = now_s();
        clEnqueueWriteBuffer(g_queue, buf, CL_TRUE, 0, nbytes, host_w, 0, NULL, NULL);
        clEnqueueReadBuffer(g_queue, buf, CL_TRUE, 0, nbytes, host_r, 0, NULL, NULL);
        double t1 = now_s();
        samples[i] = (t1 - t0) * 1e6;
    }
    qsort(samples, repeats, sizeof(double), cmp_double);
    double p50 = pct(samples, repeats, 0.50);
    free(samples);

    clReleaseMemObject(buf);
    free(host_w);
    free(host_r);
    return p50;
}

/*
 * Simulate N consecutive crossings with a no-op GPU kernel in between
 * to model the actual decode pattern: write → compute → read → write → ...
 * Returns the total latency for N crossings in µs.
 */
static double meas_n_crossings(size_t nbytes, int n_crossings, int repeats) {
    cl_int err;
    cl_mem buf = clCreateBuffer(g_ctx, CL_MEM_READ_WRITE, nbytes, NULL, &err);
    cl_check(err, "clCreateBuffer");

    void *host = xmalloc(nbytes);
    memset(host, 0xDD, nbytes);

    /* warmup */
    for (int i = 0; i < WARMUPS; i++) {
        clEnqueueWriteBuffer(g_queue, buf, CL_TRUE, 0, nbytes, host, 0, NULL, NULL);
        clEnqueueReadBuffer(g_queue, buf, CL_TRUE, 0, nbytes, host, 0, NULL, NULL);
    }

    double *samples = xmalloc(repeats * sizeof(double));
    for (int i = 0; i < repeats; i++) {
        double t0 = now_s();
        for (int j = 0; j < n_crossings; j++) {
            clEnqueueWriteBuffer(g_queue, buf, CL_TRUE, 0, nbytes, host, 0, NULL, NULL);
            clEnqueueReadBuffer(g_queue, buf, CL_TRUE, 0, nbytes, host, 0, NULL, NULL);
        }
        double t1 = now_s();
        samples[i] = (t1 - t0) * 1e6;
    }
    qsort(samples, repeats, sizeof(double), cmp_double);
    double p50 = pct(samples, repeats, 0.50);
    free(samples);

    clReleaseMemObject(buf);
    free(host);
    return p50;
}

/* ------------------------------------------------------------------ */
/* Main                                                                */
/* ------------------------------------------------------------------ */

/* Payload sizes — from small control packets to the actual decode tensor. */
static const struct {
    size_t bytes;
    const char *label;
} payloads[] = {
    { 512,        "512B" },
    { 1024,       "1KB" },
    { 5120,       "5KB_hidden_fp16" },   /* hidden_size 2560 × 2 bytes */
    { 10240,      "10KB_hidden_fp32" },  /* hidden_size 2560 × 4 bytes */
    { 51200,      "50KB" },
    { 102400,     "100KB" },
    { 1048576,    "1MB" },
    { 0, NULL }
};

int main(int argc, char **argv) {
    int repeats = 100;
    int csv_mode = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--csv") == 0) csv_mode = 1;
        if (strcmp(argv[i], "--repeats") == 0 && i + 1 < argc)
            repeats = atoi(argv[++i]);
    }
    if (repeats > MAX_REPEATS) repeats = MAX_REPEATS;

    init_opencl();

    /* --- 16-crossing simulation (the 3:1 hybrid decode pattern) --- */
    const int n_crossings = 16;
    const size_t decode_payload = 5120;  /* 5KB fp16 hidden state */

    if (!csv_mode) {
        fprintf(stderr, "=== Engine Boundary-Crossing Cost ===\n");
        fprintf(stderr, "Repeats: %d  (warmups: %d)\n\n", repeats, WARMUPS);

        fprintf(stderr, "--- Per-payload transfer latency (p50) ---\n");
        fprintf(stderr, "%-20s %10s %10s %10s %12s\n",
               "Payload", "Write_us", "Read_us", "Round_us", "WriteEv_us");
        fprintf(stderr, "%-20s %10s %10s %10s %12s\n",
               "-------", "-------", "------", "-------", "---------");
    } else {
        printf("kernel,dim1,dim2,dim3,p50_ms,p95_ms,bw_mibs\n");
    }

    for (int i = 0; payloads[i].bytes > 0; i++) {
        size_t sz = payloads[i].bytes;

        double wr_p50  = meas_write_blocking(sz, repeats);
        double rd_p50  = meas_read_blocking(sz, repeats);
        double rt_p50  = meas_roundtrip_blocking(sz, repeats);
        double wr_ev   = meas_write_event(sz, repeats);

        if (!csv_mode) {
            fprintf(stderr, "%-20s %10.2f %10.2f %10.2f",
                   payloads[i].label, wr_p50, rd_p50, rt_p50);
            if (wr_ev > 0)
                fprintf(stderr, " %12.2f", wr_ev);
            else
                fprintf(stderr, " %12s", "n/a");
            fprintf(stderr, "\n");
        } else {
            /* CSV: three "kernels" per payload (write/read/roundtrip) */
            double wr_bw = (sz / (wr_p50 * 1e-6)) / (1024.0 * 1024.0);  /* MiB/s */
            double rd_bw = (sz / (rd_p50 * 1e-6)) / (1024.0 * 1024.0);
            printf("write_blocking,%s,,,%6.4f,0,%.1f\n",
                   payloads[i].label, wr_p50 / 1000.0, wr_bw);
            printf("read_blocking,%s,,,%6.4f,0,%.1f\n",
                   payloads[i].label, rd_p50 / 1000.0, rd_bw);
            printf("roundtrip_blocking,%s,,,%6.4f,0,0\n",
                   payloads[i].label, rt_p50 / 1000.0);
        }
    }

    /* --- 16-crossing simulation --- */
    double x16_total = meas_n_crossings(decode_payload, n_crossings, repeats);
    double x16_per   = x16_total / n_crossings;
    double x16_overhead_pct = x16_total / 1e6 * 1000.0; /* total ms for 16 crossings */

    if (!csv_mode) {
        fprintf(stderr, "\n--- 3:1 Hybrid Decode Simulation ---\n");
        fprintf(stderr, "Payload per crossing: 5KB (hidden_size=2560 fp16)\n");
        fprintf(stderr, "Crossings per token:  %d\n", n_crossings);
        fprintf(stderr, "Total latency:        %.2f µs (%.3f ms)\n", x16_total, x16_overhead_pct);
        fprintf(stderr, "Per-crossing:         %.2f µs\n", x16_per);
        fprintf(stderr, "\nContext: at 30 tokens/s decode, each token has %.3f ms budget.\n",
               1000.0 / 30.0);
        fprintf(stderr, "16 crossings take %.3f ms = %.1f%% of token budget.\n",
               x16_overhead_pct, x16_overhead_pct / (1000.0 / 30.0) * 100.0);
    } else {
        printf("n_crossings_16,%d,,,%6.4f,0,0\n", n_crossings,
               x16_total / 1000.0);
    }

    if (!csv_mode) {
        fprintf(stderr, "\nDone.\n");
    }

    return 0;
}
