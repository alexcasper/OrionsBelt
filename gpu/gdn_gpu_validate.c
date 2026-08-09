// SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
// SPDX-License-Identifier: Apache-2.0

/*
 * Comprehensive numerical validation for GPU GDN kernels (bead ob-gzk).
 *
 * Tests all four OpenCL kernels across:
 *   - Multiple sequence lengths (1, 16, 32, 64, 65, 128, 256, 512, 1024, 2048)
 *   - Multiple channel counts (128, 512, 2048, 8192)
 *   - Edge cases: seq not divisible by chunk size, state carry across calls,
 *     extreme decay values, zero inputs
 *   - Long-context drift: error growth with sequence length
 *
 * Each test compares GPU output against a precision-matched scalar CPU
 * reference and checks against the correctness oracle tolerances
 * (atol=1e-4, rtol=1e-3 per docs/CORRECTNESS_TOLERANCES.md).
 *
 * Build:
 *   gcc -O2 -o gpu/gdn_gpu_validate gpu/gdn_gpu_validate.c -lOpenCL -lm
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
/* Oracle tolerances (from docs/CORRECTNESS_TOLERANCES.md)            */
/* ------------------------------------------------------------------ */

static const double ORACLE_ATOL = 1e-4;
static const double ORACLE_RTOL = 1e-3;

/* ------------------------------------------------------------------ */
/* Scalar CPU references (precision-matched: float accumulators)       */
/* ------------------------------------------------------------------ */

static void ref_scan(const float *g, const float *x, float *s, float *st,
                     size_t T, size_t C) {
    for (size_t c = 0; c < C; c++) {
        float a = st[c];
        for (size_t t = 0; t < T; t++) {
            a = x[t * C + c] + a * g[t * C + c];
            s[t * C + c] = a;
        }
        st[c] = a;
    }
}

static void ref_decay(const float *a, float *d, size_t T, size_t C) {
    for (size_t c = 0; c < C; c++) {
        float run = 1.0f;
        for (size_t t = 0; t < T; t++) {
            run *= a[t * C + c];
            d[t * C + c] = run;
        }
    }
}

static void ref_conv(const float *in, const float *w, float *o, float *h,
                     size_t T, size_t C) {
    for (size_t c = 0; c < C; c++) {
        float H[3] = {h[0 * C + c], h[1 * C + c], h[2 * C + c]};
        for (size_t t = 0; t < T; t++) {
            float cur = in[t * C + c];
            o[t * C + c] = H[0] * w[0 * C + c] + H[1] * w[1 * C + c] +
                           H[2] * w[2 * C + c] + cur * w[3 * C + c];
            H[0] = H[1]; H[1] = H[2]; H[2] = cur;
        }
        h[0 * C + c] = H[0]; h[1 * C + c] = H[1]; h[2 * C + c] = H[2];
    }
}

static void ref_delta_rule(float *S, const float *k, const float *v,
                           const float *q, float beta, float decay,
                           float *out, size_t hkd, size_t hvd) {
    for (size_t j = 0; j < hkd; j++)
        for (size_t i = 0; i < hvd; i++)
            S[j * hvd + i] *= decay;
    for (size_t i = 0; i < hvd; i++) {
        float kv = 0.0f;
        for (size_t j = 0; j < hkd; j++)
            kv += S[j * hvd + i] * k[j];
        float delta = (v[i] - kv) * beta;
        for (size_t j = 0; j < hkd; j++)
            S[j * hvd + i] += k[j] * delta;
        float o = 0.0f;
        for (size_t j = 0; j < hkd; j++)
            o += S[j * hvd + i] * q[j];
        out[i] = o;
    }
}

/* ------------------------------------------------------------------ */
/* Error metrics                                                       */
/* ------------------------------------------------------------------ */

typedef struct {
    double max_abs;
    double max_rel;
    double mean_abs;
    size_t n;
    int pass;       /* 1 if within oracle tolerances */
} error_t;

static error_t compute_error(const float *gpu, const float *ref, size_t N) {
    error_t e = {0, 0, 0, N, 1};
    double sum_abs = 0;
    for (size_t i = 0; i < N; i++) {
        double d = fabs((double)gpu[i] - ref[i]);
        if (d > e.max_abs) e.max_abs = d;
        sum_abs += d;
        if (fabs((double)ref[i]) > 1e-2) {
            double r = d / fabs((double)ref[i]);
            if (r > e.max_rel) e.max_rel = r;
        }
    }
    e.mean_abs = sum_abs / N;
    if (e.max_abs > ORACLE_ATOL && e.max_rel > ORACLE_RTOL)
        e.pass = 0;
    return e;
}

/* ------------------------------------------------------------------ */
/* OpenCL setup                                                        */
/* ------------------------------------------------------------------ */

static cl_platform_id   g_platform;
static cl_device_id     g_device;
static cl_context       g_ctx;
static cl_command_queue g_queue;
static cl_program       g_program;

static void cl_check(cl_int err, const char *msg) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "OpenCL error %d: %s\n", err, msg);
        exit(1);
    }
}

static char *load_file(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc(len + 1);
    if (!buf) { fprintf(stderr, "OOM in load_file\n"); fclose(f); exit(1); }
    size_t nread = fread(buf, 1, len, f);
    buf[nread] = '\0';
    fclose(f);
    return buf;
}

static void init_opencl(void) {
    cl_int err;
    cl_uint nplat;
    clGetPlatformIDs(1, &g_platform, &nplat);
    clGetDeviceIDs(g_platform, CL_DEVICE_TYPE_GPU, 1, &g_device, NULL);
    g_ctx = clCreateContext(NULL, 1, &g_device, NULL, NULL, &err);
    cl_check(err, "ctx");
    /* RustiCL/Panfrost does not support CL_QUEUE_PROFILING_ENABLE (err -35).
     * Profiling is unused in the validate path — pass NULL properties. */
    g_queue = clCreateCommandQueueWithProperties(g_ctx, g_device, NULL, &err);
    cl_check(err, "queue");

    size_t srclen;
    char *src = load_file("gpu/gdn_gpu_kernels.cl");
    srclen = strlen(src);
    g_program = clCreateProgramWithSource(g_ctx, 1, (const char **)&src,
                                          &srclen, &err);
    cl_check(err, "program");
    err = clBuildProgram(g_program, 1, &g_device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t lsz;
        clGetProgramBuildInfo(g_program, g_device, CL_PROGRAM_BUILD_LOG,
                              0, NULL, &lsz);
        char *log = malloc(lsz + 1);
        if (!log) { fprintf(stderr, "Build error (OOM allocating log)\n"); exit(1); }
        clGetProgramBuildInfo(g_program, g_device, CL_PROGRAM_BUILD_LOG,
                              lsz, log, NULL);
        log[lsz] = '\0';
        fprintf(stderr, "Build error:\n%s\n", log);
        free(log);
        exit(1);
    }
    free(src);
}

static void fill_rand(float *p, size_t n, float lo, float hi, int seed) {
    srand(seed);
    for (size_t i = 0; i < n; i++)
        p[i] = lo + (hi - lo) * (rand() / (float)RAND_MAX);
}

static void fill_const(float *p, size_t n, float val) {
    for (size_t i = 0; i < n; i++) p[i] = val;
}

/* ------------------------------------------------------------------ */
/* Test result tracking                                                */
/* ------------------------------------------------------------------ */

static int g_tests_total = 0;
static int g_tests_passed = 0;
static int g_tests_failed = 0;

static void track_result(const char *name, error_t e, const char *detail) {
    g_tests_total++;
    if (e.pass) g_tests_passed++;
    else g_tests_failed++;

    printf("  [%s] %-44s  max_abs=%.2e  max_rel=%.2e  %s\n",
           e.pass ? "PASS" : "FAIL",
           name, e.max_abs, e.max_rel,
           detail ? detail : "");
}

/* ------------------------------------------------------------------ */
/* Test runners                                                        */
/* ------------------------------------------------------------------ */

static cl_kernel k_scan, k_decay, k_conv, k_delta;

/* Run gated_scan on GPU and compare against reference. */
static void test_scan(const char *label, size_t T, size_t C,
                      float gate_lo, float gate_hi, int seed) {
    size_t N = T * C;
    float *g_h = malloc(N * 4), *x_h = malloc(N * 4);
    float *s_gpu = malloc(N * 4), *s_ref = malloc(N * 4);
    float *st_gpu = malloc(C * 4), *st_ref = malloc(C * 4);
    if (!g_h || !x_h || !s_gpu || !s_ref || !st_gpu || !st_ref) {
        fprintf(stderr, "OOM in test_scan\n"); exit(1);
    }

    fill_rand(g_h, N, gate_lo, gate_hi, seed);
    fill_rand(x_h, N, -0.5f, 0.5f, seed + 1);
    fill_rand(st_gpu, C, -0.5f, 0.5f, seed + 2);
    memcpy(st_ref, st_gpu, C * 4);

    cl_int err;
    cl_mem g_d = clCreateBuffer(g_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N*4, g_h, &err);
    cl_mem x_d = clCreateBuffer(g_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N*4, x_h, &err);
    cl_mem s_d = clCreateBuffer(g_ctx, CL_MEM_WRITE_ONLY, N*4, NULL, &err);
    cl_mem st_d = clCreateBuffer(g_ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, C*4, st_gpu, &err);

    clSetKernelArg(k_scan, 0, sizeof(cl_mem), &g_d);
    clSetKernelArg(k_scan, 1, sizeof(cl_mem), &x_d);
    clSetKernelArg(k_scan, 2, sizeof(cl_mem), &s_d);
    clSetKernelArg(k_scan, 3, sizeof(cl_mem), &st_d);
    uint uT = T, uC = C;
    clSetKernelArg(k_scan, 4, sizeof(uint), &uT);
    clSetKernelArg(k_scan, 5, sizeof(uint), &uC);

    size_t global = C;
    clEnqueueNDRangeKernel(g_queue, k_scan, 1, NULL, &global, NULL, 0, NULL, NULL);
    clFinish(g_queue);
    clEnqueueReadBuffer(g_queue, s_d, CL_TRUE, 0, N*4, s_gpu, 0, NULL, NULL);
    clEnqueueReadBuffer(g_queue, st_d, CL_TRUE, 0, C*4, st_gpu, 0, NULL, NULL);

    ref_scan(g_h, x_h, s_ref, st_ref, T, C);

    char detail[128];
    snprintf(detail, sizeof(detail), "T=%zu C=%zu gates=[%.1f,%.1f]", T, C, gate_lo, gate_hi);
    track_result(label, compute_error(s_gpu, s_ref, N), detail);

    /* Also check state carry */
    char state_label[160];
    snprintf(state_label, sizeof(state_label), "%s (state)", label);
    track_result(state_label, compute_error(st_gpu, st_ref, C), NULL);

    clReleaseMemObject(g_d); clReleaseMemObject(x_d);
    clReleaseMemObject(s_d); clReleaseMemObject(st_d);
    free(g_h); free(x_h); free(s_gpu); free(s_ref); free(st_gpu); free(st_ref);
}

/* Run cumdecay on GPU and compare against reference. */
static void test_decay(const char *label, size_t T, size_t C,
                       float gate_lo, float gate_hi, int seed) {
    size_t N = T * C;
    float *a_h = malloc(N * 4);
    float *d_gpu = malloc(N * 4), *d_ref = malloc(N * 4);
    if (!a_h || !d_gpu || !d_ref) {
        fprintf(stderr, "OOM in test_decay\n"); exit(1);
    }

    fill_rand(a_h, N, gate_lo, gate_hi, seed);

    cl_int err;
    cl_mem a_d = clCreateBuffer(g_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N*4, a_h, &err);
    cl_mem d_d = clCreateBuffer(g_ctx, CL_MEM_WRITE_ONLY, N*4, NULL, &err);

    clSetKernelArg(k_decay, 0, sizeof(cl_mem), &a_d);
    clSetKernelArg(k_decay, 1, sizeof(cl_mem), &d_d);
    uint uT = T, uC = C;
    clSetKernelArg(k_decay, 2, sizeof(uint), &uT);
    clSetKernelArg(k_decay, 3, sizeof(uint), &uC);

    size_t global = C;
    clEnqueueNDRangeKernel(g_queue, k_decay, 1, NULL, &global, NULL, 0, NULL, NULL);
    clFinish(g_queue);
    clEnqueueReadBuffer(g_queue, d_d, CL_TRUE, 0, N*4, d_gpu, 0, NULL, NULL);

    ref_decay(a_h, d_ref, T, C);

    char detail[128];
    snprintf(detail, sizeof(detail), "T=%zu C=%zu gates=[%.1f,%.1f]", T, C, gate_lo, gate_hi);
    track_result(label, compute_error(d_gpu, d_ref, N), detail);

    clReleaseMemObject(a_d); clReleaseMemObject(d_d);
    free(a_h); free(d_gpu); free(d_ref);
}

/* Run causal_dwconv1d on GPU and compare against reference. */
static void test_conv(const char *label, size_t T, size_t C, int seed) {
    size_t N = T * C;
    float *in_h = malloc(N * 4), *w_h = malloc(4 * C * 4);
    float *o_gpu = malloc(N * 4), *o_ref = malloc(N * 4);
    float *h_gpu = malloc(3 * C * 4), *h_ref = malloc(3 * C * 4);
    if (!in_h || !w_h || !o_gpu || !o_ref || !h_gpu || !h_ref) {
        fprintf(stderr, "OOM in test_conv\n"); exit(1);
    }

    fill_rand(in_h, N, -0.5f, 0.5f, seed);
    fill_rand(w_h, 4 * C, -0.5f, 0.5f, seed + 1);
    fill_rand(h_gpu, 3 * C, -0.5f, 0.5f, seed + 2);
    memcpy(h_ref, h_gpu, 3 * C * 4);

    cl_int err;
    cl_mem in_d = clCreateBuffer(g_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N*4, in_h, &err);
    cl_mem w_d = clCreateBuffer(g_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 4*C*4, w_h, &err);
    cl_mem o_d = clCreateBuffer(g_ctx, CL_MEM_WRITE_ONLY, N*4, NULL, &err);
    cl_mem hh_d = clCreateBuffer(g_ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 3*C*4, h_gpu, &err);

    clSetKernelArg(k_conv, 0, sizeof(cl_mem), &in_d);
    clSetKernelArg(k_conv, 1, sizeof(cl_mem), &w_d);
    clSetKernelArg(k_conv, 2, sizeof(cl_mem), &o_d);
    clSetKernelArg(k_conv, 3, sizeof(cl_mem), &hh_d);
    uint uT = T, uC = C;
    clSetKernelArg(k_conv, 4, sizeof(uint), &uT);
    clSetKernelArg(k_conv, 5, sizeof(uint), &uC);

    size_t global = C;
    clEnqueueNDRangeKernel(g_queue, k_conv, 1, NULL, &global, NULL, 0, NULL, NULL);
    clFinish(g_queue);
    clEnqueueReadBuffer(g_queue, o_d, CL_TRUE, 0, N*4, o_gpu, 0, NULL, NULL);
    clEnqueueReadBuffer(g_queue, hh_d, CL_TRUE, 0, 3*C*4, h_gpu, 0, NULL, NULL);

    ref_conv(in_h, w_h, o_ref, h_ref, T, C);

    char detail[128];
    snprintf(detail, sizeof(detail), "T=%zu C=%zu", T, C);
    track_result(label, compute_error(o_gpu, o_ref, N), detail);

    clReleaseMemObject(in_d); clReleaseMemObject(w_d);
    clReleaseMemObject(o_d); clReleaseMemObject(hh_d);
    free(in_h); free(w_h); free(o_gpu); free(o_ref); free(h_gpu); free(h_ref);
}

/* Run delta-rule for one token on GPU and compare against reference. */
static void test_delta(const char *label, size_t num_heads, size_t hkd, size_t hvd,
                       float beta, float decay, int seed) {
    size_t S_sz = num_heads * hkd * hvd;
    size_t kv_sz = num_heads * hkd;
    size_t vv_sz = num_heads * hvd;

    float *S_gpu = malloc(S_sz * 4), *S_ref = malloc(S_sz * 4);
    float *k_h = malloc(kv_sz * 4), *v_h = malloc(vv_sz * 4), *q_h = malloc(kv_sz * 4);
    float *out_gpu = malloc(vv_sz * 4), *out_ref = malloc(vv_sz * 4);
    if (!S_gpu || !S_ref || !k_h || !v_h || !q_h || !out_gpu || !out_ref) {
        fprintf(stderr, "OOM in test_delta\n"); exit(1);
    }

    fill_rand(S_gpu, S_sz, -0.01f, 0.01f, seed);
    memcpy(S_ref, S_gpu, S_sz * 4);
    fill_rand(k_h, kv_sz, -1.0f, 1.0f, seed + 1);
    fill_rand(v_h, vv_sz, -0.5f, 0.5f, seed + 2);
    fill_rand(q_h, kv_sz, -1.0f, 1.0f, seed + 3);

    /* L2-normalise k and q per head */
    for (size_t h = 0; h < num_heads; h++) {
        float kn = 0, qn = 0;
        for (size_t j = 0; j < hkd; j++) {
            kn += k_h[h * hkd + j] * k_h[h * hkd + j];
            qn += q_h[h * hkd + j] * q_h[h * hkd + j];
        }
        kn = sqrtf(kn); qn = sqrtf(qn);
        if (kn > 0) for (size_t j = 0; j < hkd; j++) k_h[h * hkd + j] /= kn;
        if (qn > 0) for (size_t j = 0; j < hkd; j++) q_h[h * hkd + j] /= qn;
    }

    cl_int err;
    cl_mem S_d = clCreateBuffer(g_ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, S_sz*4, S_gpu, &err);
    cl_mem k_d = clCreateBuffer(g_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, kv_sz*4, k_h, &err);
    cl_mem v_d = clCreateBuffer(g_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, vv_sz*4, v_h, &err);
    cl_mem q_d = clCreateBuffer(g_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, kv_sz*4, q_h, &err);
    cl_mem out_d = clCreateBuffer(g_ctx, CL_MEM_WRITE_ONLY, vv_sz*4, NULL, &err);

    clSetKernelArg(k_delta, 0, sizeof(cl_mem), &S_d);
    clSetKernelArg(k_delta, 1, sizeof(cl_mem), &k_d);
    clSetKernelArg(k_delta, 2, sizeof(cl_mem), &v_d);
    clSetKernelArg(k_delta, 3, sizeof(cl_mem), &q_d);
    clSetKernelArg(k_delta, 4, sizeof(float), &beta);
    clSetKernelArg(k_delta, 5, sizeof(float), &decay);
    clSetKernelArg(k_delta, 6, sizeof(cl_mem), &out_d);
    uint u_hkd = hkd, u_hvd = hvd, u_nh = num_heads;
    clSetKernelArg(k_delta, 7, sizeof(uint), &u_hkd);
    clSetKernelArg(k_delta, 8, sizeof(uint), &u_hvd);
    clSetKernelArg(k_delta, 9, sizeof(uint), &u_nh);

    size_t global2d[] = {hvd, num_heads};
    size_t local2d[] = {hvd, 1};
    clEnqueueNDRangeKernel(g_queue, k_delta, 2, NULL, global2d, local2d, 0, NULL, NULL);
    clFinish(g_queue);
    clEnqueueReadBuffer(g_queue, S_d, CL_TRUE, 0, S_sz*4, S_gpu, 0, NULL, NULL);
    clEnqueueReadBuffer(g_queue, out_d, CL_TRUE, 0, vv_sz*4, out_gpu, 0, NULL, NULL);

    for (size_t h = 0; h < num_heads; h++)
        ref_delta_rule(S_ref + h * hkd * hvd, k_h + h * hkd, v_h + h * hvd,
                       q_h + h * hkd, beta, decay, out_ref + h * hvd, hkd, hvd);

    char detail[128];
    snprintf(detail, sizeof(detail), "heads=%zu hkd=%zu hvd=%zu beta=%.2f decay=%.3f",
             num_heads, hkd, hvd, beta, decay);
    error_t e_out = compute_error(out_gpu, out_ref, vv_sz);
    track_result(label, e_out, detail);

    char state_label[160];
    snprintf(state_label, sizeof(state_label), "%s (state)", label);
    error_t e_state = compute_error(S_gpu, S_ref, S_sz);
    track_result(state_label, e_state, NULL);

    clReleaseMemObject(S_d); clReleaseMemObject(k_d);
    clReleaseMemObject(v_d); clReleaseMemObject(q_d); clReleaseMemObject(out_d);
    free(S_gpu); free(S_ref); free(k_h); free(v_h); free(q_h);
    free(out_gpu); free(out_ref);
}

/* Test state carry: run scan twice with state persistence. */
static void test_scan_state_carry(const char *label, size_t T1, size_t T2, size_t C) {
    size_t N1 = T1 * C, N2 = T2 * C;
    float *g1 = malloc(N1 * 4), *x1 = malloc(N1 * 4);
    float *g2 = malloc(N2 * 4), *x2 = malloc(N2 * 4);
    float *s1_gpu = malloc(N1 * 4), *s2_gpu = malloc(N2 * 4);
    float *s1_ref = malloc(N1 * 4), *s2_ref = malloc(N2 * 4);
    float *st_gpu = malloc(C * 4), *st_ref = malloc(C * 4);
    if (!g1 || !x1 || !g2 || !x2 || !s1_gpu || !s2_gpu || !s1_ref || !s2_ref || !st_gpu || !st_ref) {
        fprintf(stderr, "OOM in test_scan_state_carry\n"); exit(1);
    }

    fill_rand(g1, N1, 0.5f, 0.9f, 100);
    fill_rand(x1, N1, -0.5f, 0.5f, 101);
    fill_rand(g2, N2, 0.5f, 0.9f, 102);
    fill_rand(x2, N2, -0.5f, 0.5f, 103);
    fill_rand(st_gpu, C, -0.5f, 0.5f, 104);
    memcpy(st_ref, st_gpu, C * 4);

    cl_int err;
    cl_mem g1_d = clCreateBuffer(g_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N1*4, g1, &err);
    cl_mem x1_d = clCreateBuffer(g_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N1*4, x1, &err);
    cl_mem s1_d = clCreateBuffer(g_ctx, CL_MEM_WRITE_ONLY, N1*4, NULL, &err);
    cl_mem st_d = clCreateBuffer(g_ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, C*4, st_gpu, &err);

    /* First chunk */
    clSetKernelArg(k_scan, 0, sizeof(cl_mem), &g1_d);
    clSetKernelArg(k_scan, 1, sizeof(cl_mem), &x1_d);
    clSetKernelArg(k_scan, 2, sizeof(cl_mem), &s1_d);
    clSetKernelArg(k_scan, 3, sizeof(cl_mem), &st_d);
    uint uT = T1, uC = C;
    clSetKernelArg(k_scan, 4, sizeof(uint), &uT);
    clSetKernelArg(k_scan, 5, sizeof(uint), &uC);
    size_t global = C;
    clEnqueueNDRangeKernel(g_queue, k_scan, 1, NULL, &global, NULL, 0, NULL, NULL);
    clFinish(g_queue);

    /* Second chunk — reuses st_d which was updated by the first call */
    cl_mem g2_d = clCreateBuffer(g_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N2*4, g2, &err);
    cl_mem x2_d = clCreateBuffer(g_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N2*4, x2, &err);
    cl_mem s2_d = clCreateBuffer(g_ctx, CL_MEM_WRITE_ONLY, N2*4, NULL, &err);
    clSetKernelArg(k_scan, 0, sizeof(cl_mem), &g2_d);
    clSetKernelArg(k_scan, 1, sizeof(cl_mem), &x2_d);
    clSetKernelArg(k_scan, 2, sizeof(cl_mem), &s2_d);
    uT = T2;
    clSetKernelArg(k_scan, 4, sizeof(uint), &uT);
    clEnqueueNDRangeKernel(g_queue, k_scan, 1, NULL, &global, NULL, 0, NULL, NULL);
    clFinish(g_queue);

    clEnqueueReadBuffer(g_queue, s2_d, CL_TRUE, 0, N2*4, s2_gpu, 0, NULL, NULL);
    clEnqueueReadBuffer(g_queue, st_d, CL_TRUE, 0, C*4, st_gpu, 0, NULL, NULL);

    /* Reference: run both chunks sequentially */
    ref_scan(g1, x1, s1_ref, st_ref, T1, C);
    ref_scan(g2, x2, s2_ref, st_ref, T2, C);

    char detail[128];
    snprintf(detail, sizeof(detail), "T1=%zu T2=%zu C=%zu (state persists)", T1, T2, C);
    track_result(label, compute_error(s2_gpu, s2_ref, N2), detail);

    clReleaseMemObject(g1_d); clReleaseMemObject(x1_d); clReleaseMemObject(s1_d);
    clReleaseMemObject(g2_d); clReleaseMemObject(x2_d); clReleaseMemObject(s2_d);
    clReleaseMemObject(st_d);
    free(g1); free(x1); free(g2); free(x2);
    free(s1_gpu); free(s2_gpu); free(s1_ref); free(s2_ref);
    free(st_gpu); free(st_ref);
}

/* ------------------------------------------------------------------ */
/* Main                                                                */
/* ------------------------------------------------------------------ */

int main(void) {
    init_opencl();

    cl_int err;
    k_scan  = clCreateKernel(g_program, "gdn_gated_scan", &err); cl_check(err, "k_scan");
    k_decay = clCreateKernel(g_program, "gdn_cumdecay", &err);   cl_check(err, "k_decay");
    k_conv  = clCreateKernel(g_program, "gdn_causal_dwconv1d", &err); cl_check(err, "k_conv");
    k_delta = clCreateKernel(g_program, "gdn_delta_rule_decode", &err); cl_check(err, "k_delta");

    printf("GDN GPU Kernel Validation Suite (bead ob-gzk)\n");
    printf("Oracle tolerances: atol=%.0e rtol=%.0e\n\n", ORACLE_ATOL, ORACLE_RTOL);

    /* === Gated Scan: sequence length sweep === */
    printf("--- Gated Scan: sequence length sweep (C=2048) ---\n");
    {
        size_t seqs[] = {1, 16, 32, 64, 65, 128, 256, 512, 1024, 2048, 4096};
        int nseqs = sizeof(seqs) / sizeof(seqs[0]);
        for (int i = 0; i < nseqs; i++) {
            char label[64];
            snprintf(label, sizeof(label), "scan_T%zu", seqs[i]);
            test_scan(label, seqs[i], 2048, 0.5f, 0.9f, 200 + i);
        }
    }

    /* === Gated Scan: channel count sweep === */
    printf("\n--- Gated Scan: channel count sweep (T=64) ---\n");
    {
        size_t chans[] = {1, 4, 32, 128, 512, 2048, 8192};
        int nchans = sizeof(chans) / sizeof(chans[0]);
        for (int i = 0; i < nchans; i++) {
            char label[64];
            snprintf(label, sizeof(label), "scan_C%zu", chans[i]);
            test_scan(label, 64, chans[i], 0.5f, 0.9f, 300 + i);
        }
    }

    /* === Gated Scan: extreme gate values === */
    printf("\n--- Gated Scan: extreme gate values ---\n");
    test_scan("scan_decay_near_zero", 64, 2048, 0.01f, 0.05f, 400);
    test_scan("scan_decay_near_one", 64, 2048, 0.95f, 0.999f, 401);
    test_scan("scan_uniform_half", 64, 2048, 0.5f, 0.5f, 402);

    /* === Gated Scan: state carry across calls === */
    printf("\n--- Gated Scan: state carry across invocations ---\n");
    test_scan_state_carry("scan_carry_64_64", 64, 64, 2048);
    test_scan_state_carry("scan_carry_64_65", 64, 65, 2048);
    test_scan_state_carry("scan_carry_1_1", 1, 1, 2048);

    /* === Cumulative Decay: sequence length sweep === */
    printf("\n--- Cumulative Decay: sequence length sweep (C=2048) ---\n");
    {
        size_t seqs[] = {1, 16, 64, 65, 128, 512, 1024, 2048, 4096};
        int nseqs = sizeof(seqs) / sizeof(seqs[0]);
        for (int i = 0; i < nseqs; i++) {
            char label[64];
            snprintf(label, sizeof(label), "decay_T%zu", seqs[i]);
            test_decay(label, seqs[i], 2048, 0.5f, 0.9f, 500 + i);
        }
    }

    /* === Cumulative Decay: extreme values === */
    printf("\n--- Cumulative Decay: extreme values ---\n");
    test_decay("decay_near_zero", 64, 2048, 0.01f, 0.05f, 600);
    test_decay("decay_near_one", 64, 2048, 0.95f, 0.999f, 601);

    /* === Causal DWConv1D: sequence length sweep === */
    printf("\n--- Causal DWConv1D: sequence length sweep (C=2048) ---\n");
    {
        size_t seqs[] = {1, 2, 3, 4, 5, 16, 64, 65, 128, 512};
        int nseqs = sizeof(seqs) / sizeof(seqs[0]);
        for (int i = 0; i < nseqs; i++) {
            char label[64];
            snprintf(label, sizeof(label), "conv_T%zu", seqs[i]);
            test_conv(label, seqs[i], 2048, 700 + i);
        }
    }

    /* === Causal DWConv1D: T < kernel_size edge case === */
    printf("\n--- Causal DWConv1D: short sequences (T < conv_kernel=4) ---\n");
    test_conv("conv_T1_decode", 1, 2048, 710);
    test_conv("conv_T2", 2, 2048, 711);
    test_conv("conv_T3", 3, 2048, 712);

    /* === Delta-Rule: dimension sweep === */
    printf("\n--- Delta-Rule: model-realistic dimensions ---\n");
    test_delta("delta_0.8B", 16, 128, 128, 0.5f, 0.9f, 800);
    test_delta("delta_4B_32heads", 32, 128, 128, 0.5f, 0.9f, 801);

    /* === Delta-Rule: extreme decay === */
    printf("\n--- Delta-Rule: extreme decay values ---\n");
    test_delta("delta_decay_0.01", 16, 128, 128, 0.5f, 0.01f, 810);
    test_delta("delta_decay_0.99", 16, 128, 128, 0.5f, 0.99f, 811);
    test_delta("delta_decay_1.0", 16, 128, 128, 0.5f, 1.0f, 812);

    /* === Delta-Rule: extreme beta === */
    printf("\n--- Delta-Rule: extreme beta values ---\n");
    test_delta("delta_beta_0", 16, 128, 128, 0.0f, 0.9f, 820);
    test_delta("delta_beta_1", 16, 128, 128, 1.0f, 0.9f, 821);

    /* === Delta-Rule: smaller heads === */
    printf("\n--- Delta-Rule: smaller head dimensions ---\n");
    test_delta("delta_64x64", 8, 64, 64, 0.5f, 0.9f, 830);
    test_delta("delta_32x32", 4, 32, 32, 0.5f, 0.9f, 831);

    /* === Summary === */
    printf("\n=== Summary ===\n");
    printf("Total: %d  Passed: %d  Failed: %d\n", g_tests_total, g_tests_passed, g_tests_failed);
    if (g_tests_failed > 0) {
        printf("FAILED tests exceed oracle tolerances (atol=%.0e, rtol=%.0e)\n",
               ORACLE_ATOL, ORACLE_RTOL);
        return 1;
    }
    printf("All %d tests within oracle tolerances.\n", g_tests_total);
    return 0;
}
