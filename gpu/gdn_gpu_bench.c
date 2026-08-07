/*
 * OpenCL host harness for GDN GPU kernels (bead ob-q44).
 *
 * Loads gdn_gpu_kernels.cl, validates each kernel against a precision-matched
 * scalar CPU reference, then benchmarks with the same methodology as the CPU
 * bench (warmups discarded, N repeats, p50/p95).
 *
 * Build:
 *   gcc -O2 -o gdn_gpu_bench gdn_gpu_bench.c -lOpenCL -lm
 *
 * Run:
 *   ./gdn_gpu_bench              # validate + benchmark
 *   ./gdn_gpu_bench --csv        # CSV output for results pipeline
 *   ./gdn_gpu_bench --repeats 50 # more repeats
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
/* Timing                                                              */
/* ------------------------------------------------------------------ */

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

/* Full delta-rule reference for one token, one head:
 *   S[hkd * hvd] row-major, k[hkd], v[hvd], q[hkd]
 *   decay S, retrieve kv, correct, rank-1 update, read output
 */
static void ref_delta_rule(float *S, const float *k, const float *v,
                           const float *q, float beta, float decay,
                           float *out, size_t hkd, size_t hvd) {
    /* decay */
    for (size_t j = 0; j < hkd; j++)
        for (size_t i = 0; i < hvd; i++)
            S[j * hvd + i] *= decay;

    /* retrieve + correct + write + read */
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
/* Error reporting                                                     */
/* ------------------------------------------------------------------ */

static void report(const char *name, const float *a, const float *b, size_t N) {
    double mabs = 0, mrel = 0;
    for (size_t i = 0; i < N; i++) {
        double d = fabs((double)a[i] - b[i]);
        if (d > mabs) mabs = d;
        if (fabs((double)b[i]) > 1e-2) {
            double r = d / fabs((double)b[i]);
            if (r > mrel) mrel = r;
        }
    }
    printf("  %-30s max_abs=%.3e  max_rel=%.3e\n", name, mabs, mrel);
}

/* ------------------------------------------------------------------ */
/* OpenCL helpers                                                      */
/* ------------------------------------------------------------------ */

static void cl_check(cl_int err, const char *msg) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "OpenCL error %d: %s\n", err, msg);
        exit(1);
    }
}

static char *load_file(const char *path, size_t *len_out) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc(len + 1);
    fread(buf, 1, len, f);
    buf[len] = '\0';
    fclose(f);
    if (len_out) *len_out = (size_t)len;
    return buf;
}

/* ------------------------------------------------------------------ */
/* Global OpenCL state                                                 */
/* ------------------------------------------------------------------ */

static cl_platform_id   g_platform;
static cl_device_id     g_device;
static cl_context       g_ctx;
static cl_command_queue g_queue;
static cl_program       g_program;
static char             g_dev_name[128];
static char             g_dev_version[128];
static int              g_use_profiling = 0;  /* set in init_opencl() */

static void init_opencl(const char *kernel_path) {
    cl_int err;

    /* Platform */
    cl_uint nplat;
    err = clGetPlatformIDs(1, &g_platform, &nplat);
    cl_check(err, "clGetPlatformIDs");
    if (nplat == 0) { fprintf(stderr, "No OpenCL platforms found.\n"); exit(1); }

    /* Device — prefer GPU */
    err = clGetDeviceIDs(g_platform, CL_DEVICE_TYPE_GPU, 1, &g_device, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "No GPU device, trying ALL...\n");
        err = clGetDeviceIDs(g_platform, CL_DEVICE_TYPE_ALL, 1, &g_device, NULL);
    }
    cl_check(err, "clGetDeviceIDs");

    cl_device_info queries[] = {CL_DEVICE_NAME, CL_DEVICE_VERSION};
    char *dsts[] = {g_dev_name, g_dev_version};
    for (int i = 0; i < 2; i++) {
        size_t sz;
        clGetDeviceInfo(g_device, queries[i], sizeof(g_dev_name), dsts[i], &sz);
        dsts[i][sz] = '\0';  /* clGetDeviceInfo may not null-terminate if buffer is larger */
    }

    /* Context */
    g_ctx = clCreateContext(NULL, 1, &g_device, NULL, NULL, &err);
    cl_check(err, "clCreateContext");

    /* Queue: try profiling first; fall back to plain queue for drivers
     * like RustiCL/Panfrost that reject CL_QUEUE_PROFILING_ENABLE (err -35). */
    const cl_queue_properties qprops[] = {CL_QUEUE_PROPERTIES,
        CL_QUEUE_PROFILING_ENABLE, 0};
    g_queue = clCreateCommandQueueWithProperties(g_ctx, g_device, qprops, &err);
    if (err != CL_SUCCESS) {
        g_use_profiling = 0;
        g_queue = clCreateCommandQueueWithProperties(g_ctx, g_device, NULL, &err);
        cl_check(err, "clCreateCommandQueueWithProperties (no profiling)");
        printf("  [info] GPU profiling not supported; using wall-clock timing\n");
    } else {
        g_use_profiling = 1;
    }

    /* Build program from source */
    size_t src_len;
    char *src = load_file(kernel_path, &src_len);
    g_program = clCreateProgramWithSource(g_ctx, 1, (const char **)&src,
                                          &src_len, &err);
    cl_check(err, "clCreateProgramWithSource");

    err = clBuildProgram(g_program, 1, &g_device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_sz = 0;
        clGetProgramBuildInfo(g_program, g_device, CL_PROGRAM_BUILD_LOG,
                              0, NULL, &log_sz);
        char *log = (char *)malloc(log_sz + 1);
        clGetProgramBuildInfo(g_program, g_device, CL_PROGRAM_BUILD_LOG,
                              log_sz, log, NULL);
        log[log_sz] = '\0';
        fprintf(stderr, "Build failed:\n%s\n", log);
        free(log);
        exit(1);
    }
    free(src);

    printf("OpenCL device: %s (%s)\n\n", g_dev_name, g_dev_version);
}

/* Run a kernel and return GPU time in milliseconds via profiling events. */
static double run_kernel_timed(cl_kernel kern, cl_uint nargs,
                               cl_mem *args, size_t *global, size_t *local) {
    cl_int err;
    for (cl_uint i = 0; i < nargs; i++) {
        err = clSetKernelArg(kern, i, sizeof(cl_mem), &args[i]);
        cl_check(err, "clSetKernelArg");
    }
    cl_event ev;
    double t0, t1;
    if (g_use_profiling) {
        err = clEnqueueNDRangeKernel(g_queue, kern, (local ? 2 : 1), NULL,
                                     global, local, 0, NULL, &ev);
        cl_check(err, "clEnqueueNDRangeKernel");
        clFinish(g_queue);

        cl_ulong t_start, t_end;
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t_start),
                                &t_start, NULL);
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t_end),
                                &t_end, NULL);
        clReleaseEvent(ev);
        return (double)(t_end - t_start) * 1e-6;  /* ns → ms */
    } else {
        t0 = now_s();
        err = clEnqueueNDRangeKernel(g_queue, kern, (local ? 2 : 1), NULL,
                                     global, local, 0, NULL, &ev);
        cl_check(err, "clEnqueueNDRangeKernel");
        clFinish(g_queue);
        t1 = now_s();
        clReleaseEvent(ev);
        return (t1 - t0) * 1e3;  /* s → ms */
    }
}

/* For 1D kernels with scalar args (uint/float), we set args differently. */
static double run_kernel_1d(cl_kernel kern, cl_uint n_buf_args, cl_mem *bufs,
                            int n_scalar_args, const void **scalar_args,
                            const size_t *scalar_sizes,
                            size_t global_size) {
    cl_int err;
    cl_uint ai = 0;
    for (cl_uint i = 0; i < n_buf_args; i++) {
        err = clSetKernelArg(kern, ai++, sizeof(cl_mem), &bufs[i]);
        cl_check(err, "clSetKernelArg(buf)");
    }
    for (int i = 0; i < n_scalar_args; i++) {
        err = clSetKernelArg(kern, ai++, scalar_sizes[i], scalar_args[i]);
        cl_check(err, "clSetKernelArg(scalar)");
    }
    cl_event ev;
    double t0, t1;
    if (g_use_profiling) {
        err = clEnqueueNDRangeKernel(g_queue, kern, 1, NULL, &global_size, NULL,
                                     0, NULL, &ev);
        cl_check(err, "clEnqueueNDRangeKernel");
        clFinish(g_queue);

        cl_ulong t_start, t_end;
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t_start),
                                &t_start, NULL);
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t_end),
                                &t_end, NULL);
        clReleaseEvent(ev);
        return (double)(t_end - t_start) * 1e-6;
    } else {
        t0 = now_s();
        err = clEnqueueNDRangeKernel(g_queue, kern, 1, NULL, &global_size, NULL,
                                     0, NULL, &ev);
        cl_check(err, "clEnqueueNDRangeKernel");
        clFinish(g_queue);
        t1 = now_s();
        clReleaseEvent(ev);
        return (t1 - t0) * 1e3;  /* s → ms */
    }
}

/* ------------------------------------------------------------------ */
/* RNG — same seed as test_gdn_sve.c for reproducibility              */
/* ------------------------------------------------------------------ */

static void fill_random(float *p, size_t n, float lo, float hi, int seed) {
    srand(seed);
    for (size_t i = 0; i < n; i++)
        p[i] = lo + (hi - lo) * (rand() / (float)RAND_MAX);
}

/* ------------------------------------------------------------------ */
/* Main: validate + benchmark                                          */
/* ------------------------------------------------------------------ */

#define WARMUPS 3
#define MAX_REPEATS 256

int main(int argc, char **argv) {
    int repeats = 30;
    int csv_mode = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--csv") == 0) csv_mode = 1;
        if (strcmp(argv[i], "--repeats") == 0 && i + 1 < argc)
            repeats = atoi(argv[++i]);
    }
    if (repeats > MAX_REPEATS) repeats = MAX_REPEATS;

    init_opencl("gpu/gdn_gpu_kernels.cl");

    /* ================================================================
     * Kernels
     * ================================================================ */
    cl_int err;
    cl_kernel k_scan  = clCreateKernel(g_program, "gdn_gated_scan", &err); cl_check(err, "create scan");
    cl_kernel k_decay = clCreateKernel(g_program, "gdn_cumdecay", &err);   cl_check(err, "create decay");
    cl_kernel k_conv  = clCreateKernel(g_program, "gdn_causal_dwconv1d", &err); cl_check(err, "create conv");
    cl_kernel k_delta = clCreateKernel(g_program, "gdn_delta_rule_decode", &err); cl_check(err, "create delta");

    /* CSV header — standard (no # prefix so csv.DictReader and validate_results.py parse it) */
    if (csv_mode)
        printf("kernel,dim1,dim2,dim3,p50_ms,p95_ms,bw_mibs\n");

    /* ================================================================
     * Test 1: Gated scan — correctness
     * ================================================================ */
    printf("=== Validation ===\n");
    {
        size_t T = 64, C = 2048, N = T * C;
        float *g_h = malloc(N * 4), *x_h = malloc(N * 4);
        float *s_gpu = malloc(N * 4), *s_ref = malloc(N * 4);
        float *st_gpu = malloc(C * 4), *st_ref = malloc(C * 4);

        fill_random(g_h, N, 0.5f, 0.9f, 7);
        fill_random(x_h, N, -0.5f, 0.5f, 8);
        fill_random(st_gpu, C, -0.5f, 0.5f, 9);
        memcpy(st_ref, st_gpu, C * 4);

        cl_mem g_d = clCreateBuffer(g_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    N * 4, g_h, &err);
        cl_mem x_d = clCreateBuffer(g_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    N * 4, x_h, &err);
        cl_mem s_d = clCreateBuffer(g_ctx, CL_MEM_WRITE_ONLY, N * 4, NULL, &err);
        cl_mem st_d = clCreateBuffer(g_ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                     C * 4, st_gpu, &err);

        uint u32_T = (uint)T, u32_C = (uint)C;
        cl_mem bufs[] = {g_d, x_d, s_d, st_d};
        const void *scalars[] = {&u32_T, &u32_C};
        const size_t sz[] = {sizeof(uint), sizeof(uint)};

        /* one run for correctness */
        run_kernel_1d(k_scan, 4, bufs, 2, scalars, sz, C);
        clEnqueueReadBuffer(g_queue, s_d, CL_TRUE, 0, N * 4, s_gpu, 0, NULL, NULL);
        clEnqueueReadBuffer(g_queue, st_d, CL_TRUE, 0, C * 4, st_gpu, 0, NULL, NULL);

        ref_scan(g_h, x_h, s_ref, st_ref, T, C);
        report("gated_scan", s_gpu, s_ref, N);
        report("gated_scan state", st_gpu, st_ref, C);

        /* benchmark */
        double times[MAX_REPEATS];
        for (int r = 0; r < WARMUPS + repeats; r++) {
            /* reset state each time for correctness of state carry */
            clEnqueueWriteBuffer(g_queue, st_d, CL_TRUE, 0, C * 4, st_ref, 0, NULL, NULL);
            double ms = run_kernel_1d(k_scan, 4, bufs, 2, scalars, sz, C);
            if (r >= WARMUPS) times[r - WARMUPS] = ms;
        }
        qsort(times, repeats, sizeof(double), cmp_double);
        double p50 = pct(times, repeats, 0.50);
        double p95 = pct(times, repeats, 0.95);
        double bytes = 4.0 * (3.0 * N + 2.0 * C);  /* read g, x + write s + read/write state */
        if (!csv_mode)
            printf("  bench: p50=%.4f ms  p95=%.4f ms  bw=%.1f MiB/s\n\n",
                   p50, p95, bytes / (p50 * 1e-3) / (1024 * 1024));
        else
            printf("gdn_gated_scan,%zu,%zu,,%.4f,%.4f,%.1f\n", T, C, p50, p95,
                   bytes / (p50 * 1e-3) / (1024 * 1024));

        clReleaseMemObject(g_d); clReleaseMemObject(x_d);
        clReleaseMemObject(s_d); clReleaseMemObject(st_d);
        free(g_h); free(x_h); free(s_gpu); free(s_ref); free(st_gpu); free(st_ref);
    }

    /* ================================================================
     * Test 2: Cumulative decay — correctness
     * ================================================================ */
    {
        size_t T = 64, C = 2048, N = T * C;
        float *a_h = malloc(N * 4);
        float *d_gpu = malloc(N * 4), *d_ref = malloc(N * 4);

        fill_random(a_h, N, 0.5f, 0.9f, 11);

        cl_mem a_d = clCreateBuffer(g_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    N * 4, a_h, &err);
        cl_mem d_d = clCreateBuffer(g_ctx, CL_MEM_WRITE_ONLY, N * 4, NULL, &err);

        uint u32_T = (uint)T, u32_C = (uint)C;
        cl_mem bufs[] = {a_d, d_d};
        const void *scalars[] = {&u32_T, &u32_C};
        const size_t sz[] = {sizeof(uint), sizeof(uint)};

        run_kernel_1d(k_decay, 2, bufs, 2, scalars, sz, C);
        clEnqueueReadBuffer(g_queue, d_d, CL_TRUE, 0, N * 4, d_gpu, 0, NULL, NULL);

        ref_decay(a_h, d_ref, T, C);
        report("cumdecay", d_gpu, d_ref, N);

        double times[MAX_REPEATS];
        for (int r = 0; r < WARMUPS + repeats; r++) {
            double ms = run_kernel_1d(k_decay, 2, bufs, 2, scalars, sz, C);
            if (r >= WARMUPS) times[r - WARMUPS] = ms;
        }
        qsort(times, repeats, sizeof(double), cmp_double);
        double p50 = pct(times, repeats, 0.50);
        double bytes = 4.0 * 2.0 * N;
        if (!csv_mode)
            printf("  bench: p50=%.4f ms  bw=%.1f MiB/s\n\n", p50,
                   bytes / (p50 * 1e-3) / (1024 * 1024));
        else
            printf("gdn_cumdecay,%zu,%zu,,%.4f,,%.1f\n", T, C, p50,
                   bytes / (p50 * 1e-3) / (1024 * 1024));

        clReleaseMemObject(a_d); clReleaseMemObject(d_d);
        free(a_h); free(d_gpu); free(d_ref);
    }

    /* ================================================================
     * Test 3: Causal DWConv1D — correctness
     * ================================================================ */
    {
        size_t T = 64, C = 2048, N = T * C;
        float *in_h = malloc(N * 4), *w_h = malloc(4 * C * 4);
        float *o_gpu = malloc(N * 4), *o_ref = malloc(N * 4);
        float *h_gpu = malloc(3 * C * 4), *h_ref = malloc(3 * C * 4);

        fill_random(in_h, N, -0.5f, 0.5f, 13);
        fill_random(w_h, 4 * C, -0.5f, 0.5f, 14);
        fill_random(h_gpu, 3 * C, -0.5f, 0.5f, 15);
        memcpy(h_ref, h_gpu, 3 * C * 4);

        cl_mem in_d = clCreateBuffer(g_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     N * 4, in_h, &err);
        cl_mem w_d = clCreateBuffer(g_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    4 * C * 4, w_h, &err);
        cl_mem o_d = clCreateBuffer(g_ctx, CL_MEM_WRITE_ONLY, N * 4, NULL, &err);
        cl_mem h_d = clCreateBuffer(g_ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                    3 * C * 4, h_gpu, &err);

        uint u32_T = (uint)T, u32_C = (uint)C;
        cl_mem bufs[] = {in_d, w_d, o_d, h_d};
        const void *scalars[] = {&u32_T, &u32_C};
        const size_t sz[] = {sizeof(uint), sizeof(uint)};

        run_kernel_1d(k_conv, 4, bufs, 2, scalars, sz, C);
        clEnqueueReadBuffer(g_queue, o_d, CL_TRUE, 0, N * 4, o_gpu, 0, NULL, NULL);
        clEnqueueReadBuffer(g_queue, h_d, CL_TRUE, 0, 3 * C * 4, h_gpu, 0, NULL, NULL);

        ref_conv(in_h, w_h, o_ref, h_ref, T, C);
        report("causal_dwconv1d", o_gpu, o_ref, N);
        report("conv history", h_gpu, h_ref, 3 * C);

        double times[MAX_REPEATS];
        for (int r = 0; r < WARMUPS + repeats; r++) {
            clEnqueueWriteBuffer(g_queue, h_d, CL_TRUE, 0, 3 * C * 4, h_ref, 0, NULL, NULL);
            double ms = run_kernel_1d(k_conv, 4, bufs, 2, scalars, sz, C);
            if (r >= WARMUPS) times[r - WARMUPS] = ms;
        }
        qsort(times, repeats, sizeof(double), cmp_double);
        double p50 = pct(times, repeats, 0.50);
        double bytes = 4.0 * (N + 4.0 * C + N + 2.0 * 3.0 * C);
        if (!csv_mode)
            printf("  bench: p50=%.4f ms  bw=%.1f MiB/s\n\n", p50,
                   bytes / (p50 * 1e-3) / (1024 * 1024));
        else
            printf("gdn_causal_dwconv1d,%zu,%zu,,%.4f,,%.1f\n", T, C, p50,
                   bytes / (p50 * 1e-3) / (1024 * 1024));

        clReleaseMemObject(in_d); clReleaseMemObject(w_d);
        clReleaseMemObject(o_d); clReleaseMemObject(h_d);
        free(in_h); free(w_h); free(o_gpu); free(o_ref); free(h_gpu); free(h_ref);
    }

    /* ================================================================
     * Test 4: Full delta-rule decode — correctness + benchmark
     *
     * Qwen3.5-0.8B dimensions: 16 heads, head_k_dim=128, head_v_dim=128
     * We test with one token: verify S update and output vector.
     * ================================================================ */
    {
        size_t num_heads = 16, hkd = 128, hvd = 128;
        size_t S_sz = num_heads * hkd * hvd;
        size_t kv_sz = num_heads * hkd;
        size_t vv_sz = num_heads * hvd;

        float *S_gpu = malloc(S_sz * 4), *S_ref = malloc(S_sz * 4);
        float *k_h = malloc(kv_sz * 4), *v_h = malloc(vv_sz * 4), *q_h = malloc(kv_sz * 4);
        float *out_gpu = malloc(vv_sz * 4), *out_ref = malloc(vv_sz * 4);

        /* Initialize state to small random values; k, q normalised to unit length per head */
        fill_random(S_gpu, S_sz, -0.01f, 0.01f, 21);
        memcpy(S_ref, S_gpu, S_sz * 4);
        fill_random(k_h, kv_sz, -1.0f, 1.0f, 22);
        fill_random(v_h, vv_sz, -0.5f, 0.5f, 23);
        fill_random(q_h, kv_sz, -1.0f, 1.0f, 24);

        /* Normalise k and q per head (L2 norm, as GDN does) */
        for (size_t h = 0; h < num_heads; h++) {
            float kn = 0, qn = 0;
            for (size_t j = 0; j < hkd; j++) {
                kn += k_h[h * hkd + j] * k_h[h * hkd + j];
                qn += q_h[h * hkd + j] * q_h[h * hkd + j];
            }
            kn = sqrtf(kn); qn = sqrtf(qn);
            for (size_t j = 0; j < hkd; j++) {
                k_h[h * hkd + j] /= kn;
                q_h[h * hkd + j] /= qn;
            }
        }

        float beta = 0.5f, decay = 0.9f;

        /* GPU buffers */
        cl_mem S_d = clCreateBuffer(g_ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                    S_sz * 4, S_gpu, &err);
        cl_mem k_d = clCreateBuffer(g_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    kv_sz * 4, k_h, &err);
        cl_mem v_d = clCreateBuffer(g_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    vv_sz * 4, v_h, &err);
        cl_mem q_d = clCreateBuffer(g_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    kv_sz * 4, q_h, &err);
        cl_mem out_d = clCreateBuffer(g_ctx, CL_MEM_WRITE_ONLY, vv_sz * 4, NULL, &err);

        /* Set kernel args for delta-rule (2D launch: global = (hvd, num_heads)) */
        clSetKernelArg(k_delta, 0, sizeof(cl_mem), &S_d);
        clSetKernelArg(k_delta, 1, sizeof(cl_mem), &k_d);
        clSetKernelArg(k_delta, 2, sizeof(cl_mem), &v_d);
        clSetKernelArg(k_delta, 3, sizeof(cl_mem), &q_d);
        clSetKernelArg(k_delta, 4, sizeof(float), &beta);
        clSetKernelArg(k_delta, 5, sizeof(float), &decay);
        clSetKernelArg(k_delta, 6, sizeof(cl_mem), &out_d);
        uint u_hkd = (uint)hkd, u_hvd = (uint)hvd, u_nh = (uint)num_heads;
        clSetKernelArg(k_delta, 7, sizeof(uint), &u_hkd);
        clSetKernelArg(k_delta, 8, sizeof(uint), &u_hvd);
        clSetKernelArg(k_delta, 9, sizeof(uint), &u_nh);

        size_t global2d[] = {hvd, num_heads};
        size_t local2d[] = {hvd, 1};  /* one work-group per head */

        cl_event ev;
        clEnqueueNDRangeKernel(g_queue, k_delta, 2, NULL, global2d, local2d, 0, NULL, &ev);
        clFinish(g_queue);
        clEnqueueReadBuffer(g_queue, S_d, CL_TRUE, 0, S_sz * 4, S_gpu, 0, NULL, NULL);
        clEnqueueReadBuffer(g_queue, out_d, CL_TRUE, 0, vv_sz * 4, out_gpu, 0, NULL, NULL);

        /* CPU reference: run delta-rule for each head */
        for (size_t h = 0; h < num_heads; h++) {
            ref_delta_rule(S_ref + h * hkd * hvd,
                          k_h + h * hkd, v_h + h * hvd, q_h + h * hkd,
                          beta, decay, out_ref + h * hvd, hkd, hvd);
        }

        report("delta_rule output", out_gpu, out_ref, vv_sz);
        report("delta_rule state", S_gpu, S_ref, S_sz);

        /* Benchmark: single-token decode */
        double times[MAX_REPEATS];
        for (int r = 0; r < WARMUPS + repeats; r++) {
            clSetKernelArg(k_delta, 0, sizeof(cl_mem), &S_d);
            clSetKernelArg(k_delta, 1, sizeof(cl_mem), &k_d);
            clSetKernelArg(k_delta, 2, sizeof(cl_mem), &v_d);
            clSetKernelArg(k_delta, 3, sizeof(cl_mem), &q_d);
            clSetKernelArg(k_delta, 6, sizeof(cl_mem), &out_d);

            clEnqueueNDRangeKernel(g_queue, k_delta, 2, NULL, global2d, local2d, 0, NULL, &ev);

            if (g_use_profiling) {
                clFinish(g_queue);
                cl_ulong t_s, t_e;
                clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t_s), &t_s, NULL);
                clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t_e), &t_e, NULL);
                clReleaseEvent(ev);
                if (r >= WARMUPS) times[r - WARMUPS] = (double)(t_e - t_s) * 1e-6;
            } else {
                double t0 = now_s();
                clFinish(g_queue);
                double t1 = now_s();
                clReleaseEvent(ev);
                if (r >= WARMUPS) times[r - WARMUPS] = (t1 - t0) * 1e3;  /* s → ms */
            }
        }
        qsort(times, repeats, sizeof(double), cmp_double);
        double p50 = pct(times, repeats, 0.50);
        double p95 = pct(times, repeats, 0.95);

        /* For delta-rule: 3 full matrix passes over S (decay, retrieve, write)
         * plus the read pass. S is hkd*hvd per head. */
        double bytes = 4.0 * (4.0 * S_sz + kv_sz + vv_sz + kv_sz + vv_sz);
        if (!csv_mode) {
            printf("  delta_rule p50=%.4f ms  p95=%.4f ms  bw=%.1f MiB/s\n",
                   p50, p95, bytes / (p50 * 1e-3) / (1024 * 1024));
            printf("  per-head: %.4f ms  state_size=%zu floats (%.1f KiB)\n\n",
                   p50 / num_heads, hkd * hvd, hkd * hvd * 4.0 / 1024.0);
        } else {
            printf("gdn_delta_rule_decode,%zu,%zu,%zu,%.4f,%.4f,%.1f\n",
                   num_heads, hkd, hvd, p50, p95,
                   bytes / (p50 * 1e-3) / (1024 * 1024));
        }

        clReleaseMemObject(S_d); clReleaseMemObject(k_d);
        clReleaseMemObject(v_d); clReleaseMemObject(q_d); clReleaseMemObject(out_d);
        free(S_gpu); free(S_ref); free(k_h); free(v_h); free(q_h);
        free(out_gpu); free(out_ref);
    }

    /* Cleanup */
    clReleaseKernel(k_scan);
    clReleaseKernel(k_decay);
    clReleaseKernel(k_conv);
    clReleaseKernel(k_delta);
    clReleaseProgram(g_program);
    clReleaseCommandQueue(g_queue);
    clReleaseContext(g_ctx);

    return 0;
}
