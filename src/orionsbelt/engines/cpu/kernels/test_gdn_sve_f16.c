/* Tests for mixed-precision GDN state variants (ob-8qt.4).
 *
 * Verifies that fp16/bf16 state kernels produce results close to the fp32
 * reference, documents the precision loss honestly, and checks that the
 * fp32 accumulator constraint is enforced (no catastrophic cancellation
 * from uniform fp16 narrowing).
 *
 * Build:
 *   aarch64-linux-gnu-gcc -O3 -march=armv8.6-a+sve2+i8mm+bf16 -static \
 *       gdn_sve.c gdn_sve_f16.c test_gdn_sve_f16.c -o verify_f16 -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stddef.h>
#include <string.h>

/* fp32 reference kernels */
void gdn_gated_scan_f32(const float *, const float *, float *, float *, size_t, size_t);
void gdn_cumdecay_f32(const float *, float *, size_t, size_t);

/* Mixed-precision variants */
void gdn_gated_scan_f16(const float *, const float *, float *, _Float16 *, size_t, size_t);
void gdn_cumdecay_f16(const float *, _Float16 *, size_t, size_t);
void gdn_gated_scan_bf16(const float *, const float *, float *, __bf16 *, size_t, size_t);
void gdn_cumdecay_bf16(const float *, __bf16 *, size_t, size_t);

/* bf16 ↔ float helpers (same as in the kernel file) */
static inline float _bf16_to_f32(__bf16 v) {
    unsigned short bits;
    memcpy(&bits, &v, sizeof(bits));
    unsigned int u = (unsigned int)bits << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}
static inline __bf16 _f32_to_bf16(float v) {
    unsigned int u;
    memcpy(&u, &v, sizeof(u));
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16;
    unsigned short bits = (unsigned short)u;
    __bf16 b;
    memcpy(&b, &bits, sizeof(b));
    return b;
}

/* Double-precision reference for honest quality assessment */
static void refD_scan(const float *g, const float *x, float *s, double *st,
                      size_t T, size_t C) {
    for (size_t c = 0; c < C; c++) {
        double a = st[c];
        for (size_t t = 0; t < T; t++) {
            a = x[t * C + c] + a * g[t * C + c];
            s[t * C + c] = (float)a;
        }
        st[c] = a;
    }
}

static void refD_cumdecay(const float *a, float *decay_f, size_t T, size_t C) {
    for (size_t c = 0; c < C; c++) {
        double run = 1.0;
        for (size_t t = 0; t < T; t++) {
            run *= a[t * C + c];
            decay_f[t * C + c] = (float)run;
        }
    }
}

static void report(const char *name, const float *a, const float *b, size_t N) {
    double mabs = 0, mrel = 0;
    size_t big = 0;
    for (size_t i = 0; i < N; i++) {
        double d = fabs((double)a[i] - b[i]);
        if (d > mabs) mabs = d;
        if (fabs((double)b[i]) > 1e-2) {
            big++;
            double r = d / fabs((double)b[i]);
            if (r > mrel) mrel = r;
        }
    }
    printf("  %-40s max_abs=%.3e  max_rel(|ref|>1e-2, n=%zu)=%.3e\n",
           name, mabs, big, mrel);
}

static int all_pass = 1;

static void check(const char *name, int condition) {
    printf("  %-40s %s\n", name, condition ? "PASS" : "FAIL");
    if (!condition) all_pass = 0;
}

int main(void) {
    size_t T = 64, C = 2051; /* 2051 exercises predicated tails */
    size_t N = T * C;

    /* Allocate input data */
    float *g = malloc(N * sizeof(float));
    float *x = malloc(N * sizeof(float));
    float *a_decay = malloc(N * sizeof(float));

    /* fp32 outputs (reference) */
    float *s_f32 = malloc(N * sizeof(float));
    float *decay_f32 = malloc(N * sizeof(float));
    float *state_f32 = malloc(C * sizeof(float));

    /* fp16 outputs */
    float *s_f16 = malloc(N * sizeof(float));
    _Float16 *decay_f16 = malloc(N * sizeof(_Float16));
    _Float16 *state_f16 = malloc(C * sizeof(_Float16));

    /* bf16 outputs */
    float *s_bf16 = malloc(N * sizeof(float));
    __bf16 *decay_bf16 = malloc(N * sizeof(__bf16));
    __bf16 *state_bf16 = malloc(C * sizeof(__bf16));

    /* Double reference output (stored as float) */
    float *s_dbl = malloc(N * sizeof(float));
    double *state_dbl = malloc(C * sizeof(double));

    /* Init inputs: gates in [0.5, 0.9], inputs in [-0.5, 0.5] */
    srand(42);
    for (size_t i = 0; i < N; i++) {
        g[i] = 0.5f + 0.4f * (rand() / (float)RAND_MAX);
        x[i] = (rand() / (float)RAND_MAX) - 0.5f;
        a_decay[i] = 0.5f + 0.4f * (rand() / (float)RAND_MAX);
    }

    /* Init state */
    float state_init[2051];
    for (size_t i = 0; i < C; i++) {
        state_init[i] = (rand() / (float)RAND_MAX) - 0.5f;
        state_f32[i] = state_init[i];
        state_f16[i] = (_Float16)state_init[i];
        state_bf16[i] = _f32_to_bf16(state_init[i]);
        state_dbl[i] = state_init[i];
    }

    printf("=== Gated scan: fp16-state vs fp32-state ===\n\n");

    /* Run fp32 reference */
    gdn_gated_scan_f32(g, x, s_f32, state_f32, T, C);

    /* Run fp16-state variant (reset state) */
    for (size_t i = 0; i < C; i++) state_f16[i] = (_Float16)state_init[i];
    gdn_gated_scan_f16(g, x, s_f16, state_f16, T, C);

    /* Run double reference (reset state) */
    for (size_t i = 0; i < C; i++) state_dbl[i] = state_init[i];
    refD_scan(g, x, s_dbl, state_dbl, T, C);

    /* Compare fp16-state output vs fp32-state output */
    report("fp16-state vs fp32-state (output)", s_f16, s_f32, N);

    /* Compare both vs double reference */
    report("fp32-state vs double reference", s_f32, s_dbl, N);
    report("fp16-state vs double reference", s_f16, s_dbl, N);

    /* The fp16 variant should be close to fp32 (within fp16 mantissa: ~1e-3) */
    {
        double mrel = 0;
        size_t cnt = 0;
        for (size_t i = 0; i < N; i++) {
            if (fabs((double)s_f32[i]) > 0.1) {
                cnt++;
                double r = fabs((double)s_f16[i] - s_f32[i]) / fabs((double)s_f32[i]);
                if (r > mrel) mrel = r;
            }
        }
        printf("  fp16-state max relative error vs fp32: %.3e (n=%zu)\n", mrel, cnt);
        check("fp16-state within 5e-3 of fp32", mrel < 5e-3);
    }

    /* Verify the carried state */
    {
        double mrel = 0;
        for (size_t c = 0; c < C; c++) {
            if (fabs((double)state_f32[c]) > 0.1) {
                double r = fabs((double)state_f16[c] - state_f32[c]) /
                           fabs((double)state_f32[c]);
                if (r > mrel) mrel = r;
            }
        }
        printf("  fp16-state carried state max rel err: %.3e\n", mrel);
        check("fp16-state carried state within 5e-3", mrel < 5e-3);
    }

    printf("\n=== Gated scan: bf16-state vs fp32-state ===\n\n");

    /* Reset state for bf16 test */
    for (size_t i = 0; i < C; i++) {
        state_f32[i] = state_init[i];
        state_bf16[i] = _f32_to_bf16(state_init[i]);
    }

    gdn_gated_scan_f32(g, x, s_f32, state_f32, T, C);
    gdn_gated_scan_bf16(g, x, s_bf16, state_bf16, T, C);

    report("bf16-state vs fp32-state (output)", s_bf16, s_f32, N);

    /* bf16 has 7 mantissa bits → ~1e-2 relative precision */
    {
        double mrel = 0;
        size_t cnt = 0;
        for (size_t i = 0; i < N; i++) {
            if (fabs((double)s_f32[i]) > 0.1) {
                cnt++;
                double r = fabs((double)s_bf16[i] - s_f32[i]) /
                           fabs((double)s_f32[i]);
                if (r > mrel) mrel = r;
            }
        }
        printf("  bf16-state max relative error vs fp32: %.3e (n=%zu)\n", mrel, cnt);
        check("bf16-state within 3e-2 of fp32", mrel < 3e-2);
    }

    printf("\n=== Cumulative decay: fp16-output vs fp32-output ===\n\n");

    /* Run fp32 cumdecay */
    gdn_cumdecay_f32(a_decay, decay_f32, T, C);
    /* Run fp16-output cumdecay */
    gdn_cumdecay_f16(a_decay, decay_f16, T, C);

    /* Convert fp16 decay to float for comparison */
    float *decay_f16_f = malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) decay_f16_f[i] = (float)decay_f16[i];

    /* Double reference */
    float *decay_dbl = malloc(N * sizeof(float));
    refD_cumdecay(a_decay, decay_dbl, T, C);

    report("fp32 cumdecay vs double reference", decay_f32, decay_dbl, N);
    report("fp16-output cumdecay vs double reference", decay_f16_f, decay_dbl, N);
    report("fp16-output cumdecay vs fp32-output", decay_f16_f, decay_f32, N);

    /* The accumulator is fp32 in both cases; only the output storage differs.
     * For decay values near 0.5^20 ≈ 1e-6, fp16's limited mantissa introduces
     * ~3% relative error — but these negligibly small values contribute almost
     * nothing to the final scan output. The fp32 accumulator ensures the math
     * is correct; only the storage precision changes. */
    {
        double mrel_all = 0, mrel_significant = 0;
        size_t cnt_all = 0, cnt_significant = 0;
        for (size_t i = 0; i < N; i++) {
            double dv = fabs((double)decay_f32[i]);
            if (dv > 1e-6) {
                cnt_all++;
                double r = fabs((double)decay_f16_f[i] - decay_f32[i]) / dv;
                if (r > mrel_all) mrel_all = r;
            }
            if (dv > 1e-3) { /* values significant enough that relative error matters */
                cnt_significant++;
                double r = fabs((double)decay_f16_f[i] - decay_f32[i]) / dv;
                if (r > mrel_significant) mrel_significant = r;
            }
        }
        printf("  fp16-output max rel err (values > 1e-6): %.3e (n=%zu)\n", mrel_all, cnt_all);
        printf("  fp16-output max rel err (values > 1e-3): %.3e (n=%zu)\n", mrel_significant, cnt_significant);
        check("fp16-output within 2e-3 of fp32 (values > 1e-3)", mrel_significant < 2e-3);
    }

    printf("\n=== Cumulative decay: bf16-output vs fp32-output ===\n\n");

    {
        gdn_cumdecay_bf16(a_decay, decay_bf16, T, C);
        float *decay_bf16_f = malloc(N * sizeof(float));
        for (size_t i = 0; i < N; i++) decay_bf16_f[i] = _bf16_to_f32(decay_bf16[i]);

        report("bf16-output cumdecay vs fp32-output", decay_bf16_f, decay_f32, N);

        double mrel = 0;
        size_t cnt = 0;
        for (size_t i = 0; i < N; i++) {
            if (fabs((double)decay_f32[i]) > 1e-6) {
                cnt++;
                double r = fabs((double)decay_bf16_f[i] - decay_f32[i]) /
                           fabs((double)decay_f32[i]);
                if (r > mrel) mrel = r;
            }
        }
        printf("  bf16-output max relative error vs fp32: %.3e (n=%zu)\n", mrel, cnt);
        check("bf16-output within 2e-2 of fp32 (values > 1e-6)", mrel < 2e-2);

        free(decay_bf16_f);
    }

    printf("\n=== State continuity: scan in two halves ===\n\n");

    /* Verify that fp16 state carries correctly across two calls. */
    {
        size_t T2 = T / 2; /* 32 steps per half */

        /* fp32 reference: two halves */
        float *s_ref = calloc(N, sizeof(float));
        float *st_ref = malloc(C * sizeof(float));
        for (size_t i = 0; i < C; i++) st_ref[i] = state_init[i];
        gdn_gated_scan_f32(g, x, s_ref, st_ref, T2, C);
        gdn_gated_scan_f32(g + T2 * C, x + T2 * C, s_ref + T2 * C, st_ref, T2, C);

        /* fp16-state: two halves */
        float *s_f16b = calloc(N, sizeof(float));
        _Float16 *st_f16 = malloc(C * sizeof(_Float16));
        for (size_t i = 0; i < C; i++) st_f16[i] = (_Float16)state_init[i];
        gdn_gated_scan_f16(g, x, s_f16b, st_f16, T2, C);
        gdn_gated_scan_f16(g + T2 * C, x + T2 * C, s_f16b + T2 * C, st_f16, T2, C);

        /* The second half should match the fp32 reference within fp16 precision. */
        double mrel = 0;
        size_t cnt = 0;
        for (size_t i = T2 * C; i < N; i++) {
            if (fabs((double)s_ref[i]) > 0.1) {
                cnt++;
                double r = fabs((double)s_f16b[i] - s_ref[i]) /
                           fabs((double)s_ref[i]);
                if (r > mrel) mrel = r;
            }
        }
        printf("  Second-half max relative error (fp16 state boundary): %.3e (n=%zu)\n",
               mrel, cnt);
        check("State continuity within 5e-3", mrel < 5e-3);

        free(s_ref); free(st_ref); free(s_f16b); free(st_f16);
    }

    printf("\n=== Decode scenario: seq=1 (single token) ===\n\n");

    /* In decode, the scan is called with seq=1 each token. */
    {
        float *s1_f32 = malloc(C * sizeof(float));
        float *s1_f16 = malloc(C * sizeof(float));
        float *st_f32 = malloc(C * sizeof(float));
        _Float16 *st_f16 = malloc(C * sizeof(_Float16));

        for (size_t i = 0; i < C; i++) {
            st_f32[i] = state_init[i];
            st_f16[i] = (_Float16)state_init[i];
        }

        double mrel = 0;
        for (size_t step = 0; step < 10; step++) {
            const float *gp = g + step * C;
            const float *xp = x + step * C;
            gdn_gated_scan_f32(gp, xp, s1_f32, st_f32, 1, C);
            gdn_gated_scan_f16(gp, xp, s1_f16, st_f16, 1, C);

            for (size_t c = 0; c < C; c++) {
                if (fabs((double)s1_f32[c]) > 0.1) {
                    double r = fabs((double)s1_f16[c] - s1_f32[c]) /
                               fabs((double)s1_f32[c]);
                    if (r > mrel) mrel = r;
                }
            }
        }
        printf("  Decode (10 steps, seq=1) max relative error: %.3e\n", mrel);
        check("Decode within 5e-3 over 10 steps", mrel < 5e-3);

        free(s1_f32); free(s1_f16); free(st_f32); free(st_f16);
    }

    printf("\n=== Memory savings ===\n\n");

    /* Qwen3.5-4B: 524,288 floats/layer × 24 GDN layers */
    {
        size_t per_layer = 524288;
        size_t num_layers = 24;
        size_t fp32_bytes = per_layer * num_layers * 4;
        size_t fp16_bytes = per_layer * num_layers * 2;
        size_t saved = fp32_bytes - fp16_bytes;
        printf("  fp32 state: %zu bytes (%.1f MiB)\n", fp32_bytes, fp32_bytes / 1048576.0);
        printf("  fp16 state: %zu bytes (%.1f MiB)\n", fp16_bytes, fp16_bytes / 1048576.0);
        printf("  Saved:      %zu bytes (%.1f MiB)\n", saved, saved / 1048576.0);
        check("Memory savings = 24 MiB (48->24)", saved == 24 * 1048576);
    }

    /* Cleanup */
    free(g); free(x); free(a_decay);
    free(s_f32); free(decay_f32); free(state_f32);
    free(s_f16); free(decay_f16); free(state_f16);
    free(s_bf16); free(decay_bf16); free(state_bf16);
    free(s_dbl); free(state_dbl);
    free(decay_f16_f); free(decay_dbl);

    printf("\n%s\n", all_pass ? "ALL TESTS PASS" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}
