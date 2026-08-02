/* Standalone microbenchmark for the Gated DeltaNet CPU kernels.
 *
 * Bead ob-8ms.2. Turns "verified correct" into "measured" on real Arm silicon.
 *
 * Deliberately dependency-free and statically linkable, so a single binary can be copied to
 * any aarch64 device -- an Armv8 phone or SBC, a Graviton instance, or the Orion O6 -- and run
 * with no toolchain, no Python, and no shared libraries on the target.
 *
 *   aarch64-linux-gnu-gcc -O3 -march=armv8-a -static gdn_sve.c bench_gdn.c -o bench_gdn -lm
 *   scp bench_gdn device:/tmp/ && ssh device /tmp/bench_gdn --csv
 *
 * Follows docs/METRICS.md: warmups are discarded, N repeats are timed individually, and p50/p95
 * are reported. Never reports a single best run. Emits rows matching docs/RESULTS_SCHEMA.md.
 *
 * ARMV8 EXPECTATIONS, stated up front so the output is not misread: most Armv8-A cores have no
 * SVE at all, so on those this measures the NEON path -- which is the point, since that is the
 * path the large installed base of Armv8 devices actually runs. i8mm requires Armv8.6-A and
 * dotprod Armv8.2-A; neither is used by these fp32 kernels. The binary reports the dispatch path
 * that was compiled in so results are never attributed to the wrong one.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void gdn_cumdecay_f32(const float *, float *, size_t, size_t);
void gdn_gated_scan_f32(const float *, const float *, float *, float *, size_t, size_t);
void gdn_causal_dwconv1d_f32(const float *, const float *, float *, float *, size_t, size_t);
void gdn_cumdecay_bf16(const float *, uint16_t *, size_t, size_t);
void gdn_gated_scan_bf16(const float *, const float *, float *, uint16_t *, size_t, size_t);
void gdn_cumdecay_f16(const float *, uint16_t *, size_t, size_t);
void gdn_gated_scan_f16(const float *, const float *, float *, uint16_t *, size_t, size_t);

/* Which path did the compiler actually select in gdn_sve.c? Mirrors its guard order exactly. */
#if defined(__ARM_FEATURE_SVE)
#define DISPATCH_PATH "sve"
#elif defined(__ARM_NEON)
#define DISPATCH_PATH "neon"
#else
#define DISPATCH_PATH "scalar"
#endif

#define WARMUPS 3
#define MAX_REPEATS 64

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* Percentile by nearest-rank on the sorted sample. At N=10-30 there is no meaningful
 * difference between interpolation methods, and nearest-rank never invents a value that
 * was not observed. */
static double pct(double *sorted, int n, double p) {
    if (n <= 0) return 0.0;
    int idx = (int)(p * (double)(n - 1) + 0.5);
    if (idx < 0) idx = 0;
    if (idx >= n) idx = n - 1;
    return sorted[idx];
}

typedef struct {
    double p50, p95, min, max;
    int repeats;
} stats_t;

static stats_t summarize(double *samples, int n) {
    qsort(samples, (size_t)n, sizeof(double), cmp_double);
    stats_t s;
    s.repeats = n;
    s.p50 = pct(samples, n, 0.50);
    s.p95 = pct(samples, n, 0.95);
    s.min = samples[0];
    s.max = samples[n - 1];
    return s;
}

typedef enum { K_DECAY, K_SCAN, K_CONV,
               K_DECAY_BF16, K_SCAN_BF16,
               K_DECAY_F16, K_SCAN_F16,
               K_MAX } kernel_id;

static const char *kernel_name(kernel_id k) {
    switch (k) {
        case K_DECAY:      return "gdn_cumdecay";
        case K_SCAN:       return "gdn_gated_scan";
        case K_CONV:       return "gdn_causal_dwconv1d";
        case K_DECAY_BF16: return "gdn_cumdecay_bf16";
        case K_SCAN_BF16:  return "gdn_gated_scan_bf16";
        case K_DECAY_F16:  return "gdn_cumdecay_f16";
        case K_SCAN_F16:   return "gdn_gated_scan_f16";
        case K_MAX:        return "?";
    }
    return "?";
}

/* Bytes of traffic per call, counting only the unavoidable streams. This is what the
 * bandwidth-bound argument in docs/METRICS.md is built on, so the accounting is explicit
 * rather than folded into a constant. */
static double bytes_per_call(kernel_id k, size_t seq, size_t ch) {
    double e = (double)sizeof(float);   /* 4 bytes (fp32) */
    double h = (double)sizeof(uint16_t); /* 2 bytes (bf16/fp16) */
    double s = (double)seq, c = (double)ch;
    switch (k) {
        /* read a[], write decay[] — both fp32 */
        case K_DECAY:      return e * 2.0 * s * c;
        /* read g[], read x[], write s[], plus state read+write — all fp32 */
        case K_SCAN:       return e * (3.0 * s * c + 2.0 * c);
        /* read in[], write out[], weights + history are small and resident */
        case K_CONV:       return e * (2.0 * s * c + 7.0 * c);
        /* read a[fp32], write decay[bf16] — saves 25% output traffic */
        case K_DECAY_BF16: return e * s * c + h * s * c;
        /* read g+x[fp32], write s[fp32], read+write state[bf16] — saves state I/O */
        case K_SCAN_BF16:  return e * 3.0 * s * c + h * 2.0 * c;
        /* read a[fp32], write decay[fp16] — same traffic as bf16 variant */
        case K_DECAY_F16:  return e * s * c + h * s * c;
        /* read g+x[fp32], write s[fp32], read+write state[fp16] */
        case K_SCAN_F16:   return e * 3.0 * s * c + h * 2.0 * c;
        case K_MAX:        return 0.0;
    }
    return 0.0;
}

static double flops_per_call(kernel_id k, size_t seq, size_t ch) {
    double n = (double)seq * (double)ch;
    switch (k) {
        case K_DECAY:      return n;             /* one multiply per element */
        case K_SCAN:       return 2.0 * n;       /* one FMA per element */
        case K_CONV:       return 8.0 * n;       /* 4 taps, mul + 3 FMA */
        /* Narrow-format variants do the same FLOPs — only I/O width changes */
        case K_DECAY_BF16: return n;
        case K_SCAN_BF16:  return 2.0 * n;
        case K_DECAY_F16:  return n;
        case K_SCAN_F16:   return 2.0 * n;
        case K_MAX:        return 0.0;
    }
    return 0.0;
}

int main(int argc, char **argv) {
    int csv = 0, repeats = 15;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--csv")) csv = 1;
        else if (!strcmp(argv[i], "--repeats") && i + 1 < argc) repeats = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--help")) {
            printf("usage: %s [--csv] [--repeats N]\n", argv[0]);
            return 0;
        }
    }
    if (repeats < 5) repeats = 5;            /* docs/METRICS.md: never report N<5 */
    if (repeats > MAX_REPEATS) repeats = MAX_REPEATS;

    /* Verified Qwen3.5 shapes (ADR 0003). channels = n_value_heads * head_dim. */
    struct { const char *model; size_t seq, ch, gdn_layers; } cfgs[] = {
        {"Qwen3.5-4B", 64, 32 * 128, 24},
        {"Qwen3.5-0.8B", 64, 16 * 128, 18},
        /* Decode (seq=1): state I/O is ~40% of traffic — where narrowing helps */
        {"Qwen3.5-4B_decode", 1, 32 * 128, 24},
        {"Qwen3.5-0.8B_decode", 1, 16 * 128, 18},
    };
    const int n_cfg = (int)(sizeof(cfgs) / sizeof(cfgs[0]));

    if (!csv) {
        printf("GDN CPU kernel microbenchmark\n");
        printf("  dispatch path compiled in : %s\n", DISPATCH_PATH);
        printf("  sizeof(float)             : %zu\n", sizeof(float));
        printf("  warmups (discarded)       : %d\n", WARMUPS);
        printf("  timed repeats             : %d\n", repeats);
        printf("  protocol                  : docs/METRICS.md (p50/p95, no single-best runs)\n\n");
    } else {
        printf("model,kernel,dispatch_path,seq,channels,repeats,"
               "p50_us,p95_us,spread_pct,gib_per_s_p50,gflop_per_s_p50\n");
    }

    for (int c = 0; c < n_cfg; ++c) {
        size_t seq = cfgs[c].seq, ch = cfgs[c].ch;
        size_t n = seq * ch;

        float *a = malloc(n * sizeof(float));
        float *out = malloc(n * sizeof(float));
        float *g = malloc(n * sizeof(float));
        float *x = malloc(n * sizeof(float));
        float *state = malloc(ch * sizeof(float));
        float *w = malloc(4 * ch * sizeof(float));
        float *hist = malloc(3 * ch * sizeof(float));
        uint16_t *decay_narrow = malloc(n * sizeof(uint16_t));
        uint16_t *state_narrow = malloc(ch * sizeof(uint16_t));
        if (!a || !out || !g || !x || !state || !w || !hist ||
            !decay_narrow || !state_narrow) {
            fprintf(stderr, "allocation failed for %s (needs ~%.0f MiB)\n", cfgs[c].model,
                    (double)(4 * n + 8 * ch) * sizeof(float) / 1048576.0);
            return 1;
        }
        /* Decay values in (0.90, 0.99): representative, and keeps the cumulative product
         * well inside fp32 range over a 64-step chunk. */
        for (size_t i = 0; i < n; ++i) {
            a[i] = 0.90f + 0.09f * (float)((i * 2654435761u) % 1000) / 1000.0f;
            g[i] = 0.50f + 0.40f * (float)((i * 40503u) % 1000) / 1000.0f;
            x[i] = (float)((i * 69069u) % 2000) / 1000.0f - 1.0f;
        }
        for (size_t i = 0; i < ch; ++i) state[i] = 0.0f;
        for (size_t i = 0; i < ch; ++i) state_narrow[i] = 0;
        for (size_t i = 0; i < 4 * ch; ++i) w[i] = 0.1f;
        for (size_t i = 0; i < 3 * ch; ++i) hist[i] = 0.0f;

        if (!csv)
            printf("%s  (seq=%zu, channels=%zu, %zu GDN layers)\n", cfgs[c].model, seq, ch,
                   cfgs[c].gdn_layers);

        for (kernel_id k = K_DECAY; k < K_MAX; ++k) {
            double samples[MAX_REPEATS];
            for (int r = 0; r < WARMUPS + repeats; ++r) {
                double t0 = now_s();
                switch (k) {
                    case K_DECAY:      gdn_cumdecay_f32(a, out, seq, ch); break;
                    case K_SCAN:       gdn_gated_scan_f32(g, x, out, state, seq, ch); break;
                    case K_CONV:       gdn_causal_dwconv1d_f32(x, w, out, hist, seq, ch); break;
                    case K_DECAY_BF16: gdn_cumdecay_bf16(a, decay_narrow, seq, ch); break;
                    case K_SCAN_BF16:  gdn_gated_scan_bf16(g, x, out, state_narrow, seq, ch); break;
                    case K_DECAY_F16:  gdn_cumdecay_f16(a, decay_narrow, seq, ch); break;
                    case K_SCAN_F16:   gdn_gated_scan_f16(g, x, out, state_narrow, seq, ch); break;
                    case K_MAX:        break;
                }
                double dt = now_s() - t0;
                if (r >= WARMUPS) samples[r - WARMUPS] = dt;
            }
            stats_t s = summarize(samples, repeats);
            double gibs = bytes_per_call(k, seq, ch) / s.p50 / 1073741824.0;
            double gflops = flops_per_call(k, seq, ch) / s.p50 / 1e9;
            double spread = s.p50 > 0.0 ? (s.p95 - s.p50) / s.p50 * 100.0 : 0.0;

            if (csv) {
                printf("%s,%s,%s,%zu,%zu,%d,%.3f,%.3f,%.1f,%.2f,%.2f\n", cfgs[c].model,
                       kernel_name(k), DISPATCH_PATH, seq, ch, repeats, s.p50 * 1e6,
                       s.p95 * 1e6, spread, gibs, gflops);
            } else {
                printf("  %-22s p50 %8.2f us   p95 %8.2f us   spread %5.1f%%   "
                       "%6.2f GiB/s   %6.2f GFLOP/s\n",
                       kernel_name(k), s.p50 * 1e6, s.p95 * 1e6, spread, gibs, gflops);
            }
        }

        /* Per-token decode cost extrapolated across all GDN layers -- the figure that connects
         * these microbenchmarks to the end-to-end bandwidth-bound argument. seq=1 is the decode
         * case, so scale the per-chunk scan cost down by the chunk length. */
        if (!csv) {
            printf("\n");
        }
        free(a); free(out); free(g); free(x); free(state); free(w); free(hist);
        free(decay_narrow); free(state_narrow);
    }

    if (!csv) {
        printf("Notes\n");
        printf("  - '%s' is the path the compiler selected, not necessarily the best the CPU\n",
               DISPATCH_PATH);
        printf("    can do. On Armv8-A without SVE the NEON path is expected and correct.\n");
        printf("  - GiB/s counts only unavoidable streams (see bytes_per_call in the source).\n");
        printf("  - Compare GiB/s against the device's spec bandwidth to test the\n");
        printf("    bandwidth-bound thesis rather than assuming it.\n");
    }
    return 0;
}
