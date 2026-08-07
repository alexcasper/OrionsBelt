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

#define _POSIX_C_SOURCE 200112L  /* clock_gettime, CLOCK_MONOTONIC */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "gdn_sve.h"

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
#define BATCH 100  /* batched timing: amortize clock_gettime overhead on cores
                    * with coarse CLOCK_MONOTONIC_RAW granularity (A57: ~2.4 µs).
                    * Matches the KleidiAI bench fix (commit 7f418d2). */

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
               K_DECAY_F16, K_SCAN_F16, K_DECAY_BF16, K_SCAN_BF16,
               K_SCAN2, K_LAST } kernel_id;

static const char *kernel_name(kernel_id k) {
    switch (k) {
        case K_DECAY:      return "gdn_cumdecay";
        case K_SCAN:       return "gdn_gated_scan";
        case K_CONV:       return "gdn_causal_dwconv1d";
        case K_DECAY_F16:  return "gdn_cumdecay_f16";
        case K_SCAN_F16:   return "gdn_gated_scan_f16";
        case K_DECAY_BF16: return "gdn_cumdecay_bf16";
        case K_SCAN_BF16:  return "gdn_gated_scan_bf16";
        case K_SCAN2:      return "gdn2_gated_scan";
        case K_LAST:       break;
    }
    return "?";
}

/* Bytes of traffic per call, counting only the unavoidable streams. This is what the
 * bandwidth-bound argument in docs/METRICS.md is built on, so the accounting is explicit
 * rather than folded into a constant. */
static double bytes_per_call(kernel_id k, size_t seq, size_t ch) {
    double e = (double)sizeof(float);
    switch (k) {
        /* read a[], write decay[] */
        case K_DECAY: return e * 2.0 * (double)seq * (double)ch;
        /* read g[], read x[], write s[], plus state read+write */
        case K_SCAN: return e * (3.0 * (double)seq * (double)ch + 2.0 * (double)ch);
        /* read in[], write out[], weights + history are small and resident */
        case K_CONV: return e * (2.0 * (double)seq * (double)ch + 7.0 * (double)ch);
        /* fp16/bf16 variants: state/output is 2 bytes instead of 4.
         * The arithmetic stays fp32; only the persistent storage is narrowed. */
        case K_DECAY_F16:  /* read a[] (fp32) + write decay[] (fp16) */
            return e * (double)seq * (double)ch + 2.0 * (double)seq * (double)ch;
        case K_DECAY_BF16: return e * (double)seq * (double)ch + 2.0 * (double)seq * (double)ch;
        case K_SCAN_F16:   /* read g[], x[] (fp32) + write s[] (fp32) + state (fp16) */
            return e * 3.0 * (double)seq * (double)ch + 2.0 * 2.0 * (double)ch;
        case K_SCAN_BF16:  return e * 3.0 * (double)seq * (double)ch + 2.0 * 2.0 * (double)ch;
        /* GDN-2: same as scan plus 2 extra streams (b_gate, w_gate) */
        case K_SCAN2:      return e * (5.0 * (double)seq * (double)ch + 2.0 * (double)ch);
        case K_LAST: break;
    }
    return 0.0;
}

static double flops_per_call(kernel_id k, size_t seq, size_t ch) {
    double n = (double)seq * (double)ch;
    switch (k) {
        case K_DECAY: return n;             /* one multiply per element */
        case K_SCAN: return 2.0 * n;        /* one FMA per element */
        case K_CONV: return 8.0 * n;        /* 4 taps, mul + 3 FMA */
        /* fp16/bf16 variants: identical arithmetic to fp32 (mixed precision) */
        case K_DECAY_F16:  case K_DECAY_BF16: return n;
        case K_SCAN_F16:   case K_SCAN_BF16:  return 2.0 * n;
        /* GDN-2: 1 FMA + 2 extra muls = 4 FLOPs/element */
        case K_SCAN2:      return 4.0 * n;
        case K_LAST: break;
    }
    return 0.0;
}

/* ---- Sustained-load mode (ob-mrd.2, ported from j1) ----
 * Runs gdn_gated_scan on the largest config continuously for N seconds,
 * sampling throughput and CPU temperature every ~5 seconds to reveal
 * thermal throttling.  PLAN.md risk R7: burst numbers that cannot be
 * sustained are misleading on passively-cooled edge hardware. */

static double read_thermal_millideg(void) {
    FILE *f = fopen("/sys/class/thermal/thermal_zone0/temp", "r");
    if (!f) return -1.0;
    double val = -1.0;
    if (fscanf(f, "%lf", &val) != 1) val = -1.0;
    fclose(f);
    return val;                       /* millidegrees Celsius, or -1 */
}

static kernel_id parse_kernel(const char *s) {
    if (!strcmp(s, "cumdecay"))        return K_DECAY;
    if (!strcmp(s, "gated_scan"))      return K_SCAN;
    if (!strcmp(s, "dwconv1d"))        return K_CONV;
    if (!strcmp(s, "cumdecay_f16"))    return K_DECAY_F16;
    if (!strcmp(s, "gated_scan_f16"))  return K_SCAN_F16;
    if (!strcmp(s, "cumdecay_bf16"))   return K_DECAY_BF16;
    if (!strcmp(s, "gated_scan_bf16")) return K_SCAN_BF16;
    if (!strcmp(s, "gdn2_gated_scan")) return K_SCAN2;
    return K_LAST;  /* invalid */
}

static void run_sustained(int seconds, int csv_mode,
                          kernel_id kid, const char *model_name,
                          size_t seq, size_t ch) {
    size_t n = seq * ch;

    /* Allocate all buffers (some kernels don't use all of them, but the
     * allocation is tiny relative to the run time and keeps the code simple). */
    float *a   = malloc(n * sizeof(float));
    float *o   = malloc(n * sizeof(float));
    float *g   = malloc(n * sizeof(float));
    float *x   = malloc(n * sizeof(float));
    float *wg  = malloc(n * sizeof(float));  /* GDN-2 write gate (separate from x) */
    float *st  = malloc(ch * sizeof(float));
    float *w   = malloc(4 * ch * sizeof(float));
    float *hist = malloc(3 * ch * sizeof(float));
    __fp16 *decay_f16  = malloc(n * sizeof(__fp16));
    __fp16 *state_f16  = malloc(ch * sizeof(__fp16));
    uint16_t *decay_bf16 = malloc(n * sizeof(uint16_t));
    uint16_t *state_bf16 = malloc(ch * sizeof(uint16_t));
    if (!a || !o || !g || !x || !wg || !st || !w || !hist ||
        !decay_f16 || !state_f16 || !decay_bf16 || !state_bf16) {
        fprintf(stderr, "sustained: allocation failed\n");
        return;
    }

    for (size_t i = 0; i < n; ++i) {
        a[i] = 0.90f + 0.09f * (float)((i * 2654435761u) % 1000) / 1000.0f;
        g[i] = 0.50f + 0.40f * (float)((i * 40503u) % 1000) / 1000.0f;
        x[i] = (float)((i * 69069u) % 2000) / 1000.0f - 1.0f;
        wg[i] = 0.50f + 0.49f * (float)((i * 2246822519u) % 1000) / 1000.0f;
    }
    for (size_t i = 0; i < ch; ++i) { st[i] = 0.0f; state_f16[i] = 0; state_bf16[i] = 0; }
    for (size_t i = 0; i < 4 * ch; ++i) w[i] = 0.1f;
    for (size_t i = 0; i < 3 * ch; ++i) hist[i] = 0.0f;

    double bpc = bytes_per_call(kid, seq, ch);
    double sample_int = 5.0;          /* seconds between samples */
    double t0 = now_s();
    double deadline = t0 + (double)seconds;
    double next_sample = t0 + sample_int;
    long calls_in_window = 0;
    double window_start = t0;
    double first_window_gibs = 0.0;
    int window_idx = 0;

    if (!csv_mode) {
        printf("Sustained-load: %s, %s (seq=%zu), %d seconds\n",
               kernel_name(kid), model_name, seq, seconds);
        printf("  dispatch path: %s\n\n", DISPATCH_PATH);
        printf("  elapsed  throughput   thermal   vs_first\n");
        printf("   (sec)    (GiB/s)     (C)        (%%)\n");
        printf("  ------  ---------   --------   -------\n");
    } else {
        printf("sustained_model,sustained_kernel,dispatch_path,elapsed_s,"
               "throughput_gibs,thermal_c,vs_first_pct\n");
    }

    while (now_s() < deadline) {
        switch (kid) {
            case K_DECAY:      gdn_cumdecay_f32(a, o, seq, ch); break;
            case K_SCAN:       gdn_gated_scan_f32(g, x, o, st, seq, ch); break;
            case K_CONV:       gdn_causal_dwconv1d_f32(x, w, o, hist, seq, ch); break;
            case K_DECAY_F16:  gdn_cumdecay_f16(a, decay_f16, seq, ch); break;
            case K_SCAN_F16:   gdn_gated_scan_f16(g, x, o, state_f16, seq, ch); break;
            case K_DECAY_BF16: gdn_cumdecay_bf16(a, decay_bf16, seq, ch); break;
            case K_SCAN_BF16:  gdn_gated_scan_bf16(g, x, o, state_bf16, seq, ch); break;
            case K_SCAN2:      gdn2_gated_scan_f32(g, a, wg, x, o, st, seq, ch); break;
            case K_LAST:       break;
        }
        calls_in_window++;

        double now = now_s();
        if (now >= next_sample) {
            double dt = now - window_start;
            double avg_s = calls_in_window > 0 ? dt / (double)calls_in_window : dt;
            double gibs = bpc / avg_s / 1073741824.0;
            double temp = read_thermal_millideg();
            if (temp >= 0.0) temp /= 1000.0;
            double vs_first = 0.0;
            if (window_idx == 0) first_window_gibs = gibs;
            if (first_window_gibs > 0.0)
                vs_first = (gibs - first_window_gibs) / first_window_gibs * 100.0;

            if (csv_mode) {
                printf("%s,%s,%s,%.1f,%.2f,%.1f,%.1f\n",
                       model_name, kernel_name(kid), DISPATCH_PATH,
                       now - t0, gibs, temp, vs_first);
            } else {
                printf("  %6.1f   %8.2f   %7.1f   %+6.1f\n",
                       now - t0, gibs, temp, vs_first);
            }
            fflush(stdout);
            calls_in_window = 0;
            window_start = now;
            next_sample += sample_int;
            window_idx++;
        }
    }

    if (!csv_mode && window_idx > 1) {
        printf("\n  Burst throughput (first 5 s): %.2f GiB/s\n", first_window_gibs);
        printf("  Steady-state visible in throughput column — look for decay.\n");
    }

    free(a); free(o); free(g); free(x); free(wg); free(st);
    free(w); free(hist);
    free(decay_f16); free(state_f16);
    free(decay_bf16); free(state_bf16);
}

int main(int argc, char **argv) {
    int csv = 0, repeats = 15, sustained = 0;
    kernel_id sus_kernel = K_SCAN;
    const char *sus_model = "Qwen3.5-4B";
    size_t sus_seq = 64, sus_ch = 32 * 128;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--csv")) csv = 1;
        else if (!strcmp(argv[i], "--repeats") && i + 1 < argc) repeats = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--sustained") && i + 1 < argc) sustained = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--sustained-kernel") && i + 1 < argc) {
            sus_kernel = parse_kernel(argv[++i]);
            if (sus_kernel == K_LAST) {
                fprintf(stderr, "unknown kernel: %s\n  valid: cumdecay gated_scan dwconv1d "
                        "cumdecay_f16 gated_scan_f16 cumdecay_bf16 gated_scan_bf16\n", argv[i]);
                return 1;
            }
        }
        else if (!strcmp(argv[i], "--sustained-model") && i + 1 < argc) {
            const char *m = argv[++i];
            if (!strcmp(m, "4B")) { sus_model = "Qwen3.5-4B"; sus_ch = 32 * 128; }
            else if (!strcmp(m, "0.8B")) { sus_model = "Qwen3.5-0.8B"; sus_ch = 16 * 128; }
            else { fprintf(stderr, "unknown model: %s (use 4B or 0.8B)\n", m); return 1; }
        }
        else if (!strcmp(argv[i], "--sustained-seq") && i + 1 < argc) {
            sus_seq = (size_t)atoi(argv[++i]);
        }
        else if (!strcmp(argv[i], "--help")) {
            printf("usage: %s [--csv] [--repeats N] [--sustained SECONDS] \\\n", argv[0]);
            printf("         [--sustained-kernel NAME] [--sustained-model 4B|0.8B] [--sustained-seq N]\n");
            printf("\n  --sustained N       Run a kernel for N seconds, sampling\n");
            printf("                      throughput and thermal every 5 s to reveal throttling.\n");
            printf("  --sustained-kernel  Kernel for sustained mode (default: gated_scan)\n");
            printf("                      Valid: cumdecay gated_scan dwconv1d\n");
            printf("                             cumdecay_f16 gated_scan_f16\n");
            printf("                             cumdecay_bf16 gated_scan_bf16\n");
            printf("  --sustained-model   Model config: 4B or 0.8B (default: 4B)\n");
            printf("  --sustained-seq     Sequence length: 64=prefill, 1=decode (default: 64)\n");
            return 0;
        }
    }
    if (repeats < 5) repeats = 5;            /* docs/METRICS.md: never report N<5 */
    if (repeats > MAX_REPEATS) repeats = MAX_REPEATS;

    /* Sustained-load mode: separate code path from the burst benchmark. */
    if (sustained > 0) {
        run_sustained(sustained, csv, sus_kernel, sus_model, sus_seq, sus_ch);
        return 0;
    }

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
        printf("  batched calls/repeat      : %d max (adaptive: probed per kernel)\n", BATCH);
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
        float *wg = malloc(n * sizeof(float));  /* GDN-2 write gate (separate from x) */
        float *state = malloc(ch * sizeof(float));
        float *w = malloc(4 * ch * sizeof(float));
        float *hist = malloc(3 * ch * sizeof(float));
        __fp16 *state_f16 = malloc(ch * sizeof(__fp16));
        uint16_t *state_bf16 = malloc(ch * sizeof(uint16_t));
        __fp16 *decay_f16 = malloc(n * sizeof(__fp16));
        uint16_t *decay_bf16 = malloc(n * sizeof(uint16_t));
        if (!a || !out || !g || !x || !wg || !state || !w || !hist ||
            !state_f16 || !state_bf16 || !decay_f16 || !decay_bf16) {
            fprintf(stderr, "allocation failed for %s (needs ~%.0f MiB)\n", cfgs[c].model,
                    (double)(5 * n + 8 * ch) * sizeof(float) / 1048576.0);
            return 1;
        }
        /* Decay values in (0.90, 0.99): representative, and keeps the cumulative product
         * well inside fp32 range over a 64-step chunk. */
        for (size_t i = 0; i < n; ++i) {
            a[i] = 0.90f + 0.09f * (float)((i * 2654435761u) % 1000) / 1000.0f;
            g[i] = 0.50f + 0.40f * (float)((i * 40503u) % 1000) / 1000.0f;
            x[i] = (float)((i * 69069u) % 2000) / 1000.0f - 1.0f;
            wg[i] = 0.50f + 0.49f * (float)((i * 2246822519u) % 1000) / 1000.0f;
        }
        for (size_t i = 0; i < ch; ++i) state[i] = 0.0f;
        for (size_t i = 0; i < 4 * ch; ++i) w[i] = 0.1f;
        for (size_t i = 0; i < 3 * ch; ++i) hist[i] = 0.0f;
        for (size_t i = 0; i < ch; ++i) state_f16[i] = 0.0f;
        for (size_t i = 0; i < ch; ++i) state_bf16[i] = 0;

        if (!csv)
            printf("%s  (seq=%zu, channels=%zu, %zu GDN layers)\n", cfgs[c].model, seq, ch,
                   cfgs[c].gdn_layers);

        for (kernel_id k = K_DECAY; k < K_LAST; ++k) {
            double samples[MAX_REPEATS];

            /* Adaptive batching: probe one call. If it's short enough that
             * clock_gettime overhead (~2.4 µs on A57, ~291 ns on RK3588)
             * dominates, use batched timing. Otherwise single-call — avoids
             * thermal-throttling and state-accumulation artefacts on long
             * kernels. */
            double t_probe = now_s();
            switch (k) {
                case K_DECAY:      gdn_cumdecay_f32(a, out, seq, ch); break;
                case K_SCAN:       gdn_gated_scan_f32(g, x, out, state, seq, ch); break;
                case K_CONV:       gdn_causal_dwconv1d_f32(x, w, out, hist, seq, ch); break;
                case K_DECAY_F16:  gdn_cumdecay_f16(a, decay_f16, seq, ch); break;
                case K_SCAN_F16:   gdn_gated_scan_f16(g, x, out, state_f16, seq, ch); break;
                case K_DECAY_BF16: gdn_cumdecay_bf16(a, decay_bf16, seq, ch); break;
                case K_SCAN_BF16:  gdn_gated_scan_bf16(g, x, out, state_bf16, seq, ch); break;
                case K_SCAN2:      gdn2_gated_scan_f32(g, a, wg, x, out, state, seq, ch); break;
                case K_LAST: break;
            }
            t_probe = now_s() - t_probe;
            int batch = (t_probe < 20e-6) ? BATCH : 1;

            for (int r = 0; r < WARMUPS + repeats; ++r) {
                double t0 = now_s();
                for (int b = 0; b < batch; ++b) {
                    switch (k) {
                        case K_DECAY:      gdn_cumdecay_f32(a, out, seq, ch); break;
                        case K_SCAN:       gdn_gated_scan_f32(g, x, out, state, seq, ch); break;
                        case K_CONV:       gdn_causal_dwconv1d_f32(x, w, out, hist, seq, ch); break;
                        case K_DECAY_F16:  gdn_cumdecay_f16(a, decay_f16, seq, ch); break;
                        case K_SCAN_F16:   gdn_gated_scan_f16(g, x, out, state_f16, seq, ch); break;
                        case K_DECAY_BF16: gdn_cumdecay_bf16(a, decay_bf16, seq, ch); break;
                        case K_SCAN_BF16:  gdn_gated_scan_bf16(g, x, out, state_bf16, seq, ch); break;
                        case K_SCAN2:      gdn2_gated_scan_f32(g, a, wg, x, out, state, seq, ch); break;
                        case K_LAST: break;
                    }
                }
                double dt = (now_s() - t0) / batch;
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
        free(a); free(out); free(g); free(x); free(wg); free(state); free(w); free(hist);
        free(state_f16); free(state_bf16); free(decay_f16); free(decay_bf16);
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
