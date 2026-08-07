/* End-to-end CPU decode loop for Qwen3.5 (bead ob-8qt.9).
 *
 * Wires the optimized GDN kernels (gdn_sve.c, gdn_delta_matmul.c) into a
 * full forward pass over every GDN + full-attention layer + FFN block,
 * measuring per-token wall-clock time and reporting tokens/sec, first-token
 * latency (TTFT), and a coarse per-phase bottleneck breakdown.
 *
 * Two verified checkpoints (src/orionsbelt/model/gdn_layer_info.py), selected
 * at compile time (default 4B):
 *   4B:   hidden=2560  layers=32 (24 GDN + 8 full, 8x(3 GDN,1 full))
 *         key_dim=2048  value_dim=4096  conv_dim=8192  intermediate=9216
 *         full_attn: heads=16 head_dim=256 kv_heads=4
 *   0.8B: hidden=1024  layers=24 (18 GDN + 6 full, 6x(3 GDN,1 full))
 *         key_dim=2048  value_dim=2048  conv_dim=6144  intermediate=3584
 *         full_attn: heads=8  head_dim=256 kv_heads=2
 * Both: vocab=248320  conv_kernel=4  num_key_heads=16  head_dim=128
 *
 * Weights are random (benchmark only — correctness is validated by the
 * per-kernel test suites). The point is to measure the kernel + memory
 * traffic cost of a real decode step at real shapes, not to produce text.
 *
 * Build (native on any aarch64):
 *   cc -O3 -fopenmp -march=<isa> -static \
 *     gdn_sve.c gdn_delta_matmul.c gdn_e2e_decode.c -I. -o gdn_e2e_decode -lm
 *   # add -DMODEL_08B for the 0.8B variant (build a separate binary)
 *
 * Run:
 *   ./gdn_e2e_decode --tokens 128           # human-readable
 *   ./gdn_e2e_decode --tokens 128 --csv     # CSV for fleet sweep
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdint.h>

#include "gdn_delta_matmul.h"

/* ---- Kernel prototypes (gdn_sve.c) ---- */
extern void gdn_cumdecay_f32(const float *a, float *decay, size_t seq, size_t channels);
extern void gdn_gated_scan_f32(const float *g, const float *x, float *s, float *state,
                               size_t seq, size_t channels);
extern void gdn_causal_dwconv1d_f32(const float *in, const float *w, float *out,
                                    float *hist, size_t seq, size_t channels);

/* ---- Qwen3.5 layer geometry ----
 * Default is 4B. Build with -DMODEL_08B for the 0.8B variant (both verified
 * against src/orionsbelt/model/gdn_layer_info.py). */
#ifdef MODEL_08B
#define MODEL_NAME   "Qwen3.5-0.8B"
#define HIDDEN       1024
#define NUM_LAYERS   24
#define NUM_GDN      18
#define NUM_FULL     6
#define KEY_DIM      2048
#define VALUE_DIM    2048
#define CONV_DIM     6144
#define INTER        3584
#define VOCAB        248320
#define CONV_K       4
#define NUM_K_HEADS  16
#define NUM_V_HEADS  16
#define HEAD_DIM     128
#define FULL_HEADS   8
#define FULL_HEAD_DIM 256
#define FULL_KV_HEADS 2
#else
#define MODEL_NAME   "Qwen3.5-4B"
#define HIDDEN       2560
#define NUM_LAYERS   32
#define NUM_GDN      24
#define NUM_FULL     8
#define KEY_DIM      2048
#define VALUE_DIM    4096
#define CONV_DIM     8192
#define INTER        9216
#define VOCAB        248320
#define CONV_K       4
#define NUM_K_HEADS  16
#define NUM_V_HEADS  32
#define HEAD_DIM     128
#define FULL_HEADS   16
#define FULL_HEAD_DIM 256
#define FULL_KV_HEADS 4
#endif

/* ---- Helpers ---- */
static float *alloc_aligned(size_t n) {
    void *p = NULL;
    if (posix_memalign(&p, 64, n * sizeof(float)) != 0) {
        fprintf(stderr, "OOM allocating %zu floats\n", n);
        exit(1);
    }
    memset(p, 0, n * sizeof(float));
    return (float *)p;
}

static void fill_rand(float *buf, size_t n, unsigned *seed) {
    for (size_t i = 0; i < n; ++i)
        buf[i] = ((float)(rand_r(seed) % 2000) - 1000) / 1000.0f;
}

/* GEMV/MATMUL for projection layers.
 * B is row-major [K x N]. For decode (M=1) this is a GEMV — the critical path.
 * Uses NEON for the inner reduction when available; scalar fallback otherwise. */
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

static void gemv_neon(const float *a, const float *B, float *c, size_t K, size_t N) {
    /* a is [K], B is [K x N] row-major, c is [N].
     * Walk N in NEON-width strides, reduce over K. */
    size_t j = 0;
    for (; j + 4 <= N; j += 4) {
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (size_t k = 0; k < K; ++k) {
            float32x4_t bk = vld1q_f32(B + k * N + j);
            float32x4_t ak = vdupq_n_f32(a[k]);
            acc = vfmaq_f32(acc, ak, bk);
        }
        vst1q_f32(c + j, acc);
    }
    /* Tail: remaining columns when N % 4 != 0 */
    for (; j < N; ++j) {
        float acc = 0.0f;
        for (size_t k = 0; k < K; ++k)
            acc += a[k] * B[k * N + j];
        c[j] = acc;
    }
}

static void matmul(const float *A, const float *B, float *C,
                   size_t M, size_t K, size_t N) {
#ifdef __ARM_NEON
    if (M == 1) {
        gemv_neon(A, B, C, K, N);
        return;
    }
#endif
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (size_t k = 0; k < K; ++k)
                acc += A[i * K + k] * B[k * N + j];
            C[i * N + j] = acc;
        }
    }
}

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

/* ---- Bottleneck breakdown: coarse phase timing, reset before the measured
 * loop and accumulated across every layer + token. Single-threaded caller
 * only (main's decode loop), so plain globals are fine. */
static double g_t_gdn_proj = 0;   /* GDN q/k/v projections */
static double g_t_gdn_conv = 0;   /* causal depthwise conv1d */
static double g_t_gdn_decay = 0;  /* cumulative decay */
static double g_t_gdn_scan = 0;   /* gated delta-rule scan */
static double g_t_gdn_oproj = 0;  /* GDN output projection */
static double g_t_full = 0;       /* full-attention layers (whole) */
static double g_t_ffn = 0;        /* FFN blocks (whole) */

/* ---- Per-layer persistent state ---- */
typedef struct {
    /* GDN recurrent state: NUM_V_HEADS * HEAD_DIM = 4096 floats */
    float *gdn_state;
    /* Conv1D history: CONV_K-1 = 3 * CONV_DIM channels */
    float *conv_hist;
    /* Full-attention KV cache (simulated, 1 entry for single-token decode) */
    float *kv_cache_k;
    float *kv_cache_v;
} LayerState;

/* ---- Weights (random, benchmark only) ---- */
typedef struct {
    /* GDN layer weights */
    float *g_q_proj;   /* HIDDEN x KEY_DIM */
    float *g_k_proj;   /* HIDDEN x KEY_DIM */
    float *g_v_proj;   /* HIDDEN x VALUE_DIM */
    float *g_o_proj;   /* VALUE_DIM x HIDDEN */
    float *g_conv_w;   /* CONV_DIM x CONV_K */
    float *g_beta;     /* KEY_DIM */
    /* Full-attention weights */
    float *f_q_proj;   /* HIDDEN x (FULL_HEADS*FULL_HEAD_DIM) */
    float *f_k_proj;   /* HIDDEN x (FULL_KV_HEADS*FULL_HEAD_DIM) */
    float *f_v_proj;   /* HIDDEN x (FULL_KV_HEADS*FULL_HEAD_DIM) */
    float *f_o_proj;   /* (FULL_HEADS*FULL_HEAD_DIM) x HIDDEN */
    /* FFN weights (shared for all layers for benchmark simplicity) */
    float *gate_proj;  /* HIDDEN x INTER */
    float *up_proj;    /* HIDDEN x INTER */
    float *down_proj;  /* INTER x HIDDEN */
    /* Embedding / LM head */
    float *embed;      /* VOCAB x HIDDEN (tied) */
} Weights;

static void gdn_layer_forward(const Weights *w, LayerState *st, const float *hidden_in,
                              float *hidden_out, size_t seq) {
    float *q = alloc_aligned(seq * KEY_DIM);
    float *k = alloc_aligned(seq * KEY_DIM);
    float *v = alloc_aligned(seq * VALUE_DIM);
    float *conv_in = alloc_aligned(seq * CONV_DIM);
    float *conv_out = alloc_aligned(seq * CONV_DIM);
    float *beta = alloc_aligned(seq * KEY_DIM);
    float *decay = alloc_aligned(seq * KEY_DIM);
    float *scanned = alloc_aligned(seq * VALUE_DIM);
    float *attn_out = alloc_aligned(seq * HIDDEN);

    double t0;

    /* Q/K/V projections */
    t0 = now_us();
    matmul(hidden_in, w->g_q_proj, q, seq, HIDDEN, KEY_DIM);
    matmul(hidden_in, w->g_k_proj, k, seq, HIDDEN, KEY_DIM);
    matmul(hidden_in, w->g_v_proj, v, seq, HIDDEN, VALUE_DIM);
    g_t_gdn_proj += now_us() - t0;

    /* Project Q to conv_dim for depthwise conv */
    /* (Simplified: conv operates on CONV_DIM channels, we use first CONV_DIM of Q padded) */
    t0 = now_us();
    for (size_t t = 0; t < seq; ++t) {
        size_t copy = KEY_DIM < CONV_DIM ? KEY_DIM : CONV_DIM;
        memcpy(conv_in + t * CONV_DIM, q + t * KEY_DIM, copy * sizeof(float));
    }

    /* Causal depthwise Conv1D */
    gdn_causal_dwconv1d_f32(conv_in, w->g_conv_w, conv_out, st->conv_hist, seq, CONV_DIM);

    /* Copy conv output back to Q (first KEY_DIM channels) */
    for (size_t t = 0; t < seq; ++t)
        memcpy(q + t * KEY_DIM, conv_out + t * CONV_DIM, KEY_DIM * sizeof(float));
    g_t_gdn_conv += now_us() - t0;

    /* Beta = SiLU(conv_out[:KEY_DIM]) * g_beta — simplified to just beta projection */
    for (size_t t = 0; t < seq; ++t)
        for (size_t c = 0; c < KEY_DIM; ++c)
            beta[t * KEY_DIM + c] = w->g_beta[c];

    /* Cumulative decay: decay[t] = prod_{i<=t} beta[i] */
    t0 = now_us();
    gdn_cumdecay_f32(beta, decay, seq, KEY_DIM);
    g_t_gdn_decay += now_us() - t0;

    /* Delta-rule update: S = S + (v - S*k^T) * decay * k
     * Simplified to gated_scan with decay as gate */
    t0 = now_us();
    gdn_gated_scan_f32(decay, v, scanned, st->gdn_state, seq, VALUE_DIM);
    g_t_gdn_scan += now_us() - t0;

    /* Output projection */
    t0 = now_us();
    matmul(scanned, w->g_o_proj, attn_out, seq, VALUE_DIM, HIDDEN);
    g_t_gdn_oproj += now_us() - t0;

    /* Residual */
    for (size_t i = 0; i < seq * HIDDEN; ++i)
        hidden_out[i] = hidden_in[i] + attn_out[i];

    free(q); free(k); free(v); free(conv_in); free(conv_out);
    free(beta); free(decay); free(scanned); free(attn_out);
}

static void full_attn_layer_forward(const Weights *w, LayerState *st,
                                    const float *hidden_in, float *hidden_out, size_t seq) {
    size_t q_dim = FULL_HEADS * FULL_HEAD_DIM;     /* 4096 */
    size_t kv_dim = FULL_KV_HEADS * FULL_HEAD_DIM;  /* 1024 */

    float *q = alloc_aligned(seq * q_dim);
    float *k = alloc_aligned(seq * kv_dim);
    float *v = alloc_aligned(seq * kv_dim);
    float *attn_out = alloc_aligned(seq * HIDDEN);

    matmul(hidden_in, w->f_q_proj, q, seq, HIDDEN, q_dim);
    matmul(hidden_in, w->f_k_proj, k, seq, HIDDEN, kv_dim);
    matmul(hidden_in, w->f_v_proj, v, seq, HIDDEN, kv_dim);

    /* Simulated attention (identity for benchmark — KV cache growth is the point,
     * measured separately by the memory instrumentation harness) */
    /* Update KV cache (single token) */
    if (seq == 1) {
        memcpy(st->kv_cache_k, k, kv_dim * sizeof(float));
        memcpy(st->kv_cache_v, v, kv_dim * sizeof(float));
    }

    /* Output projection */
    matmul(q, w->f_o_proj, attn_out, seq, q_dim, HIDDEN);

    for (size_t i = 0; i < seq * HIDDEN; ++i)
        hidden_out[i] = hidden_in[i] + attn_out[i];

    free(q); free(k); free(v); free(attn_out);
}

static void ffn_forward(const Weights *w, const float *hidden_in, float *hidden_out, size_t seq) {
    float *gate = alloc_aligned(seq * INTER);
    float *up = alloc_aligned(seq * INTER);
    float *act = alloc_aligned(seq * INTER);
    float *down = alloc_aligned(seq * HIDDEN);

    matmul(hidden_in, w->gate_proj, gate, seq, HIDDEN, INTER);
    matmul(hidden_in, w->up_proj, up, seq, HIDDEN, INTER);

    /* SiLU(gate) * up */
    for (size_t i = 0; i < seq * INTER; ++i) {
        float g = gate[i];
        act[i] = g / (1.0f + expf(-g)) * up[i];
    }

    matmul(act, w->down_proj, down, seq, INTER, HIDDEN);

    for (size_t i = 0; i < seq * HIDDEN; ++i)
        hidden_out[i] = hidden_in[i] + down[i];

    free(gate); free(up); free(act); free(down);
}

int main(int argc, char **argv) {
    int num_tokens = 8;
    int csv = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc)
            num_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "--csv") == 0)
            csv = 1;
        else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [--tokens N] [--csv]\n", argv[0]);
            return 0;
        }
    }

    unsigned seed = 12345;

    /* Allocate weights (random, benchmark only) */
    Weights w;
    w.g_q_proj  = alloc_aligned(HIDDEN * KEY_DIM);
    w.g_k_proj  = alloc_aligned(HIDDEN * KEY_DIM);
    w.g_v_proj  = alloc_aligned(HIDDEN * VALUE_DIM);
    w.g_o_proj  = alloc_aligned(VALUE_DIM * HIDDEN);
    w.g_conv_w  = alloc_aligned(CONV_DIM * CONV_K);
    w.g_beta    = alloc_aligned(KEY_DIM);
    w.f_q_proj  = alloc_aligned(HIDDEN * FULL_HEADS * FULL_HEAD_DIM);
    w.f_k_proj  = alloc_aligned(HIDDEN * FULL_KV_HEADS * FULL_HEAD_DIM);
    w.f_v_proj  = alloc_aligned(HIDDEN * FULL_KV_HEADS * FULL_HEAD_DIM);
    w.f_o_proj  = alloc_aligned(FULL_HEADS * FULL_HEAD_DIM * HIDDEN);
    w.gate_proj = alloc_aligned(HIDDEN * INTER);
    w.up_proj   = alloc_aligned(HIDDEN * INTER);
    w.down_proj = alloc_aligned(INTER * HIDDEN);
    /* Embedding is huge (VOCAB*HIDDEN = ~638M floats = 2.5GB) — skip alloc,
     * use a dummy lookup for the benchmark. The LM head matmul is not the
     * bottleneck we're measuring; decode kernels and projections are. */

    fill_rand(w.g_q_proj,  HIDDEN * KEY_DIM, &seed);
    fill_rand(w.g_k_proj,  HIDDEN * KEY_DIM, &seed);
    fill_rand(w.g_v_proj,  HIDDEN * VALUE_DIM, &seed);
    fill_rand(w.g_o_proj,  VALUE_DIM * HIDDEN, &seed);
    fill_rand(w.g_conv_w,  CONV_DIM * CONV_K, &seed);
    fill_rand(w.g_beta,    KEY_DIM, &seed);
    fill_rand(w.f_q_proj,  HIDDEN * FULL_HEADS * FULL_HEAD_DIM, &seed);
    fill_rand(w.f_k_proj,  HIDDEN * FULL_KV_HEADS * FULL_HEAD_DIM, &seed);
    fill_rand(w.f_v_proj,  HIDDEN * FULL_KV_HEADS * FULL_HEAD_DIM, &seed);
    fill_rand(w.f_o_proj,  FULL_HEADS * FULL_HEAD_DIM * HIDDEN, &seed);
    fill_rand(w.gate_proj, HIDDEN * INTER, &seed);
    fill_rand(w.up_proj,   HIDDEN * INTER, &seed);
    fill_rand(w.down_proj, INTER * HIDDEN, &seed);

    /* Per-layer state */
    LayerState states[NUM_LAYERS];
    for (int l = 0; l < NUM_LAYERS; ++l) {
        states[l].gdn_state  = alloc_aligned(VALUE_DIM);
        states[l].conv_hist  = alloc_aligned((CONV_K - 1) * CONV_DIM);
        size_t kv_dim = FULL_KV_HEADS * FULL_HEAD_DIM;
        states[l].kv_cache_k = alloc_aligned(kv_dim);
        states[l].kv_cache_v = alloc_aligned(kv_dim);
    }

    /* Hidden state buffers */
    float *hidden = alloc_aligned(HIDDEN);
    float *hidden_next = alloc_aligned(HIDDEN);

    /* Determine layer types: NUM_LAYERS/4 blocks of (3 GDN, 1 full) */
    int is_gdn[NUM_LAYERS];
    for (int b = 0; b < NUM_LAYERS / 4; ++b)
        for (int s = 0; s < 4; ++s)
            is_gdn[b * 4 + s] = (s < 3) ? 1 : 0;

    if (!csv) {
        printf(MODEL_NAME " CPU decode benchmark\n");
        printf("  layers=%d (GDN=%d, full=%d), hidden=%d\n", NUM_LAYERS, NUM_GDN, NUM_FULL, HIDDEN);
        printf("  tokens=%d\n\n", num_tokens);
    }

    /* Warmup (3 tokens) */
    for (int t = 0; t < 3; ++t) {
        fill_rand(hidden, HIDDEN, &seed);
        for (int l = 0; l < NUM_LAYERS; ++l) {
            if (is_gdn[l])
                gdn_layer_forward(&w, &states[l], hidden, hidden_next, 1);
            else
                full_attn_layer_forward(&w, &states[l], hidden, hidden_next, 1);
            ffn_forward(&w, hidden_next, hidden, 1);
            float *tmp = hidden; hidden = hidden_next; hidden_next = tmp;
        }
    }

    /* Reset state for measured run */
    for (int l = 0; l < NUM_LAYERS; ++l) {
        memset(states[l].gdn_state, 0, VALUE_DIM * sizeof(float));
        memset(states[l].conv_hist, 0, (CONV_K - 1) * CONV_DIM * sizeof(float));
    }
    /* Warmup also touched the phase timers — zero them so the breakdown only
     * reflects the measured tokens below. */
    g_t_gdn_proj = g_t_gdn_conv = g_t_gdn_decay = g_t_gdn_scan = g_t_gdn_oproj = 0;
    g_t_full = g_t_ffn = 0;

    /* Measured decode loop */
    double *tok_times = malloc(num_tokens * sizeof(double));
    double t_start_all = now_us();

    for (int t = 0; t < num_tokens; ++t) {
        fill_rand(hidden, HIDDEN, &seed);
        double tok_start = now_us();

        for (int l = 0; l < NUM_LAYERS; ++l) {
            if (is_gdn[l]) {
                gdn_layer_forward(&w, &states[l], hidden, hidden_next, 1);
            } else {
                double t0 = now_us();
                full_attn_layer_forward(&w, &states[l], hidden, hidden_next, 1);
                g_t_full += now_us() - t0;
            }
            double t0 = now_us();
            ffn_forward(&w, hidden_next, hidden, 1);
            g_t_ffn += now_us() - t0;
            float *tmp = hidden; hidden = hidden_next; hidden_next = tmp;
        }

        tok_times[t] = now_us() - tok_start;
    }

    double total_us = now_us() - t_start_all;

    /* Stats */
    double ttft_us = tok_times[0];
    double sum = 0; for (int t = 0; t < num_tokens; ++t) sum += tok_times[t];
    double mean_us = sum / num_tokens;
    /* Sort for percentiles */
    for (int i = 0; i < num_tokens - 1; ++i)
        for (int j = i + 1; j < num_tokens; ++j)
            if (tok_times[j] < tok_times[i]) {
                double tmp = tok_times[i]; tok_times[i] = tok_times[j]; tok_times[j] = tmp;
            }
    double p50 = tok_times[num_tokens / 2];
    double p95 = tok_times[(int)(num_tokens * 0.95)];
    double p99 = tok_times[(int)(num_tokens * 0.99)];
    double tok_per_sec = 1e6 / mean_us;
    double ttft_ms = ttft_us / 1e3;

    /* Bottleneck breakdown: phase totals as a fraction of summed measured time.
     * Note this sums to slightly more than the wall-clock decode time on some
     * boards because of scheduling noise between now_us() calls — treat as
     * relative proportions, not an exact partition. */
    double phase_sum = g_t_gdn_proj + g_t_gdn_conv + g_t_gdn_decay + g_t_gdn_scan
                      + g_t_gdn_oproj + g_t_full + g_t_ffn;

    if (csv) {
        printf("model,tokens,ttft_ms,tok_per_sec_mean,p50_us,p95_us,p99_us,mean_us,"
               "gdn_proj_pct,gdn_conv_pct,gdn_decay_pct,gdn_scan_pct,gdn_oproj_pct,full_pct,ffn_pct\n");
        printf(MODEL_NAME ",%d,%.2f,%.2f,%.0f,%.0f,%.0f,%.0f,"
               "%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f\n",
               num_tokens, ttft_ms, tok_per_sec, p50, p95, p99, mean_us,
               100.0 * g_t_gdn_proj / phase_sum, 100.0 * g_t_gdn_conv / phase_sum,
               100.0 * g_t_gdn_decay / phase_sum, 100.0 * g_t_gdn_scan / phase_sum,
               100.0 * g_t_gdn_oproj / phase_sum, 100.0 * g_t_full / phase_sum,
               100.0 * g_t_ffn / phase_sum);
    } else {
        printf("Results (%d tokens):\n", num_tokens);
        printf("  TTFT (first token):  %.2f ms\n", ttft_ms);
        printf("  Tokens/sec (mean):   %.2f\n", tok_per_sec);
        printf("  Per-token latency:\n");
        printf("    p50:  %.0f us  (%.2f ms)\n", p50, p50 / 1e3);
        printf("    p95:  %.0f us  (%.2f ms)\n", p95, p95 / 1e3);
        printf("    p99:  %.0f us  (%.2f ms)\n", p99, p99 / 1e3);
        printf("    mean: %.0f us  (%.2f ms)\n", mean_us, mean_us / 1e3);
        printf("  Total wall time:     %.2f s\n", total_us / 1e6);
        printf("  Bottleneck breakdown (share of summed measured phase time):\n");
        printf("    GDN q/k/v proj:  %5.1f%%  (%.0f us/tok)\n",
               100.0 * g_t_gdn_proj / phase_sum, g_t_gdn_proj / num_tokens);
        printf("    GDN conv1d:      %5.1f%%  (%.0f us/tok)\n",
               100.0 * g_t_gdn_conv / phase_sum, g_t_gdn_conv / num_tokens);
        printf("    GDN cumdecay:    %5.1f%%  (%.0f us/tok)\n",
               100.0 * g_t_gdn_decay / phase_sum, g_t_gdn_decay / num_tokens);
        printf("    GDN gated_scan:  %5.1f%%  (%.0f us/tok)\n",
               100.0 * g_t_gdn_scan / phase_sum, g_t_gdn_scan / num_tokens);
        printf("    GDN out proj:    %5.1f%%  (%.0f us/tok)\n",
               100.0 * g_t_gdn_oproj / phase_sum, g_t_gdn_oproj / num_tokens);
        printf("    Full-attention:  %5.1f%%  (%.0f us/tok)\n",
               100.0 * g_t_full / phase_sum, g_t_full / num_tokens);
        printf("    FFN:             %5.1f%%  (%.0f us/tok)\n",
               100.0 * g_t_ffn / phase_sum, g_t_ffn / num_tokens);
    }

    /* Cleanup */
    free(tok_times); free(hidden); free(hidden_next);
    free(w.g_q_proj); free(w.g_k_proj); free(w.g_v_proj); free(w.g_o_proj);
    free(w.g_conv_w); free(w.g_beta);
    free(w.f_q_proj); free(w.f_k_proj); free(w.f_v_proj); free(w.f_o_proj);
    free(w.gate_proj); free(w.up_proj); free(w.down_proj);
    for (int l = 0; l < NUM_LAYERS; ++l) {
        free(states[l].gdn_state); free(states[l].conv_hist);
        free(states[l].kv_cache_k); free(states[l].kv_cache_v);
    }
    return 0;
}
