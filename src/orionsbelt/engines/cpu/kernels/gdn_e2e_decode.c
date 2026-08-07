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
#define _POSIX_C_SOURCE 200112L  /* clock_gettime, posix_memalign, rand_r */

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

/* ---- INT8 weight-only quantization ----
 *
 * Stores weights as int8 + per-output-column float scale. Dequantizes
 * on-the-fly in the GEMV inner loop. Cuts weight memory traffic 4× vs FP32,
 * directly improving tok/s since decode is bandwidth-bound.
 *
 * Compiled in with -DINT8_WEIGHTS. The FP32 path is unchanged for comparison.
 */
typedef struct {
    int8_t *q;   /* quantized weights [K×N] row-major int8 */
    float  *s;   /* per-column (output) scale [N] */
} QW;

/* Quantize a [K×N] float matrix to int8 + per-column scale (symmetric).
 * B_in is read-only; q and s are allocated and filled.
 * Only needed under -DINT8_WEIGHTS. */
#ifdef INT8_WEIGHTS
static void quantize_weight(const float *B_in, int8_t **q_out, float **s_out,
                            size_t K, size_t N) {
    int8_t *q = malloc(K * N);
    float  *s = malloc(N * sizeof(float));
    if (!q || !s) { fprintf(stderr, "OOM in quantize\n"); exit(1); }

    for (size_t n = 0; n < N; ++n) {
        /* Find max abs in column n */
        float max_abs = 0.0f;
        for (size_t k = 0; k < K; ++k) {
            float v = fabsf(B_in[k * N + n]);
            if (v > max_abs) max_abs = v;
        }
        s[n] = (max_abs > 0.0f) ? max_abs / 127.0f : 1.0f;
        float inv = 1.0f / s[n];
        for (size_t k = 0; k < K; ++k) {
            float scaled = B_in[k * N + n] * inv;
            int vi = (int)lroundf(scaled);
            if (vi > 127) vi = 127;
            if (vi < -128) vi = -128;
            q[k * N + n] = (int8_t)vi;
        }
    }
    *q_out = q;
    *s_out = s;
}
#endif /* INT8_WEIGHTS */

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
    /* a is [K], B is [K×N] row-major, c is [N].
     *
     * Row-sweep GEMV: K-outer, N-inner. Each row of B is accessed
     * sequentially → ~100% cache-line utilization (vs 0.17% in the old
     * column-sweep version that strides by N between consecutive K).
     * Parallelized over N-tiles via OpenMP so all big cores participate.
     *
     * For N=9216 (4B FFN), the old version wasted 576× the memory bandwidth
     * because it touched one float per 36 KB cache-line stride; this version
     * touches every float in each contiguous row segment. */
    const size_t TILE = 1024;  /* 4 KB output tile, stays in L1 */

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t jt = 0; jt < N; jt += TILE) {
        size_t tn = (N - jt >= TILE) ? TILE : (N - jt);
        float *ct = c + jt;

        /* Zero the output tile */
        memset(ct, 0, tn * sizeof(float));

        /* Sweep K — B row segments accessed sequentially */
        for (size_t k = 0; k < K; ++k) {
            float ak = a[k];
            const float *Brow = B + k * N + jt;
            size_t j = 0;
#ifdef __ARM_NEON
            float32x4_t akv = vdupq_n_f32(ak);
            for (; j + 4 <= tn; j += 4) {
                float32x4_t bk = vld1q_f32(Brow + j);
                float32x4_t ck = vld1q_f32(ct + j);
                ck = vfmaq_f32(ck, akv, bk);
                vst1q_f32(ct + j, ck);
            }
#endif
            for (; j < tn; ++j)
                ct[j] += ak * Brow[j];
        }
    }
}

/* INT8 GEMV: dequantize-on-the-fly with NEON.
 * a is [K] float, Bq is [K×N] int8, Bs is [N] float scale, c is [N] float.
 *
 * Row-sweep with per-column scale factored out:
 *   c[n] = Bs[n] * sum_k(a[k] * (float)Bq[k*N+n])
 *
 * This loads 1 byte per weight element (vs 4 for FP32), cutting memory
 * traffic ~4×. NEON dequantization (int8→int16→int32→float32) adds ~3
 * cycles per 4 elements but is hidden behind memory latency.
 * Only needed under -DINT8_WEIGHTS (called from matmul_int8). */
#ifdef INT8_WEIGHTS
static void gemv_int8_neon(const float *a, const int8_t *Bq, const float *Bs,
                           float *c, size_t K, size_t N) {
    const size_t TILE = 1024;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t jt = 0; jt < N; jt += TILE) {
        size_t tn = (N - jt >= TILE) ? TILE : (N - jt);
        float *ct = c + jt;

        memset(ct, 0, tn * sizeof(float));

        for (size_t k = 0; k < K; ++k) {
            float ak = a[k];
            const int8_t *Brow = Bq + k * N + jt;
            size_t j = 0;
#ifdef __ARM_NEON
            float32x4_t akv = vdupq_n_f32(ak);
            for (; j + 8 <= tn; j += 8) {
                /* Load 8 int8 → widen to 8 int16 → split to 2× int32x4 → 2× float32x4 */
                int8x8_t   i8  = vld1_s8(Brow + j);
                int16x8_t  i16 = vmovl_s8(i8);
                int32x4_t  i32lo = vmovl_s16(vget_low_s16(i16));
                int32x4_t  i32hi = vmovl_s16(vget_high_s16(i16));
                float32x4_t flo = vcvtq_f32_s32(i32lo);
                float32x4_t fhi = vcvtq_f32_s32(i32hi);

                float32x4_t clo = vld1q_f32(ct + j);
                float32x4_t chi = vld1q_f32(ct + j + 4);
                clo = vfmaq_f32(clo, akv, flo);
                chi = vfmaq_f32(chi, akv, fhi);
                vst1q_f32(ct + j, clo);
                vst1q_f32(ct + j + 4, chi);
            }
            for (; j + 4 <= tn; j += 4) {
                int8x8_t   i8  = vld1_s8(Brow + j);
                int16x8_t  i16 = vmovl_s8(i8);
                int32x4_t  i32 = vmovl_s16(vget_low_s16(i16));
                float32x4_t bq = vcvtq_f32_s32(i32);

                float32x4_t ck = vld1q_f32(ct + j);
                ck = vfmaq_f32(ck, akv, bq);
                vst1q_f32(ct + j, ck);
            }
#endif
            for (; j < tn; ++j)
                ct[j] += ak * (float)Brow[j];
        }

        /* Apply per-column scale (NEON-vectorized) */
        size_t j = 0;
#ifdef __ARM_NEON
        for (; j + 4 <= tn; j += 4) {
            float32x4_t cv = vld1q_f32(ct + j);
            float32x4_t sv = vld1q_f32(Bs + jt + j);
            cv = vmulq_f32(cv, sv);
            vst1q_f32(ct + j, cv);
        }
#endif
        for (; j < tn; ++j)
            ct[j] *= Bs[jt + j];
    }
}
#endif /* INT8_WEIGHTS */

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

/* INT8 matmul wrapper for M=1 decode path (falls back to scalar for M>1)
 * Only needed under -DINT8_WEIGHTS. */
#ifdef INT8_WEIGHTS
static void matmul_int8(const float *A, const int8_t *Bq, const float *Bs,
                        float *C, size_t M, size_t K, size_t N) {
    if (M == 1) {
        gemv_int8_neon(A, Bq, Bs, C, K, N);
        return;
    }
    /* Prefill path: dequantize on the fly (rare in decode benchmark) */
    float *Bf = malloc(K * N * sizeof(float));
    for (size_t k = 0; k < K; ++k)
        for (size_t n = 0; n < N; ++n)
            Bf[k * N + n] = (float)Bq[k * N + n] * Bs[n];
    matmul(A, Bf, C, M, K, N);
    free(Bf);
}
#endif /* INT8_WEIGHTS */

/* Dispatch macro: uses INT8 GEMV when compiled with -DINT8_WEIGHTS,
 * otherwise falls through to the FP32 path. Bf is the float weight,
 * Bq is the QW (int8+scale), only one is used per compilation. */
#ifdef INT8_WEIGHTS
#define MM(A, Bf, Bq, C, M, K, N) matmul_int8(A, (Bq).q, (Bq).s, C, M, K, N)
#else
#define MM(A, Bf, Bq, C, M, K, N) matmul(A, Bf, C, M, K, N)
#endif

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
    /* Growing KV cache for ctx-sweep mode (allocated on demand) */
    float *kv_k_grow;
    float *kv_v_grow;
    /* INT8 KV cache: 4× less memory traffic per attention element.
     * Per-head symmetric scale (FULL_KV_HEADS entries per K and V). */
    int8_t *kv_k_grow_q;   /* [max_ctx * kv_dim] int8 */
    int8_t *kv_v_grow_q;   /* [max_ctx * kv_dim] int8 */
    float  *kv_k_scale;    /* [FULL_KV_HEADS] per-head scale */
    float  *kv_v_scale;    /* [FULL_KV_HEADS] per-head scale */
    size_t kv_pos;        /* current position (tokens in cache) */
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
    /* INT8 quantized versions (populated when INT8_WEIGHTS is defined) */
    QW g_q_proj_q, g_k_proj_q, g_v_proj_q, g_o_proj_q;
    QW f_q_proj_q, f_k_proj_q, f_v_proj_q, f_o_proj_q;
    QW gate_proj_q, up_proj_q, down_proj_q;
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
    MM(hidden_in, w->g_q_proj, w->g_q_proj_q, q, seq, HIDDEN, KEY_DIM);
    MM(hidden_in, w->g_k_proj, w->g_k_proj_q, k, seq, HIDDEN, KEY_DIM);
    MM(hidden_in, w->g_v_proj, w->g_v_proj_q, v, seq, HIDDEN, VALUE_DIM);
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
    MM(scanned, w->g_o_proj, w->g_o_proj_q, attn_out, seq, VALUE_DIM, HIDDEN);
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

    MM(hidden_in, w->f_q_proj, w->f_q_proj_q, q, seq, HIDDEN, q_dim);
    MM(hidden_in, w->f_k_proj, w->f_k_proj_q, k, seq, HIDDEN, kv_dim);
    MM(hidden_in, w->f_v_proj, w->f_v_proj_q, v, seq, HIDDEN, kv_dim);

    /* Simulated attention (identity for benchmark — KV cache growth is the point,
     * measured separately by the memory instrumentation harness) */
    /* Update KV cache (single token) */
    if (seq == 1) {
        memcpy(st->kv_cache_k, k, kv_dim * sizeof(float));
        memcpy(st->kv_cache_v, v, kv_dim * sizeof(float));
    }

    /* Output projection */
    MM(q, w->f_o_proj, w->f_o_proj_q, attn_out, seq, q_dim, HIDDEN);

    for (size_t i = 0; i < seq * HIDDEN; ++i)
        hidden_out[i] = hidden_in[i] + attn_out[i];

    free(q); free(k); free(v); free(attn_out);
}

/* ---- Real multi-head attention with growing KV cache (ctx-sweep mode) ----
 *
 * Implements proper GQA attention: Q * K_cache^T -> softmax -> weighted V.
 * Used by --ctx-sweep to show full-attention's O(n) scaling vs GDN's O(1).
 * KV cache grows by one token per call; pre-filled with random data to
 * simulate prior prefill tokens.
 * Only compiled when KV_INT8 is NOT defined (see call site #ifdef).
 */
#ifndef KV_INT8
static void full_attn_real_forward(const Weights *w, LayerState *st,
                                   const float *hidden_in, float *hidden_out) {
    size_t q_dim = FULL_HEADS * FULL_HEAD_DIM;
    size_t kv_dim = FULL_KV_HEADS * FULL_HEAD_DIM;
    size_t pos = st->kv_pos;
    size_t ctx = pos + 1;  /* tokens to attend to (including this one) */

    float *q = alloc_aligned(q_dim);
    float *k = alloc_aligned(kv_dim);
    float *v = alloc_aligned(kv_dim);
    float *attn_out = alloc_aligned(q_dim);
    float *scores = alloc_aligned(ctx);

    /* Q/K/V projections (same as simulated mode) */
    MM(hidden_in, w->f_q_proj, w->f_q_proj_q, q, 1, HIDDEN, q_dim);
    MM(hidden_in, w->f_k_proj, w->f_k_proj_q, k, 1, HIDDEN, kv_dim);
    MM(hidden_in, w->f_v_proj, w->f_v_proj_q, v, 1, HIDDEN, kv_dim);

    /* Append new K/V to cache */
    memcpy(st->kv_k_grow + pos * kv_dim, k, kv_dim * sizeof(float));
    memcpy(st->kv_v_grow + pos * kv_dim, v, kv_dim * sizeof(float));

    /* GQA attention: each Q head attends to its KV head group */
    const size_t groups = FULL_HEADS / FULL_KV_HEADS;
    const float scale = 1.0f / sqrtf((float)FULL_HEAD_DIM);

    for (size_t h = 0; h < FULL_HEADS; ++h) {
        size_t kh = h / groups;
        const float *qh = q + h * FULL_HEAD_DIM;

        /* Score = Q[h] . K_cache[t, kh] * scale */
        float max_s = -INFINITY;
        for (size_t t = 0; t < ctx; ++t) {
            const float *kt = st->kv_k_grow + t * kv_dim + kh * FULL_HEAD_DIM;
            float s = 0.0f;
            for (size_t d = 0; d < FULL_HEAD_DIM; ++d)
                s += qh[d] * kt[d];
            s *= scale;
            scores[t] = s;
            if (s > max_s) max_s = s;
        }

        /* Softmax */
        float sum_exp = 0.0f;
        for (size_t t = 0; t < ctx; ++t) {
            scores[t] = expf(scores[t] - max_s);
            sum_exp += scores[t];
        }

        /* Weighted sum of V */
        float *out_h = attn_out + h * FULL_HEAD_DIM;
        memset(out_h, 0, FULL_HEAD_DIM * sizeof(float));
        for (size_t t = 0; t < ctx; ++t) {
            const float *vt = st->kv_v_grow + t * kv_dim + kh * FULL_HEAD_DIM;
            float weight = scores[t] / sum_exp;
            for (size_t d = 0; d < FULL_HEAD_DIM; ++d)
                out_h[d] += weight * vt[d];
        }
    }

    /* Output projection + residual */
    MM(attn_out, w->f_o_proj, w->f_o_proj_q, hidden_out, 1, q_dim, HIDDEN);
    for (size_t i = 0; i < HIDDEN; ++i)
        hidden_out[i] += hidden_in[i];

    /* Advance position */
    st->kv_pos = pos + 1;

    free(q); free(k); free(v); free(attn_out); free(scores);
}
#endif /* !KV_INT8 */

/* ---- INT8 KV cache: real attention with quantized KV (ctx-sweep mode) ----
 *
 * Same GQA attention as full_attn_real_forward, but stores the growing KV
 * cache as int8 with per-head symmetric scale. This cuts KV memory traffic
 * 4× (1 byte per element vs 4), directly improving full-attention's O(n)
 * decode cost at long context.
 *
 * Optimization: per-head scale is factored out of the inner loops.
 *   score = scale_k[kh] * sum_d(q[d] * (float)kq[d])   ← scale applied once
 *   out  += scale_v[kh] * weight * (float)vq[d]         ← scale folded into weight
 *
 * This avoids per-element scaling in the hot loop.
 * Only compiled under -DKV_INT8; guarding prevents -Wunused-function.
 */
#ifdef KV_INT8
static void full_attn_real_forward_int8kv(const Weights *w, LayerState *st,
                                          const float *hidden_in, float *hidden_out) {
    size_t q_dim = FULL_HEADS * FULL_HEAD_DIM;
    size_t kv_dim = FULL_KV_HEADS * FULL_HEAD_DIM;
    size_t pos = st->kv_pos;
    size_t ctx = pos + 1;

    float *q = alloc_aligned(q_dim);
    float *k = alloc_aligned(kv_dim);
    float *v = alloc_aligned(kv_dim);
    float *attn_out = alloc_aligned(q_dim);
    float *scores = alloc_aligned(ctx);

    MM(hidden_in, w->f_q_proj, w->f_q_proj_q, q, 1, HIDDEN, q_dim);
    MM(hidden_in, w->f_k_proj, w->f_k_proj_q, k, 1, HIDDEN, kv_dim);
    MM(hidden_in, w->f_v_proj, w->f_v_proj_q, v, 1, HIDDEN, kv_dim);

    /* Quantize new K/V token to int8, append to cache.
     * Clamp to [-128, 127] — the per-head scale was computed from pre-fill
     * data, so a new token whose K/V exceeds the pre-fill max would overflow
     * int8 without clamping (implementation-defined behavior → garbage). */
    for (size_t kh = 0; kh < FULL_KV_HEADS; ++kh) {
        float inv_ks = 1.0f / st->kv_k_scale[kh];
        float inv_vs = 1.0f / st->kv_v_scale[kh];
        for (size_t d = 0; d < FULL_HEAD_DIM; ++d) {
            int kq = (int)lroundf(k[kh * FULL_HEAD_DIM + d] * inv_ks);
            if (kq > 127) kq = 127;
            if (kq < -128) kq = -128;
            st->kv_k_grow_q[pos * kv_dim + kh * FULL_HEAD_DIM + d] = (int8_t)kq;

            int vq = (int)lroundf(v[kh * FULL_HEAD_DIM + d] * inv_vs);
            if (vq > 127) vq = 127;
            if (vq < -128) vq = -128;
            st->kv_v_grow_q[pos * kv_dim + kh * FULL_HEAD_DIM + d] = (int8_t)vq;
        }
    }

    /* GQA attention with INT8 KV cache */
    const size_t groups = FULL_HEADS / FULL_KV_HEADS;
    const float scale = 1.0f / sqrtf((float)FULL_HEAD_DIM);

    for (size_t h = 0; h < FULL_HEADS; ++h) {
        size_t kh = h / groups;
        const float *qh = q + h * FULL_HEAD_DIM;
        const float sk = st->kv_k_scale[kh];  /* per-head K scale */

        /* Score = scale_k[kh] * sum_d(Q[d] * (float)Kq[t,d]) * attention_scale
         * Factor sk out of the inner sum, multiply once per token. */
        float max_s = -INFINITY;
        for (size_t t = 0; t < ctx; ++t) {
            const int8_t *kt = st->kv_k_grow_q + t * kv_dim + kh * FULL_HEAD_DIM;
            float s = 0.0f;
            size_t d = 0;
#ifdef __ARM_NEON
            /* Dequantize K to float on-the-fly, dot with Q via FMA.
             * No dotprod on A76 — same int8→int16→int32→float chain
             * as the INT8 GEMV weight path. */
            float32x4_t acc4 = vdupq_n_f32(0.0f);
            for (; d + 8 <= FULL_HEAD_DIM; d += 8) {
                int8x8_t   i8  = vld1_s8((const int8_t*)(kt + d));
                int16x8_t  i16 = vmovl_s8(i8);
                int32x4_t  i32lo = vmovl_s16(vget_low_s16(i16));
                int32x4_t  i32hi = vmovl_s16(vget_high_s16(i16));
                float32x4_t flo = vcvtq_f32_s32(i32lo);
                float32x4_t fhi = vcvtq_f32_s32(i32hi);
                float32x4_t qlo = vld1q_f32(qh + d);
                float32x4_t qhi = vld1q_f32(qh + d + 4);
                acc4 = vfmaq_f32(acc4, qlo, flo);
                acc4 = vfmaq_f32(acc4, qhi, fhi);
            }
            for (; d + 4 <= FULL_HEAD_DIM; d += 4) {
                int8x8_t  i8  = vld1_s8((const int8_t*)(kt + d));
                int16x8_t i16 = vmovl_s8(i8);
                int32x4_t i32 = vmovl_s16(vget_low_s16(i16));
                float32x4_t fq = vcvtq_f32_s32(i32);
                float32x4_t q4 = vld1q_f32(qh + d);
                acc4 = vfmaq_f32(acc4, q4, fq);
            }
            s += vaddvq_f32(acc4);
            for (; d < FULL_HEAD_DIM; ++d)
                s += qh[d] * (float)kt[d];
#else
            for (size_t dd = 0; dd < FULL_HEAD_DIM; ++dd)
                s += qh[dd] * (float)kt[dd];
#endif
            s *= sk * scale;  /* fold in per-head scale + attention scale */
            scores[t] = s;
            if (s > max_s) max_s = s;
        }

        /* Softmax */
        float sum_exp = 0.0f;
        for (size_t t = 0; t < ctx; ++t) {
            scores[t] = expf(scores[t] - max_s);
            sum_exp += scores[t];
        }

        /* Weighted sum of V — fold sv into effective weight */
        float sv = st->kv_v_scale[kh];
        float *out_h = attn_out + h * FULL_HEAD_DIM;
        memset(out_h, 0, FULL_HEAD_DIM * sizeof(float));
        for (size_t t = 0; t < ctx; ++t) {
            const int8_t *vt = st->kv_v_grow_q + t * kv_dim + kh * FULL_HEAD_DIM;
            float ew = scores[t] / sum_exp * sv;  /* effective weight with V scale */
            size_t d = 0;
#ifdef __ARM_NEON
            float32x4_t ewv = vdupq_n_f32(ew);
            for (; d + 8 <= FULL_HEAD_DIM; d += 8) {
                int8x8_t   i8  = vld1_s8((const int8_t*)(vt + d));
                int16x8_t  i16 = vmovl_s8(i8);
                int32x4_t  i32lo = vmovl_s16(vget_low_s16(i16));
                int32x4_t  i32hi = vmovl_s16(vget_high_s16(i16));
                float32x4_t flo = vcvtq_f32_s32(i32lo);
                float32x4_t fhi = vcvtq_f32_s32(i32hi);
                float32x4_t olo = vld1q_f32(out_h + d);
                float32x4_t ohi = vld1q_f32(out_h + d + 4);
                olo = vfmaq_f32(olo, ewv, flo);
                ohi = vfmaq_f32(ohi, ewv, fhi);
                vst1q_f32(out_h + d, olo);
                vst1q_f32(out_h + d + 4, ohi);
            }
            for (; d + 4 <= FULL_HEAD_DIM; d += 4) {
                int8x8_t  i8  = vld1_s8((const int8_t*)(vt + d));
                int16x8_t i16 = vmovl_s8(i8);
                int32x4_t i32 = vmovl_s16(vget_low_s16(i16));
                float32x4_t fq = vcvtq_f32_s32(i32);
                float32x4_t oh = vld1q_f32(out_h + d);
                oh = vfmaq_f32(oh, ewv, fq);
                vst1q_f32(out_h + d, oh);
            }
#endif
            for (; d < FULL_HEAD_DIM; ++d)
                out_h[d] += ew * (float)vt[d];
        }
    }

    /* Output projection + residual */
    MM(attn_out, w->f_o_proj, w->f_o_proj_q, hidden_out, 1, q_dim, HIDDEN);
    for (size_t i = 0; i < HIDDEN; ++i)
        hidden_out[i] += hidden_in[i];

    st->kv_pos = pos + 1;

    free(q); free(k); free(v); free(attn_out); free(scores);
}
#endif /* KV_INT8 */

static void ffn_forward(const Weights *w, const float *hidden_in, float *hidden_out, size_t seq) {
    float *gate = alloc_aligned(seq * INTER);
    float *up = alloc_aligned(seq * INTER);
    float *act = alloc_aligned(seq * INTER);
    float *down = alloc_aligned(seq * HIDDEN);

    MM(hidden_in, w->gate_proj, w->gate_proj_q, gate, seq, HIDDEN, INTER);
    MM(hidden_in, w->up_proj, w->up_proj_q, up, seq, HIDDEN, INTER);

    /* SiLU(gate) * up */
    for (size_t i = 0; i < seq * INTER; ++i) {
        float g = gate[i];
        act[i] = g / (1.0f + expf(-g)) * up[i];
    }

    MM(act, w->down_proj, w->down_proj_q, down, seq, INTER, HIDDEN);

    for (size_t i = 0; i < seq * HIDDEN; ++i)
        hidden_out[i] = hidden_in[i] + down[i];

    free(gate); free(up); free(act); free(down);
}

/* ---- Cleanup helper: free all weight matrices (FP32 + optional INT8) ---- */
static void free_weights(Weights *w) {
    free(w->g_q_proj); free(w->g_k_proj); free(w->g_v_proj); free(w->g_o_proj);
    free(w->g_conv_w); free(w->g_beta);
    free(w->f_q_proj); free(w->f_k_proj); free(w->f_v_proj); free(w->f_o_proj);
    free(w->gate_proj); free(w->up_proj); free(w->down_proj);
#ifdef INT8_WEIGHTS
    free(w->g_q_proj_q.q);  free(w->g_q_proj_q.s);
    free(w->g_k_proj_q.q);  free(w->g_k_proj_q.s);
    free(w->g_v_proj_q.q);  free(w->g_v_proj_q.s);
    free(w->g_o_proj_q.q);  free(w->g_o_proj_q.s);
    free(w->f_q_proj_q.q);  free(w->f_q_proj_q.s);
    free(w->f_k_proj_q.q);  free(w->f_k_proj_q.s);
    free(w->f_v_proj_q.q);  free(w->f_v_proj_q.s);
    free(w->f_o_proj_q.q);  free(w->f_o_proj_q.s);
    free(w->gate_proj_q.q); free(w->gate_proj_q.s);
    free(w->up_proj_q.q);   free(w->up_proj_q.s);
    free(w->down_proj_q.q); free(w->down_proj_q.s);
#endif
}

/* ---- Cleanup helper: free per-layer base state (shared by both code paths) ---- */
static void free_layer_states(LayerState *states) {
    for (int l = 0; l < NUM_LAYERS; ++l) {
        free(states[l].gdn_state); free(states[l].conv_hist);
        free(states[l].kv_cache_k); free(states[l].kv_cache_v);
    }
}

int main(int argc, char **argv) {
    int num_tokens = 8;
    int csv = 0;
    int ctx_sweep = 0;
    int pure_gdn = 0;
    const char *ctx_lens_str = NULL;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc)
            num_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "--csv") == 0)
            csv = 1;
        else if (strcmp(argv[i], "--ctx-sweep") == 0) {
            ctx_sweep = 1;
            if (i + 1 < argc && argv[i + 1][0] != '-')
                ctx_lens_str = argv[++i];
        }
        else if (strcmp(argv[i], "--pure-gdn") == 0)
            pure_gdn = 1;
        else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [--tokens N] [--csv] [--ctx-sweep L1,L2,...] [--pure-gdn]\n", argv[0]);
            printf("  --ctx-sweep  Measure decode cost at growing context lengths\n");
            printf("               Comma-separated list, e.g. 1,64,256,1024,4096\n");
            printf("  --pure-gdn   All layers GDN (no full-attention) — shows ideal O(1) scaling\n");
            return 0;
        }
    }

    /* Validate token count — 0 or negative causes division by zero in
     * mean/percentile stats and array underflow in the sort loop. */
    if (num_tokens < 1) {
        fprintf(stderr, "Error: --tokens must be >= 1 (got %d)\n", num_tokens);
        return 1;
    }

    unsigned seed = 12345;

    /* Allocate weights (random, benchmark only) */
    Weights w;
    w.embed = NULL;  /* not allocated (VOCAB×HIDDEN = 2.5 GB) — NULL prevents dangling */
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

    /* INT8 quantize all GEMV weight matrices (when enabled) */
#ifdef INT8_WEIGHTS
    quantize_weight(w.g_q_proj,  &w.g_q_proj_q.q,  &w.g_q_proj_q.s,  HIDDEN, KEY_DIM);
    quantize_weight(w.g_k_proj,  &w.g_k_proj_q.q,  &w.g_k_proj_q.s,  HIDDEN, KEY_DIM);
    quantize_weight(w.g_v_proj,  &w.g_v_proj_q.q,  &w.g_v_proj_q.s,  HIDDEN, VALUE_DIM);
    quantize_weight(w.g_o_proj,  &w.g_o_proj_q.q,  &w.g_o_proj_q.s,  VALUE_DIM, HIDDEN);
    quantize_weight(w.f_q_proj,  &w.f_q_proj_q.q,  &w.f_q_proj_q.s,  HIDDEN, FULL_HEADS * FULL_HEAD_DIM);
    quantize_weight(w.f_k_proj,  &w.f_k_proj_q.q,  &w.f_k_proj_q.s,  HIDDEN, FULL_KV_HEADS * FULL_HEAD_DIM);
    quantize_weight(w.f_v_proj,  &w.f_v_proj_q.q,  &w.f_v_proj_q.s,  HIDDEN, FULL_KV_HEADS * FULL_HEAD_DIM);
    quantize_weight(w.f_o_proj,  &w.f_o_proj_q.q,  &w.f_o_proj_q.s,  FULL_HEADS * FULL_HEAD_DIM, HIDDEN);
    quantize_weight(w.gate_proj, &w.gate_proj_q.q, &w.gate_proj_q.s, HIDDEN, INTER);
    quantize_weight(w.up_proj,   &w.up_proj_q.q,   &w.up_proj_q.s,   HIDDEN, INTER);
    quantize_weight(w.down_proj, &w.down_proj_q.q, &w.down_proj_q.s, INTER, HIDDEN);
    if (!csv)
        printf("  Weight quantization: INT8 (weight-only, per-column symmetric scale)\n\n");
#endif

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

    /* --pure-gdn: override all layers to GDN (hypothetical pure-linear-attention model) */
    if (pure_gdn) {
        for (int l = 0; l < NUM_LAYERS; ++l)
            is_gdn[l] = 1;
    }

    /* ---- Context-length sweep mode ----
     *
     * For each context length C: pre-fill full-attention KV caches with C-1
     * random tokens, then decode 1 token at position C-1, measuring per-layer-
     * type cost. GDN layers have O(1) state so their cost is constant; full-
     * attention cost scales O(C). This is the headline comparison that shows
     * why GDN matters for long-context decode. */
    if (ctx_sweep) {
        /* Parse context lengths */
        size_t ctx_lens[64];
        int n_ctx = 0;
        if (ctx_lens_str) {
            char buf[512];
            strncpy(buf, ctx_lens_str, sizeof(buf) - 1);
            buf[sizeof(buf) - 1] = '\0';
            char *tok = strtok(buf, ",");
            while (tok && n_ctx < 64) {
                ctx_lens[n_ctx++] = (size_t)atol(tok);
                tok = strtok(NULL, ",");
            }
        } else {
            /* Default sweep */
            size_t defaults[] = {1, 64, 256, 1024, 4096};
            for (int i = 0; i < 5; ++i) ctx_lens[n_ctx++] = defaults[i];
        }

        size_t max_ctx = 0;
        for (int i = 0; i < n_ctx; ++i)
            if (ctx_lens[i] > max_ctx) max_ctx = ctx_lens[i];

        size_t kv_dim = FULL_KV_HEADS * FULL_HEAD_DIM;

        if (!csv) {
            printf(MODEL_NAME " context-length scaling benchmark\n");
            int n_gdn_actual = 0, n_full_actual = 0;
            for (int l = 0; l < NUM_LAYERS; ++l)
                if (is_gdn[l]) n_gdn_actual++; else n_full_actual++;
            printf("  layers=%d (GDN=%d, full=%d)%s, hidden=%d\n", NUM_LAYERS,
                   n_gdn_actual, n_full_actual,
                   pure_gdn ? " [PURE GDN]" : "", HIDDEN);
#ifdef KV_INT8
            printf("  KV cache: INT8 (per-head symmetric scale)\n");
#else
            printf("  KV cache: FP32\n");
#endif
            printf("  ctx sweep: %d points, max=%zu\n\n", n_ctx, max_ctx);
        }

        /* Allocate growing KV caches for each full-attention layer */
        for (int l = 0; l < NUM_LAYERS; ++l) {
            if (!is_gdn[l]) {
#ifdef KV_INT8
                /* INT8 KV cache: 1 byte per element + per-head float scale */
                if (posix_memalign((void**)&states[l].kv_k_grow_q, 64, max_ctx * kv_dim) != 0 ||
                    posix_memalign((void**)&states[l].kv_v_grow_q, 64, max_ctx * kv_dim) != 0) {
                    fprintf(stderr, "out of memory (KV_INT8 alloc)\n");
                    exit(1);
                }
                memset(states[l].kv_k_grow_q, 0, max_ctx * kv_dim);
                memset(states[l].kv_v_grow_q, 0, max_ctx * kv_dim);
                states[l].kv_k_scale = alloc_aligned(FULL_KV_HEADS);
                states[l].kv_v_scale = alloc_aligned(FULL_KV_HEADS);
#else
                states[l].kv_k_grow = alloc_aligned(max_ctx * kv_dim);
                states[l].kv_v_grow = alloc_aligned(max_ctx * kv_dim);
#endif
            }
            states[l].kv_pos = 0;
        }

        /* Pre-fill KV caches with random data */
        unsigned fill_seed = 99999;
        for (int l = 0; l < NUM_LAYERS; ++l) {
            if (!is_gdn[l]) {
#ifdef KV_INT8
                /* Pre-fill: generate random FP32, quantize to INT8, discard FP32 */
                float *tmp_k = alloc_aligned(max_ctx * kv_dim);
                float *tmp_v = alloc_aligned(max_ctx * kv_dim);
                fill_rand(tmp_k, max_ctx * kv_dim, &fill_seed);
                fill_rand(tmp_v, max_ctx * kv_dim, &fill_seed);

                /* Per-head symmetric quantization */
                for (size_t kh = 0; kh < FULL_KV_HEADS; ++kh) {
                    float max_k = 0, max_v = 0;
                    for (size_t t = 0; t < max_ctx; ++t)
                        for (size_t d = 0; d < FULL_HEAD_DIM; ++d) {
                            float kv = fabsf(tmp_k[t * kv_dim + kh * FULL_HEAD_DIM + d]);
                            if (kv > max_k) max_k = kv;
                            float vv = fabsf(tmp_v[t * kv_dim + kh * FULL_HEAD_DIM + d]);
                            if (vv > max_v) max_v = vv;
                        }
                    states[l].kv_k_scale[kh] = (max_k > 0) ? max_k / 127.0f : 1.0f;
                    states[l].kv_v_scale[kh] = (max_v > 0) ? max_v / 127.0f : 1.0f;
                    float inv_k = 1.0f / states[l].kv_k_scale[kh];
                    float inv_v = 1.0f / states[l].kv_v_scale[kh];
                    for (size_t t = 0; t < max_ctx; ++t)
                        for (size_t d = 0; d < FULL_HEAD_DIM; ++d) {
                            int kq = (int)lroundf(tmp_k[t * kv_dim + kh * FULL_HEAD_DIM + d] * inv_k);
                            if (kq > 127) kq = 127;
                            if (kq < -128) kq = -128;
                            states[l].kv_k_grow_q[t * kv_dim + kh * FULL_HEAD_DIM + d] = (int8_t)kq;

                            int vq = (int)lroundf(tmp_v[t * kv_dim + kh * FULL_HEAD_DIM + d] * inv_v);
                            if (vq > 127) vq = 127;
                            if (vq < -128) vq = -128;
                            states[l].kv_v_grow_q[t * kv_dim + kh * FULL_HEAD_DIM + d] = (int8_t)vq;
                        }
                }
                free(tmp_k); free(tmp_v);
#else
                fill_rand(states[l].kv_k_grow, max_ctx * kv_dim, &fill_seed);
                fill_rand(states[l].kv_v_grow, max_ctx * kv_dim, &fill_seed);
#endif
            }
        }

        if (csv) {
            printf("model,ctx_len,gdn_layer_us,full_attn_us,ffn_us,total_us,"
                   "tok_per_sec,kv_cache_mb\n");
        }

        for (int ci = 0; ci < n_ctx; ++ci) {
            size_t C = ctx_lens[ci];

            /* Set each full-attention layer to position C-1 (pre-filled) */
            for (int l = 0; l < NUM_LAYERS; ++l) {
                if (!is_gdn[l])
                    states[l].kv_pos = C - 1;
            }

            /* Reset GDN state for a clean measurement */
            for (int l = 0; l < NUM_LAYERS; ++l) {
                if (is_gdn[l]) {
                    memset(states[l].gdn_state, 0, VALUE_DIM * sizeof(float));
                    memset(states[l].conv_hist, 0, (CONV_K - 1) * CONV_DIM * sizeof(float));
                }
            }

            /* Measure: decode 1 token at position C-1 */
            fill_rand(hidden, HIDDEN, &seed);

            double gdn_total = 0, full_total = 0, ffn_total = 0;
            /* Average over a few tokens for stability */
            int reps = (C > 1024) ? 3 : 5;
            for (int r = 0; r < reps; ++r) {
                for (int l = 0; l < NUM_LAYERS; ++l) {
                    if (is_gdn[l]) {
                        double t0 = now_us();
                        gdn_layer_forward(&w, &states[l], hidden, hidden_next, 1);
                        gdn_total += now_us() - t0;
                    } else {
                        double t0 = now_us();
#ifdef KV_INT8
                        full_attn_real_forward_int8kv(&w, &states[l], hidden, hidden_next);
#else
                        full_attn_real_forward(&w, &states[l], hidden, hidden_next);
#endif
                        full_total += now_us() - t0;
                        /* Reset position so next rep measures the same ctx */
                        states[l].kv_pos = C - 1;
                    }
                    double t0 = now_us();
                    ffn_forward(&w, hidden_next, hidden, 1);
                    ffn_total += now_us() - t0;
                    /* No swap: after gdn writes hidden_next and ffn writes back to
                     * hidden (with residual), hidden already holds the correct
                     * output for the next layer. Swapping would feed the stale
                     * attention output to the next layer, dropping the FFN
                     * residual. */
                }
            }
            gdn_total /= reps;
            full_total /= reps;
            ffn_total /= reps;
            double total_us = gdn_total + full_total + ffn_total;
            double tps = 1e6 / total_us;
            /* KV cache memory: actual full-attn layers * C * kv_dim * 2 (K+V)
             * FP32: 4 bytes/element, INT8: 1 byte/element + tiny scale overhead */
            int n_full_actual = 0;
            for (int l2 = 0; l2 < NUM_LAYERS; ++l2)
                if (!is_gdn[l2]) n_full_actual++;
#ifdef KV_INT8
            double kv_mb = (double)n_full_actual * C * kv_dim * 1.0 * 2 / (1024 * 1024);
#else
            double kv_mb = (double)n_full_actual * C * kv_dim * sizeof(float) * 2 / (1024 * 1024);
#endif

            if (csv) {
                printf(MODEL_NAME ",%zu,%.0f,%.0f,%.0f,%.0f,%.2f,%.1f\n",
                       C, gdn_total, full_total, ffn_total, total_us, tps, kv_mb);
            } else {
                printf("  ctx=%5zu  GDN=%7.0f us  full_attn=%7.0f us  FFN=%7.0f us"
                       "  total=%7.0f us  %.2f tok/s  KV=%5.1f MB\n",
                       C, gdn_total, full_total, ffn_total, total_us, tps, kv_mb);
            }
        }

        /* Cleanup growing KV caches */
        for (int l = 0; l < NUM_LAYERS; ++l) {
            if (!is_gdn[l]) {
#ifdef KV_INT8
                free(states[l].kv_k_grow_q);  /* was alloc'd via alloc_aligned (float*) cast */
                free(states[l].kv_v_grow_q);
                free(states[l].kv_k_scale);
                free(states[l].kv_v_scale);
#else
                free(states[l].kv_k_grow);
                free(states[l].kv_v_grow);
#endif
            }
        }

        /* Cleanup weights + states and exit */
        free(hidden); free(hidden_next);
        free_weights(&w);
        free_layer_states(states);
        return 0;
    }

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
            /* No swap — see comment in measured loop. hidden holds the
             * correct post-FFN output for the next layer. */
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
            /* No swap — hidden already holds the correct post-FFN output.
             * The previous swap was a bug: it fed the stale attention output
             * (without the FFN residual) to the next layer. */
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
    free_weights(&w);
    free_layer_states(states);
    return 0;
}
