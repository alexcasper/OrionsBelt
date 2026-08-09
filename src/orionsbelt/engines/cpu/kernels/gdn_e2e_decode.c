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

/* Global flag: when set (--naive), M>1 matmul uses the old naive scalar loop
 * instead of gemm_neon, for before/after performance comparison (ob-8qt.15). */
static int g_use_naive_matmul = 0;

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
#if defined(INT8_WEIGHTS) && defined(__ARM_FEATURE_DOTPROD)
    int8_t *q_sdot; /* K-interleaved repack for SDOT GEMV (NULL if not DOTPROD) */
    size_t  K, N;   /* dimensions at quantize time (for SDOT dispatch) */
#endif
} QW;

/* Packed int4 weight storage (ob-8qt.16). Two signed 4-bit values per byte
 * (range -8..7): even column in the low nibble, odd column in the high
 * nibble. Per-column float scale, same design as QW/INT8. */
typedef struct {
    uint8_t *q;  /* packed int4 weights [K × ceil(N/2)] row-major */
    float   *s;  /* per-column (output) scale [N] */
} QW4;

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

/* Repack quantized weights into K-interleaved layout for SDOT GEMV.
 * Called after quantize_weight when __ARM_FEATURE_DOTPROD is active.
 * Stores the repacked pointer back into the QW struct.
 *
 * repack_int8_k_interleaved is defined further down (with gemv_int8_sdot);
 * forward-declare it here so this compiles instead of getting an implicit
 * int-returning declaration that conflicts with its real int8_t* signature. */
#if defined(__ARM_FEATURE_DOTPROD)
static int8_t *repack_int8_k_interleaved(const int8_t *Bq, size_t K, size_t N);

static void repack_qw_for_sdot(QW *qw, size_t K, size_t N) {
    qw->q_sdot = repack_int8_k_interleaved(qw->q, K, N);
    qw->K = K;
    qw->N = N;
}
#endif
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

/* ---- SDOT-accelerated INT8 GEMV (ob-8qt.14) ----
 *
 * When __ARM_FEATURE_DOTPROD is available (Cortex-A76/A720 etc.), the int8→
 * float32 dequantization in gemv_int8_neon wastes ~3 cycles per 4 elements
 * on widen/convert instructions. This kernel keeps the entire dot-product in
 * int8×int8 → int32, using the SDOT instruction, and only converts to float
 * once at the end.
 *
 * Two changes vs gemv_int8_neon:
 *   1. Input vector is quantized to int8 (symmetric, single scale factor).
 *   2. Weights are repacked into a K-interleaved layout: for each group of
 *      4 consecutive K-elements, the 4 weights for each output column are
 *      stored contiguously. This lets one vdotq_lane_s32 instruction process
 *      4 output columns × 4 K-elements = 16 multiply-accumulates.
 *
 * The repack is a one-time cost at weight quantization time; the per-call cost
 * is just the input quantization (O(K)).
 *
 * Guarded by __ARM_FEATURE_DOTPROD; falls through to gemv_int8_neon on cores
 * without dotprod (A57, A55, etc.).
 */
#if defined(INT8_WEIGHTS) && defined(__ARM_FEATURE_DOTPROD)
#include <arm_neon.h>

/* Repack [K×N] row-major int8 weights into K-interleaved layout for SDOT.
 *
 * Output layout: for each K-group g (g = 0..K_pad/4-1), for each column j:
 *   out[(g*N + j)*4 + i] = in[(g*4+i)*N + j]   for i = 0..3
 *
 * This places 4 consecutive K-elements for each column contiguously, so that
 * 4 consecutive columns' weights for a K-group form a 16-byte SDOT input.
 * K is zero-padded to a multiple of 4.
 */
static int8_t *repack_int8_k_interleaved(const int8_t *Bq, size_t K, size_t N) {
    size_t K_pad = (K + 3) & ~(size_t)3;
    int8_t *out = calloc(K_pad * N, 1);  /* zero-padded */
    if (!out) { fprintf(stderr, "OOM in repack_int8\n"); exit(1); }

    for (size_t g = 0; g < K_pad / 4; ++g) {
        for (size_t j = 0; j < N; ++j) {
            for (size_t i = 0; i < 4; ++i) {
                size_t k = g * 4 + i;
                out[(g * N + j) * 4 + i] = (k < K) ? Bq[k * N + j] : 0;
            }
        }
    }
    return out;
}

/* Quantize a float input vector to int8 with symmetric scaling.
 * Returns the scale factor via *scale_out. */
static void quantize_input_int8(const float *a, int8_t *a_q, size_t K, float *scale_out) {
    float max_abs = 0.0f;
    for (size_t k = 0; k < K; ++k) {
        float v = fabsf(a[k]);
        if (v > max_abs) max_abs = v;
    }
    float scale = (max_abs > 0.0f) ? max_abs / 127.0f : 1.0f;
    float inv = 1.0f / scale;
    for (size_t k = 0; k < K; ++k) {
        int vi = (int)lroundf(a[k] * inv);
        if (vi > 127) vi = 127;
        if (vi < -128) vi = -128;
        a_q[k] = (int8_t)vi;
    }
    *scale_out = scale;
}

/* SDOT-based INT8 GEMV.
 * a is [K] float, Bq_packed is K-interleaved repacked weights, Bs is [N] float
 * weight scale, c is [N] float output.
 *
 * c[n] = Bs[n] * a_scale * sum_k(a_q[k] * Bq[k*N+n])
 */
static void gemv_int8_sdot(const float *a, const int8_t *Bq_packed,
                           const float *Bs, float *c, size_t K, size_t N) {
    size_t K_pad = (K + 3) & ~(size_t)3;
    size_t num_g = K_pad / 4;

    /* Quantize input vector once (shared across all tiles/threads) */
    int8_t a_q[K_pad];
    memset(a_q, 0, K_pad);  /* zero-pad to multiple of 4 */
    float a_scale;
    quantize_input_int8(a, a_q, K, &a_scale);

    /* Column-tiled SDOT GEMV with K-outer sweep (ob-8qt.14).
     *
     * The naive column-outer version (j4 outer, g inner) strides N×4 bytes
     * between consecutive K-group loads — only 25% cache-line utilization,
     * making the SDOT kernel ~1.7× slower than NEON dequant despite 5× fewer
     * instructions per element.
     *
     * This version reverses the nesting: column tiles on the outside (OpenMP
     * parallelized), K-groups in the middle, and sequential 16-byte column
     * loads on the inside. Within each K-group g, the repacked data for
     * columns jt..jt+TILE is contiguous (stride = 16 bytes between loads),
     * giving 100% cache-line utilization — 4× better effective bandwidth.
     *
     * Accumulators stay in L1 (TILE × 4 bytes = 1 KB for TILE=256), loaded
     * and stored once per K-group iteration. The 2× unrolled inner loop
     * overlaps two independent SDOT chains for instruction-level parallelism.
     */
    const size_t TILE = 256;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t jt = 0; jt < N; jt += TILE) {
        size_t tn = (N - jt >= TILE) ? TILE : (N - jt);

        /* Stack accumulators — int32 dot products before final float conversion */
        int32_t acc[TILE];
        memset(acc, 0, tn * sizeof(int32_t));

        /* K-outer sweep: for each K-group, stream through all tile columns */
        for (size_t g = 0; g < num_g; ++g) {
            int8x8_t a_vec = vld1_s8(a_q + g * 4);
            const int8_t *wp = Bq_packed + (g * N + jt) * 4;
            int32_t *accp = acc;
            size_t j = 0;

            /* 2× unrolled: two independent SDOT chains for ILP */
            for (; j + 8 <= tn; j += 8) {
                int8x16_t w0 = vld1q_s8(wp);
                int8x16_t w1 = vld1q_s8(wp + 16);
                int32x4_t a0 = vld1q_s32(accp);
                int32x4_t a1 = vld1q_s32(accp + 4);
                a0 = vdotq_lane_s32(a0, w0, a_vec, 0);
                a1 = vdotq_lane_s32(a1, w1, a_vec, 0);
                vst1q_s32(accp, a0);
                vst1q_s32(accp + 4, a1);
                wp += 32;
                accp += 8;
            }
            /* Single SDOT for remaining full 4-column groups */
            for (; j + 4 <= tn; j += 4) {
                int8x16_t w0 = vld1q_s8(wp);
                int32x4_t a0 = vld1q_s32(accp);
                a0 = vdotq_lane_s32(a0, w0, a_vec, 0);
                vst1q_s32(accp, a0);
                wp += 16;
                accp += 4;
            }
        }

        /* Apply per-column scale: c[n] = acc[n] * a_scale * Bs[n] */
        float32x4_t scv = vdupq_n_f32(a_scale);
        size_t j = 0;
        for (; j + 4 <= tn; j += 4) {
            float32x4_t av = vcvtq_f32_s32(vld1q_s32(acc + j));
            float32x4_t bsv = vld1q_f32(Bs + jt + j);
            av = vmulq_f32(av, vmulq_f32(bsv, scv));
            vst1q_f32(c + jt + j, av);
        }
        for (; j < tn; ++j)
            c[jt + j] = (float)acc[j] * a_scale * Bs[jt + j];
    }
}
#endif /* INT8_WEIGHTS && __ARM_FEATURE_DOTPROD */

#endif /* INT8_WEIGHTS */

/* ---- Cache-blocked GEMM for the prefill (M>1) path (ob-8qt.15) ----
 *
 * The decode (M=1) path uses the row-sweep gemv_neon (§15, 10–15× speedup).
 * For prefill (M>1), the old code fell through to a naive triple-nested scalar
 * loop with column-major B access (stride by N between consecutive K), no NEON,
 * and no OpenMP.  This gemm_neon generalises the §15 row-sweep fix:
 *
 *   - K-outer, N-inner: each B row is accessed sequentially → good cache lines.
 *   - OpenMP `parallel for` over N-tiles (same tile size as gemv_neon).
 *   - NEON FMA inner loop, reusing each B row across all M output rows.
 *   - B elements are reused M times, giving higher arithmetic intensity.
 *
 * For M=1 this is identical to gemv_neon in effect (single M iteration).
 */
static void gemm_neon(const float *A, const float *B, float *C,
                      size_t M, size_t K, size_t N) {
    const size_t TILE = 1024;  /* 4 KB output tile per row, stays in L1 */

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t jt = 0; jt < N; jt += TILE) {
        size_t tn = (N - jt >= TILE) ? TILE : (N - jt);

        /* Zero output tile for all M rows */
        for (size_t i = 0; i < M; ++i)
            memset(C + i * N + jt, 0, tn * sizeof(float));

        /* Sweep K — B row segments accessed sequentially, reused across M */
        for (size_t k = 0; k < K; ++k) {
            const float *Brow = B + k * N + jt;
            for (size_t i = 0; i < M; ++i) {
                float ak = A[i * K + k];
                float *ct = C + i * N + jt;
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
}

/* Naive triple-nested matmul — kept for --verify-matmul correctness checking.
 * Same algorithm that was used for M>1 before ob-8qt.15. */
static void matmul_naive(const float *A, const float *B, float *C,
                         size_t M, size_t K, size_t N) {
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (size_t k = 0; k < K; ++k)
                acc += A[i * K + k] * B[k * N + j];
            C[i * N + j] = acc;
        }
    }
}

static void matmul(const float *A, const float *B, float *C,
                   size_t M, size_t K, size_t N) {
#ifdef __ARM_NEON
    if (M == 1) {
        gemv_neon(A, B, C, K, N);
        return;
    }
    /* Cache-blocked GEMM for prefill (M>1) — ob-8qt.15 */
    if (!g_use_naive_matmul) {
        gemm_neon(A, B, C, M, K, N);
        return;
    }
    /* --naive: fall through to old scalar loop for A/B comparison */
    matmul_naive(A, B, C, M, K, N);
#else
    matmul_naive(A, B, C, M, K, N);
#endif
}

/* INT8 GEMM for M>1 prefill (ob-8qt.15).
 * Dequantizes on-the-fly with NEON — no full-matrix allocation/dequant.
 * Generalisation of gemv_int8_neon to M rows. */
#ifdef INT8_WEIGHTS
static void gemm_int8_neon(const float *A, const int8_t *Bq, const float *Bs,
                           float *C, size_t M, size_t K, size_t N) {
    const size_t TILE = 1024;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t jt = 0; jt < N; jt += TILE) {
        size_t tn = (N - jt >= TILE) ? TILE : (N - jt);

        for (size_t i = 0; i < M; ++i)
            memset(C + i * N + jt, 0, tn * sizeof(float));

        for (size_t k = 0; k < K; ++k) {
            const int8_t *Brow = Bq + k * N + jt;
            for (size_t i = 0; i < M; ++i) {
                float ak = A[i * K + k];
                float *ct = C + i * N + jt;
                size_t j = 0;
#ifdef __ARM_NEON
                float32x4_t akv = vdupq_n_f32(ak);
                for (; j + 8 <= tn; j += 8) {
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
        }

        /* Apply per-column scale (NEON-vectorized) */
        for (size_t i = 0; i < M; ++i) {
            float *ct = C + i * N + jt;
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
}

/* INT8 matmul wrapper (ob-8qt.15): optimised for both M=1 and M>1. */
static void matmul_int8(const float *A, const QW *qw,
                        float *C, size_t M, size_t K, size_t N) {
    if (M == 1) {
#if defined(__ARM_FEATURE_DOTPROD)
        if (qw->q_sdot) {
            gemv_int8_sdot(A, qw->q_sdot, qw->s, C, K, N);
            return;
        }
#endif
        gemv_int8_neon(A, qw->q, qw->s, C, K, N);
        return;
    }
    /* Prefill path */
    if (g_use_naive_matmul) {
        /* Old behavior: full dequant + naive scalar matmul (for A/B comparison) */
        float *Bf = malloc(K * N * sizeof(float));
        for (size_t k = 0; k < K; ++k)
            for (size_t n = 0; n < N; ++n)
                Bf[k * N + n] = (float)qw->q[k * N + n] * qw->s[n];
        matmul_naive(A, Bf, C, M, K, N);
        free(Bf);
    } else {
        /* On-the-fly dequant NEON GEMM (no full matrix alloc) */
        gemm_int8_neon(A, qw->q, qw->s, C, M, K, N);
    }
}
#endif /* INT8_WEIGHTS */

/* ---- INT4 weight-only quantization (ob-8qt.16) ----
 *
 * Natural extension of INT8 weight-only quant: packs two signed 4-bit
 * integers per byte (-8..7), cutting weight memory traffic 8x vs FP32 and
 * 2x vs INT8. Per-column float scale (same design as INT8). Decode is
 * bandwidth-bound, so the 2x weight-traffic reduction is the primary win,
 * offset by increased nibble-unpack compute (NEON: mask + shift + interleave
 * per 16 bytes).
 *
 * Compiled in with -DINT4_WEIGHTS (mutually exclusive with INT8_WEIGHTS).
 * Packing: even column -> low nibble, odd column -> high nibble.
 *
 * Numerically verified (standalone reference-comparison test, not just a
 * benchmark) against a scalar dequant+GEMV/GEMM reference across shapes
 * including odd/non-power-of-2 K/N/M: exact match. See FINDINGS §26 for
 * the history of a real column-order bug this class of test caught in an
 * earlier, differently-convention'd implementation of this same kernel.
 */
#ifdef INT4_WEIGHTS

/* Quantize a [K×N] float matrix to packed signed int4 + per-column scale.
 * Two values per byte: even column -> low nibble, odd column -> high nibble.
 * Value range: -8..7 (4-bit signed). */
static void quantize_weight_int4(const float *B_in, uint8_t **q_out, float **s_out,
                                 size_t K, size_t N) {
    size_t q_cols = (N + 1) / 2;
    uint8_t *q = calloc(K * q_cols, 1);  /* zero-init (we OR nibbles in) */
    float  *s = malloc(N * sizeof(float));
    if (!q || !s) { fprintf(stderr, "OOM in quantize_weight_int4\n"); exit(1); }

    for (size_t n = 0; n < N; ++n) {
        float max_abs = 0.0f;
        for (size_t k = 0; k < K; ++k) {
            float v = fabsf(B_in[k * N + n]);
            if (v > max_abs) max_abs = v;
        }
        s[n] = (max_abs > 0.0f) ? max_abs / 7.0f : 1.0f;
        float inv = 1.0f / s[n];
        for (size_t k = 0; k < K; ++k) {
            float scaled = B_in[k * N + n] * inv;
            int vi = (int)lroundf(scaled);
            if (vi > 7) vi = 7;
            if (vi < -8) vi = -8;
            vi &= 0x0F;  /* store as unsigned nibble (8→-8, 15→-1, etc.) */
            if (n % 2 == 0)
                q[k * q_cols + n / 2] |= (uint8_t)vi;          /* low nibble */
            else
                q[k * q_cols + n / 2] |= (uint8_t)(vi << 4);  /* high nibble */
        }
    }
    *q_out = q;
    *s_out = s;
}

/* INT4 GEMV: dequantize-on-the-fly with NEON nibble unpack.
 * a is [K] float, Bq is [K×ceil(N/2)] packed uint8, Bs is [N] float scale,
 * c is [N] float.
 *
 * Row-sweep with per-column scale. Loads 16 bytes per K-step (32 packed
 * int4 values), splits low/high nibbles, sign-extends, interleaves to
 * restore column order, then widens to float for FMA accumulation. */
static void gemv_int4_neon(const float *a, const uint8_t *Bq, const float *Bs,
                           float *c, size_t K, size_t N) {
    const size_t TILE = 1024;
    size_t q_cols = (N + 1) / 2;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t jt = 0; jt < N; jt += TILE) {
        size_t tn = (N - jt >= TILE) ? TILE : (N - jt);
        float *ct = c + jt;

        memset(ct, 0, tn * sizeof(float));

        for (size_t k = 0; k < K; ++k) {
            float ak = a[k];
            const uint8_t *Brow = Bq + k * q_cols + jt / 2;
            size_t j = 0;
#ifdef __ARM_NEON
            float32x4_t akv = vdupq_n_f32(ak);
            for (; j + 32 <= tn; j += 32) {
                /* Load 16 bytes = 32 packed int4 values (columns j..j+31) */
                uint8x16_t raw = vld1q_u8(Brow + j / 2);
                /* Low nibbles: (byte & 0x0F) → unsigned 0-15 */
                uint8x16_t lo_u = vandq_u8(raw, vdupq_n_u8(0x0F));
                /* High nibbles: (byte >> 4) → unsigned 0-15 */
                uint8x16_t hi_u = vshrq_n_u8(raw, 4);
                /* Sign-extend from 4-bit to 8-bit: (x << 4) >> 4 (arithmetic) */
                int8x16_t lo_s = vshrq_n_s8(
                    vshlq_n_s8(vreinterpretq_s8_u8(lo_u), 4), 4);
                int8x16_t hi_s = vshrq_n_s8(
                    vshlq_n_s8(vreinterpretq_s8_u8(hi_u), 4), 4);
                /* Interleave: lo={c0,c2,...}, hi={c1,c3,...} → {c0,c1,...,c15},
                 *                                          {c16,c17,...,c31} */
                int8x16x2_t inter = vzipq_s8(lo_s, hi_s);

                /* Process 16 values from inter.val[0] (columns j..j+15) */
                {
                    int16x8_t  i16a = vmovl_s8(vget_low_s8(inter.val[0]));
                    int16x8_t  i16b = vmovl_s8(vget_high_s8(inter.val[0]));
                    int32x4_t  i32a = vmovl_s16(vget_low_s16(i16a));
                    int32x4_t  i32b = vmovl_s16(vget_high_s16(i16a));
                    int32x4_t  i32c = vmovl_s16(vget_low_s16(i16b));
                    int32x4_t  i32d = vmovl_s16(vget_high_s16(i16b));
                    float32x4_t fa = vcvtq_f32_s32(i32a);
                    float32x4_t fb = vcvtq_f32_s32(i32b);
                    float32x4_t fc = vcvtq_f32_s32(i32c);
                    float32x4_t fd = vcvtq_f32_s32(i32d);
                    float32x4_t ca = vld1q_f32(ct + j);
                    float32x4_t cb = vld1q_f32(ct + j + 4);
                    float32x4_t cc = vld1q_f32(ct + j + 8);
                    float32x4_t cd = vld1q_f32(ct + j + 12);
                    ca = vfmaq_f32(ca, akv, fa);
                    cb = vfmaq_f32(cb, akv, fb);
                    cc = vfmaq_f32(cc, akv, fc);
                    cd = vfmaq_f32(cd, akv, fd);
                    vst1q_f32(ct + j, ca);
                    vst1q_f32(ct + j + 4, cb);
                    vst1q_f32(ct + j + 8, cc);
                    vst1q_f32(ct + j + 12, cd);
                }
                /* Process 16 values from inter.val[1] (columns j+16..j+31) */
                {
                    int16x8_t  i16a = vmovl_s8(vget_low_s8(inter.val[1]));
                    int16x8_t  i16b = vmovl_s8(vget_high_s8(inter.val[1]));
                    int32x4_t  i32a = vmovl_s16(vget_low_s16(i16a));
                    int32x4_t  i32b = vmovl_s16(vget_high_s16(i16a));
                    int32x4_t  i32c = vmovl_s16(vget_low_s16(i16b));
                    int32x4_t  i32d = vmovl_s16(vget_high_s16(i16b));
                    float32x4_t fa = vcvtq_f32_s32(i32a);
                    float32x4_t fb = vcvtq_f32_s32(i32b);
                    float32x4_t fc = vcvtq_f32_s32(i32c);
                    float32x4_t fd = vcvtq_f32_s32(i32d);
                    float32x4_t ca = vld1q_f32(ct + j + 16);
                    float32x4_t cb = vld1q_f32(ct + j + 20);
                    float32x4_t cc = vld1q_f32(ct + j + 24);
                    float32x4_t cd = vld1q_f32(ct + j + 28);
                    ca = vfmaq_f32(ca, akv, fa);
                    cb = vfmaq_f32(cb, akv, fb);
                    cc = vfmaq_f32(cc, akv, fc);
                    cd = vfmaq_f32(cd, akv, fd);
                    vst1q_f32(ct + j + 16, ca);
                    vst1q_f32(ct + j + 20, cb);
                    vst1q_f32(ct + j + 24, cc);
                    vst1q_f32(ct + j + 28, cd);
                }
            }
#endif
            /* Scalar tail (handles remainder after NEON, or all if no NEON) */
            for (; j < tn; ++j) {
                uint8_t byte = Brow[j / 2];
                int8_t val = (j % 2 == 0) ? (int8_t)(byte & 0x0F)
                                          : (int8_t)(byte >> 4);
                if (val >= 8) val -= 16;  /* sign-extend from 4-bit */
                ct[j] += ak * (float)val;
            }
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

/* INT4 GEMM for M>1 prefill (ob-8qt.16).
 * Same nibble-unpack logic as gemv_int4_neon, generalised to M rows.
 * Each B row segment is reused across all M output rows. */
static void gemm_int4_neon(const float *A, const uint8_t *Bq, const float *Bs,
                           float *C, size_t M, size_t K, size_t N) {
    const size_t TILE = 1024;
    size_t q_cols = (N + 1) / 2;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t jt = 0; jt < N; jt += TILE) {
        size_t tn = (N - jt >= TILE) ? TILE : (N - jt);

        for (size_t i = 0; i < M; ++i)
            memset(C + i * N + jt, 0, tn * sizeof(float));

        for (size_t k = 0; k < K; ++k) {
            const uint8_t *Brow = Bq + k * q_cols + jt / 2;
            for (size_t i = 0; i < M; ++i) {
                float ak = A[i * K + k];
                float *ct = C + i * N + jt;
                size_t j = 0;
#ifdef __ARM_NEON
                float32x4_t akv = vdupq_n_f32(ak);
                for (; j + 32 <= tn; j += 32) {
                    uint8x16_t raw = vld1q_u8(Brow + j / 2);
                    uint8x16_t lo_u = vandq_u8(raw, vdupq_n_u8(0x0F));
                    uint8x16_t hi_u = vshrq_n_u8(raw, 4);
                    int8x16_t lo_s = vshrq_n_s8(
                        vshlq_n_s8(vreinterpretq_s8_u8(lo_u), 4), 4);
                    int8x16_t hi_s = vshrq_n_s8(
                        vshlq_n_s8(vreinterpretq_s8_u8(hi_u), 4), 4);
                    int8x16x2_t inter = vzipq_s8(lo_s, hi_s);

                    {
                        int16x8_t  i16a = vmovl_s8(vget_low_s8(inter.val[0]));
                        int16x8_t  i16b = vmovl_s8(vget_high_s8(inter.val[0]));
                        int32x4_t  i32a = vmovl_s16(vget_low_s16(i16a));
                        int32x4_t  i32b = vmovl_s16(vget_high_s16(i16a));
                        int32x4_t  i32c = vmovl_s16(vget_low_s16(i16b));
                        int32x4_t  i32d = vmovl_s16(vget_high_s16(i16b));
                        float32x4_t fa = vcvtq_f32_s32(i32a);
                        float32x4_t fb = vcvtq_f32_s32(i32b);
                        float32x4_t fc = vcvtq_f32_s32(i32c);
                        float32x4_t fd = vcvtq_f32_s32(i32d);
                        float32x4_t ca = vld1q_f32(ct + j);
                        float32x4_t cb = vld1q_f32(ct + j + 4);
                        float32x4_t cc = vld1q_f32(ct + j + 8);
                        float32x4_t cd = vld1q_f32(ct + j + 12);
                        ca = vfmaq_f32(ca, akv, fa);
                        cb = vfmaq_f32(cb, akv, fb);
                        cc = vfmaq_f32(cc, akv, fc);
                        cd = vfmaq_f32(cd, akv, fd);
                        vst1q_f32(ct + j, ca);
                        vst1q_f32(ct + j + 4, cb);
                        vst1q_f32(ct + j + 8, cc);
                        vst1q_f32(ct + j + 12, cd);
                    }
                    {
                        int16x8_t  i16a = vmovl_s8(vget_low_s8(inter.val[1]));
                        int16x8_t  i16b = vmovl_s8(vget_high_s8(inter.val[1]));
                        int32x4_t  i32a = vmovl_s16(vget_low_s16(i16a));
                        int32x4_t  i32b = vmovl_s16(vget_high_s16(i16a));
                        int32x4_t  i32c = vmovl_s16(vget_low_s16(i16b));
                        int32x4_t  i32d = vmovl_s16(vget_high_s16(i16b));
                        float32x4_t fa = vcvtq_f32_s32(i32a);
                        float32x4_t fb = vcvtq_f32_s32(i32b);
                        float32x4_t fc = vcvtq_f32_s32(i32c);
                        float32x4_t fd = vcvtq_f32_s32(i32d);
                        float32x4_t ca = vld1q_f32(ct + j + 16);
                        float32x4_t cb = vld1q_f32(ct + j + 20);
                        float32x4_t cc = vld1q_f32(ct + j + 24);
                        float32x4_t cd = vld1q_f32(ct + j + 28);
                        ca = vfmaq_f32(ca, akv, fa);
                        cb = vfmaq_f32(cb, akv, fb);
                        cc = vfmaq_f32(cc, akv, fc);
                        cd = vfmaq_f32(cd, akv, fd);
                        vst1q_f32(ct + j + 16, ca);
                        vst1q_f32(ct + j + 20, cb);
                        vst1q_f32(ct + j + 24, cc);
                        vst1q_f32(ct + j + 28, cd);
                    }
                }
#endif
                for (; j < tn; ++j) {
                    uint8_t byte = Brow[j / 2];
                    int8_t val = (j % 2 == 0) ? (int8_t)(byte & 0x0F)
                                              : (int8_t)(byte >> 4);
                    if (val >= 8) val -= 16;
                    ct[j] += ak * (float)val;
                }
            }
        }

        /* Apply per-column scale for all M rows */
        for (size_t i = 0; i < M; ++i) {
            float *ct = C + i * N + jt;
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
}

/* INT4 matmul wrapper (ob-8qt.16): dispatches GEMV (M=1) or GEMM (M>1). */
static void matmul_int4(const float *A, const QW4 *qw,
                        float *C, size_t M, size_t K, size_t N) {
    if (M == 1) {
        gemv_int4_neon(A, qw->q, qw->s, C, K, N);
        return;
    }
    gemm_int4_neon(A, qw->q, qw->s, C, M, K, N);
}

#endif /* INT4_WEIGHTS */

/* Dispatch macro: INT4 > INT8 > FP32 (mutually exclusive). */
#if defined(INT4_WEIGHTS)
#define MM(A, Bf, Bq, C, M, K, N) matmul_int4(A, &(Bq), C, M, K, N)
#elif defined(INT8_WEIGHTS)
#define MM(A, Bf, Bq, C, M, K, N) matmul_int8(A, &(Bq), C, M, K, N)
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
    /* Quantized weight versions: QW4 under INT4_WEIGHTS, QW under INT8_WEIGHTS */
#if defined(INT4_WEIGHTS)
    QW4 g_q_proj_q, g_k_proj_q, g_v_proj_q, g_o_proj_q;
    QW4 f_q_proj_q, f_k_proj_q, f_v_proj_q, f_o_proj_q;
    QW4 gate_proj_q, up_proj_q, down_proj_q;
#elif defined(INT8_WEIGHTS)
    QW g_q_proj_q, g_k_proj_q, g_v_proj_q, g_o_proj_q;
    QW f_q_proj_q, f_k_proj_q, f_v_proj_q, f_o_proj_q;
    QW gate_proj_q, up_proj_q, down_proj_q;
#endif
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
#if defined(INT4_WEIGHTS)
#define FREE_QW(qw) do { free((qw).q); free((qw).s); } while(0)
    FREE_QW(w->g_q_proj_q);
    FREE_QW(w->g_k_proj_q);
    FREE_QW(w->g_v_proj_q);
    FREE_QW(w->g_o_proj_q);
    FREE_QW(w->f_q_proj_q);
    FREE_QW(w->f_k_proj_q);
    FREE_QW(w->f_v_proj_q);
    FREE_QW(w->f_o_proj_q);
    FREE_QW(w->gate_proj_q);
    FREE_QW(w->up_proj_q);
    FREE_QW(w->down_proj_q);
#elif defined(INT8_WEIGHTS)
#if defined(__ARM_FEATURE_DOTPROD)
#define FREE_QW(qw) do { free((qw).q); free((qw).s); free((qw).q_sdot); } while(0)
#else
#define FREE_QW(qw) do { free((qw).q); free((qw).s); } while(0)
#endif
    FREE_QW(w->g_q_proj_q);
    FREE_QW(w->g_k_proj_q);
    FREE_QW(w->g_v_proj_q);
    FREE_QW(w->g_o_proj_q);
    FREE_QW(w->f_q_proj_q);
    FREE_QW(w->f_k_proj_q);
    FREE_QW(w->f_v_proj_q);
    FREE_QW(w->f_o_proj_q);
    FREE_QW(w->gate_proj_q);
    FREE_QW(w->up_proj_q);
    FREE_QW(w->down_proj_q);
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
    int prefill_M = 0;
    int verify_matmul = 0;
    int verify_int4 = 0;
    int use_naive = 0;
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
        else if (strcmp(argv[i], "--prefill") == 0 && i + 1 < argc)
            prefill_M = atoi(argv[++i]);
        else if (strcmp(argv[i], "--verify-matmul") == 0)
            verify_matmul = 1;
        else if (strcmp(argv[i], "--verify-int4") == 0)
            verify_int4 = 1;
        else if (strcmp(argv[i], "--naive") == 0)
            use_naive = 1;
        else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [OPTIONS]\n", argv[0]);
            printf("  --tokens N        Decode N tokens (default 8)\n");
            printf("  --csv             CSV output for fleet sweep\n");
            printf("  --ctx-sweep L1,L2,...  Decode cost at growing context lengths\n");
            printf("  --pure-gdn        All layers GDN (no full-attention)\n");
            printf("  --prefill M       Process M tokens in one pass (measures TTFT / prefill cost)\n");
            printf("                    Exercises the cache-blocked M>1 matmul path (ob-8qt.15)\n");
            printf("  --verify-matmul   Verify gemm_neon correctness vs naive matmul, then exit\n");
#ifdef INT4_WEIGHTS
            printf("  --verify-int4     Verify INT4 GEMV accuracy vs FP32 oracle, then exit\n");
#endif
            printf("  --naive           Force naive scalar matmul for M>1 (A/B comparison)\n");
            return 0;
        }
    }

    /* Validate token count — 0 or negative causes division by zero in
     * mean/percentile stats and array underflow in the sort loop.
     * Skip for prefill mode (doesn't use the decode loop). */
    if (prefill_M == 0 && num_tokens < 1) {
        fprintf(stderr, "Error: --tokens must be >= 1 (got %d)\n", num_tokens);
        return 1;
    }

    /* ---- --verify-matmul: correctness check of gemm_neon vs naive (ob-8qt.15) ----
     *
     * Runs both matmul implementations on small test cases covering the actual
     * layer shapes, then reports the maximum absolute difference.  Exits 0 on
     * success (max error < 1e-4 relative) or 1 on failure.
     *
     * This is a self-contained test — no weight allocation, no model state. */
    if (verify_matmul) {
        /* Test shapes drawn from the real model geometry (4B default). */
        struct { size_t M, K, N; const char *desc; } tests[] = {
            { 1,   64,  128, "M=1  K=64  N=128  (small decode)" },
            { 1,   256, 512, "M=1  K=256 N=512  (GEMV)" },
            { 4,   64,  128, "M=4  K=64  N=128  (small prefill)" },
            { 16,  128, 256, "M=16 K=128 N=256  (medium prefill)" },
            { 64,  128, 128, "M=64 K=128 N=128  (prefill chunk)" },
            { 64,  256, 1024,"M=64 K=256 N=1024 (large prefill)" },
        };
        int n_tests = sizeof(tests) / sizeof(tests[0]);
        int failures = 0;
        unsigned vseed = 42;

        printf("Verifying gemm_neon vs matmul_naive (%d test cases):\n\n", n_tests);

        for (int t = 0; t < n_tests; ++t) {
            size_t M = tests[t].M, K = tests[t].K, N = tests[t].N;
            float *A  = malloc(M * K * sizeof(float));
            float *B  = malloc(K * N * sizeof(float));
            float *C1 = malloc(M * N * sizeof(float));
            float *C2 = malloc(M * N * sizeof(float));

            for (size_t i = 0; i < M * K; ++i) A[i] = ((float)(vseed = (vseed * 1103515245 + 12345)) / 2147483647.0f - 0.5f) * 2.0f;
            for (size_t i = 0; i < K * N; ++i) B[i] = ((float)(vseed = (vseed * 1103515245 + 12345)) / 2147483647.0f - 0.5f) * 2.0f;

            matmul_naive(A, B, C1, M, K, N);
            matmul(A, B, C2, M, K, N);

            float max_err = 0.0f;
            float max_val = 0.0f;
            for (size_t i = 0; i < M * N; ++i) {
                float diff = fabsf(C1[i] - C2[i]);
                if (diff > max_err) max_err = diff;
                float av = fabsf(C1[i]);
                if (av > max_val) max_val = av;
            }

            float rel_err = (max_val > 0) ? max_err / max_val : max_err;
            int ok = (rel_err < 1e-4f);
            if (!ok) failures++;

            printf("  %s: max_abs=%.2e max_val=%.2e rel_err=%.2e %s\n",
                   tests[t].desc, max_err, max_val, rel_err,
                   ok ? "PASS" : "FAIL");

            free(A); free(B); free(C1); free(C2);
        }

        printf("\n%s: %d/%d passed\n", failures ? "FAILED" : "ALL PASS", n_tests - failures, n_tests);
        return failures ? 1 : 0;
    }

    /* ---- --verify-int4: INT4 accuracy vs FP32 oracle (ob-8qt.16) ----
     *
     * Generates random weight matrices at real model shapes, quantizes to
     * INT4, runs both FP32 GEMV and INT4 GEMV on random inputs, and compares.
     * Reports max abs error, max relative error, and SNR (dB).
     * Exits 0 on success (SNR > 20 dB on all matrices) or 1 on failure. */
#ifdef INT4_WEIGHTS
    if (verify_int4) {
        struct { size_t K, N; const char *desc; } tests[] = {
            { HIDDEN, KEY_DIM,                     "GDN q/k_proj" },
            { HIDDEN, VALUE_DIM,                   "GDN v_proj" },
            { VALUE_DIM, HIDDEN,                   "GDN o_proj" },
            { HIDDEN, FULL_HEADS * FULL_HEAD_DIM,  "Full q_proj" },
            { HIDDEN, INTER,                       "FFN gate/up" },
            { INTER, HIDDEN,                       "FFN down" },
        };
        int n_tests = sizeof(tests) / sizeof(tests[0]);
        int failures = 0;
        unsigned vseed = 42;

        printf("Verifying INT4 GEMV accuracy vs FP32 oracle (%d matrices):\n\n", n_tests);

        for (int t = 0; t < n_tests; ++t) {
            size_t K = tests[t].K, N = tests[t].N;
            float *B = malloc(K * N * sizeof(float));
            float *a = malloc(K * sizeof(float));
            float *c_fp32 = malloc(N * sizeof(float));
            float *c_int4 = malloc(N * sizeof(float));

            for (size_t i = 0; i < K * N; ++i)
                B[i] = ((float)(vseed = (vseed * 1103515245 + 12345)) / 2147483647.0f - 0.5f) * 2.0f;
            for (size_t i = 0; i < K; ++i)
                a[i] = ((float)(vseed = (vseed * 1103515245 + 12345)) / 2147483647.0f - 0.5f) * 2.0f;

            matmul(a, B, c_fp32, 1, K, N);

            uint8_t *q4; float *s4;
            quantize_weight_int4(B, &q4, &s4, K, N);
            gemv_int4_neon(a, q4, s4, c_int4, K, N);

            double sig_energy = 0.0, err_energy = 0.0;
            float max_err = 0.0f, max_val = 0.0f;
            for (size_t i = 0; i < N; ++i) {
                float diff = fabsf(c_fp32[i] - c_int4[i]);
                if (diff > max_err) max_err = diff;
                float av = fabsf(c_fp32[i]);
                if (av > max_val) max_val = av;
                sig_energy += (double)c_fp32[i] * c_fp32[i];
                err_energy += (double)diff * diff;
            }
            float rel_err = (max_val > 0) ? max_err / max_val : max_err;
            double snr_db = (err_energy > 0.0)
                ? 10.0 * log10(sig_energy / err_energy)
                : 999.0;
            int ok = (snr_db > 20.0);
            if (!ok) failures++;

            printf("  %-14s K=%-5zu N=%-5zu  SNR=%.1f dB  max_rel_err=%.2f%%  %s\n",
                   tests[t].desc, K, N, snr_db, rel_err * 100.0f,
                   ok ? "PASS" : "FAIL");

            free(B); free(a); free(c_fp32); free(c_int4); free(q4); free(s4);
        }

        printf("\n%s: %d/%d passed (SNR > 20 dB threshold)\n",
               failures ? "FAILED" : "ALL PASS", n_tests - failures, n_tests);
        return failures ? 1 : 0;
    }
#endif

    unsigned seed = 12345;

    /* Wire --naive flag to global */
    g_use_naive_matmul = use_naive;

    /* Allocate weights (random, benchmark only) */
    Weights w;
    /* Zero first: QW.q_sdot/K/N (ob-8qt.14 SDOT path) are populated below by
     * repack_qw_for_sdot() on dotprod cores; without zeroing, the QW structs
     * on non-dotprod builds would have indeterminate stack garbage in any
     * padding fields. */
    memset(&w, 0, sizeof(w));
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

    /* Repack INT8 weights into K-interleaved layout for SDOT GEMV (ob-8qt.14).
     * On dotprod cores this enables the gemv_int8_sdot path (int8×int8→int32,
     * no float32 dequant overhead). On non-dotprod cores (A57, A55 Armv8.0)
     * the SDOT path is not compiled and q_sdot stays NULL, falling through to
     * the verified gemv_int8_neon path. */
#if defined(__ARM_FEATURE_DOTPROD)
    repack_qw_for_sdot(&w.g_q_proj_q,  HIDDEN, KEY_DIM);
    repack_qw_for_sdot(&w.g_k_proj_q,  HIDDEN, KEY_DIM);
    repack_qw_for_sdot(&w.g_v_proj_q,  HIDDEN, VALUE_DIM);
    repack_qw_for_sdot(&w.g_o_proj_q,  VALUE_DIM, HIDDEN);
    repack_qw_for_sdot(&w.f_q_proj_q,  HIDDEN, FULL_HEADS * FULL_HEAD_DIM);
    repack_qw_for_sdot(&w.f_k_proj_q,  HIDDEN, FULL_KV_HEADS * FULL_HEAD_DIM);
    repack_qw_for_sdot(&w.f_v_proj_q,  HIDDEN, FULL_KV_HEADS * FULL_HEAD_DIM);
    repack_qw_for_sdot(&w.f_o_proj_q,  FULL_HEADS * FULL_HEAD_DIM, HIDDEN);
    repack_qw_for_sdot(&w.gate_proj_q, HIDDEN, INTER);
    repack_qw_for_sdot(&w.up_proj_q,   HIDDEN, INTER);
    repack_qw_for_sdot(&w.down_proj_q, INTER, HIDDEN);
    if (!csv)
        printf("  Weight quantization: INT8 (weight-only, per-column symmetric scale)\n"
               "  SDOT repack: enabled (__ARM_FEATURE_DOTPROD)\n\n");
#else
    if (!csv)
        printf("  Weight quantization: INT8 (weight-only, per-column symmetric scale)\n\n");
#endif
#endif /* INT8_WEIGHTS */

#ifdef INT4_WEIGHTS
    quantize_weight_int4(w.g_q_proj,  &w.g_q_proj_q.q,  &w.g_q_proj_q.s,  HIDDEN, KEY_DIM);
    quantize_weight_int4(w.g_k_proj,  &w.g_k_proj_q.q,  &w.g_k_proj_q.s,  HIDDEN, KEY_DIM);
    quantize_weight_int4(w.g_v_proj,  &w.g_v_proj_q.q,  &w.g_v_proj_q.s,  HIDDEN, VALUE_DIM);
    quantize_weight_int4(w.g_o_proj,  &w.g_o_proj_q.q,  &w.g_o_proj_q.s,  VALUE_DIM, HIDDEN);
    quantize_weight_int4(w.f_q_proj,  &w.f_q_proj_q.q,  &w.f_q_proj_q.s,  HIDDEN, FULL_HEADS * FULL_HEAD_DIM);
    quantize_weight_int4(w.f_k_proj,  &w.f_k_proj_q.q,  &w.f_k_proj_q.s,  HIDDEN, FULL_KV_HEADS * FULL_HEAD_DIM);
    quantize_weight_int4(w.f_v_proj,  &w.f_v_proj_q.q,  &w.f_v_proj_q.s,  HIDDEN, FULL_KV_HEADS * FULL_HEAD_DIM);
    quantize_weight_int4(w.f_o_proj,  &w.f_o_proj_q.q,  &w.f_o_proj_q.s,  FULL_HEADS * FULL_HEAD_DIM, HIDDEN);
    quantize_weight_int4(w.gate_proj, &w.gate_proj_q.q, &w.gate_proj_q.s, HIDDEN, INTER);
    quantize_weight_int4(w.up_proj,   &w.up_proj_q.q,   &w.up_proj_q.s,   HIDDEN, INTER);
    quantize_weight_int4(w.down_proj, &w.down_proj_q.q, &w.down_proj_q.s, INTER, HIDDEN);
    if (!csv)
        printf("  Weight quantization: INT4 (weight-only, packed nibbles, per-column symmetric scale)\n\n");
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

    /* ---- --prefill M: process M tokens in one pass (ob-8qt.15) ----
     *
     * Exercises the cache-blocked M>1 matmul path directly.  All projection
     * matmuls run with M=prefill_M instead of M=1, which is the prefill /
     * TTFT bottleneck path.  Measures per-layer-type cost and total TTFT.
     *
     * This is a single forward pass over M tokens (not auto-regressive decode),
     * which is exactly what "prefill" means: process the entire prompt before
     * generating the first output token. */
    if (prefill_M > 0) {
        size_t M = (size_t)prefill_M;

        /* Allocate prefill-sized buffers */
        float *h_in  = alloc_aligned(M * HIDDEN);
        float *h_out = alloc_aligned(M * HIDDEN);

        if (!csv) {
            printf(MODEL_NAME " prefill benchmark (M=%zu)\n", M);
            printf("  layers=%d (GDN=%d, full=%d), hidden=%d\n",
                   NUM_LAYERS, NUM_GDN, NUM_FULL, HIDDEN);
            printf("  tokens/prefill=%zu\n\n", M);
        }

        /* Warmup (2 passes) */
        for (int r = 0; r < 2; ++r) {
            fill_rand(h_in, M * HIDDEN, &seed);
            for (int l = 0; l < NUM_LAYERS; ++l) {
                if (is_gdn[l])
                    gdn_layer_forward(&w, &states[l], h_in, h_out, M);
                else
                    full_attn_layer_forward(&w, &states[l], h_in, h_out, M);
                ffn_forward(&w, h_out, h_in, M);
            }
        }

        /* Reset GDN state */
        for (int l = 0; l < NUM_LAYERS; ++l) {
            memset(states[l].gdn_state, 0, VALUE_DIM * sizeof(float));
            memset(states[l].conv_hist, 0, (CONV_K - 1) * CONV_DIM * sizeof(float));
        }

        /* Measured run */
        fill_rand(h_in, M * HIDDEN, &seed);
        double gdn_us = 0, full_us = 0, ffn_us = 0;

        for (int l = 0; l < NUM_LAYERS; ++l) {
            if (is_gdn[l]) {
                double t0 = now_us();
                gdn_layer_forward(&w, &states[l], h_in, h_out, M);
                gdn_us += now_us() - t0;
            } else {
                double t0 = now_us();
                full_attn_layer_forward(&w, &states[l], h_in, h_out, M);
                full_us += now_us() - t0;
            }
            double t0 = now_us();
            ffn_forward(&w, h_out, h_in, M);
            ffn_us += now_us() - t0;
        }

        double total_us = gdn_us + full_us + ffn_us;
        double ttft_ms = total_us / 1e3;
        double tok_per_sec_prefill = (double)M * 1e6 / total_us;

        if (csv) {
            printf("model,prefill_M,ttft_ms,tok_per_sec_prefill,gdn_us,full_us,ffn_us\n");
            printf(MODEL_NAME ",%zu,%.2f,%.2f,%.0f,%.0f,%.0f\n",
                   M, ttft_ms, tok_per_sec_prefill, gdn_us, full_us, ffn_us);
        } else {
            printf("Results (prefill M=%zu):\n", M);
            printf("  TTFT (prefill total):  %.2f ms  (%.0f us)\n", ttft_ms, total_us);
            printf("  Prefill throughput:    %.2f tok/s  (%.2f ms/tok)\n",
                   tok_per_sec_prefill, ttft_ms / (double)M);
            printf("  Phase breakdown:\n");
            printf("    GDN layers:     %7.0f us  (%.1f%%)\n",
                   gdn_us, 100.0 * gdn_us / total_us);
            printf("    Full-attn layers:%6.0f us  (%.1f%%)\n",
                   full_us, 100.0 * full_us / total_us);
            printf("    FFN (all layers):%6.0f us  (%.1f%%)\n",
                   ffn_us, 100.0 * ffn_us / total_us);
        }

        free(h_in); free(h_out);
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
