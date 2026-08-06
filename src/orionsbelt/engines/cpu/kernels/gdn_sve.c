/* Gated DeltaNet CPU micro-kernels for AArch64 SVE (baseline SVE1) with scalar fallback.
 *
 * Bead ob-8qt.1. Three primitives that exist nowhere today for a non-SME Armv9.2 core
 * (see docs/FINDINGS.md section 3): KleidiAI's depthwise conv is SME2-only, and it has no
 * prefix-product or scan kernel at all.
 *
 * THE LAYOUT DECISION THAT MAKES THIS EASY
 * ----------------------------------------
 * A prefix scan along the *vector lanes* needs a log-depth Hillis-Steele shuffle network.
 * We never do that. GDN's sequence axis is inherently sequential, so we vectorize across
 * the CHANNEL/HEAD axis and walk the sequence with a plain loop. Every kernel below is
 * then a sequence of independent lane-wise FMAs -- no cross-lane communication anywhere.
 *
 * ISA FLOOR IS SVE1, NOT SVE2. Every intrinsic used here -- svcntw, svdup_f32, svld1_f32,
 * svmul_f32_x, svmla_f32_x, svst1_f32, svwhilelt_b32 -- is base SVE, and the guard is
 * __ARM_FEATURE_SVE. Nothing in these fp32 kernels needs SVE2's integer/DSP additions. SVE2
 * (and i8mm) become relevant only for the quantized delta-rule matmuls, which live elsewhere.
 *
 * Verified identical on SVE1 at 128/256/512-bit (incl. -mcpu=neoverse-v1 and -mcpu=a64fx),
 * on SVE2 (-march=armv9-a, cortex-a710, neoverse-v2), and on plain -march=armv8-a via the
 * scalar fallback. That means these run on Graviton3 (SVE1) as well as Cortex-A720 (SVE2),
 * which keeps the Edge AI hedge target viable.
 *
 * On Cortex-A720 SVE is 128-bit -- the same width as NEON -- so the win is predication
 * (svwhilelt gives clean channel tails with no scalar epilogue), not extra lanes. Written
 * vector-length-agnostic regardless, so it widens for free on a core with longer vectors.
 *
 * MULTI-THREADED (bead ob-8qt.6)
 * ------------------------------
 * The channel loop is embarrassingly parallel — each channel's recurrence is independent.
 * With -fopenmp, '#pragma omp parallel for schedule(static)' distributes channel groups
 * across cores. On the Jetson Nano A57 (4 cores) this gives ~3× throughput. Without
 * -fopenmp the pragmas are ignored and the code compiles single-threaded as before.
 *
 * Build (any of these work):
 *   aarch64-linux-gnu-gcc -O3 -fopenmp -march=armv8.2-a+sve   -c gdn_sve.c   # SVE1 floor
 *   aarch64-linux-gnu-gcc -O3 -fopenmp -march=armv9.2-a+sve2+i8mm+bf16 -c gdn_sve.c
 *   clang --target=aarch64-linux-gnu -O3 -fopenmp -mcpu=cortex-a720 -c gdn_sve.c
 */

#include <stddef.h>
#include <string.h>   /* memcpy for bf16 software conversion */
#include <stdint.h>

#include "gdn_sve.h"

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

/* ===========================================================================
 * Mixed-precision state variants (bead ob-8qt.4)
 *
 * The persistent GDN recurrent state is the memory we want to halve: 48 MB fp32
 * → 24 MB fp16/bf16 across 24 layers (Qwen3.5-4B).  The CRITICAL precision
 * constraint is that the decay accumulator stays fp32 — a gate of 0.5 compounded
 * over 64 steps is ~5e-20 and underflows fp16 outright.
 *
 * Design: narrow storage, wide compute.  State is loaded from fp16/bf16, widened
 * to fp32 for the inner loop (identical arithmetic to the fp32 kernel), and
 * narrowed back on store.  The state conversion is O(channels) per chunk —
 * negligible against the O(channels × seq) inner loop — while the persistent
 * memory footprint is halved.
 *
 * bf16 has 8 exponent bits (same range as fp32) but only 7 mantissa bits.
 * fp16 has 5 exponent bits (narrower range) but 10 mantissa bits.
 * For GDN state, bf16 is generally the better choice: recurrent values stay
 * in [−1, 1] (bounded by gating), so fp16's extra mantissa precision matters
 * more than bf16's wider range — but we implement both and let the benchmark
 * decide.
 * ==========================================================================*/

/* --- Software bf16 conversion (works on any ISA, no hardware bf16 needed) --- */

static inline uint16_t f32_to_bf16_rne(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    /* Round-to-nearest-even: add bias that rounds the truncated bits. */
    uint32_t lsb = (u >> 16) & 1;
    uint32_t rounding_bias = 0x7FFFu + lsb;
    return (uint16_t)((u + rounding_bias) >> 16);
}

static inline float bf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

#ifdef __ARM_NEON
/* Vectorized NEON fp32→bf16 round-to-nearest-even (4 lanes).
 * Replaces the O(4) scalar loop with three NEON ops — critical for cumdecay
 * where the conversion runs every timestep, not just at chunk boundaries. */
static inline uint16x4_t vcvtq_bf16_from_f32(float32x4_t f) {
    uint32x4_t u = vreinterpretq_u32_f32(f);
    uint32x4_t lsb = vandq_u32(vshrq_n_u32(u, 16), vdupq_n_u32(1));
    uint32x4_t bias = vaddq_u32(vdupq_n_u32(0x7FFF), lsb);
    return vshrn_n_u32(vaddq_u32(u, bias), 16);
}
/* Vectorized NEON bf16→fp32 (4 lanes): widen + shift left 16. */
static inline float32x4_t vcvtq_f32_from_bf16(uint16x4_t b) {
    return vreinterpretq_f32_u32(vshll_n_u16(b, 16));
}
#endif

/* ---------------------------------------------------------------------------
 * 1. Gated cumulative decay  (inclusive prefix product along the sequence)
 *
 *   decay[t][c] = prod_{i<=t} a[i][c]
 *
 * Computed directly rather than as exp(cumsum(log a)): at chunk lengths of 64 the
 * direct product is both cheaper and accurate in fp32, and avoids two transcendentals
 * per element. NOTE the accumulator is fp32 even when the surrounding state is fp16 --
 * a decay of 0.5 over 64 steps is ~5e-20, which underflows fp16 outright.
 * ------------------------------------------------------------------------- */
void gdn_cumdecay_f32(const float *restrict a, float *restrict decay, size_t seq,
                      size_t channels) {
#ifdef __ARM_FEATURE_SVE
    {
        unsigned vl = (unsigned)svcntw();
        size_t n_vec = (channels + vl - 1) / vl;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (size_t vi = 0; vi < n_vec; ++vi) {
            size_t c = vi * vl;
            svbool_t pg = svwhilelt_b32((unsigned)c, (unsigned)channels);
            svfloat32_t run = svdup_f32(1.0f);
            for (size_t t = 0; t < seq; ++t) {
                svfloat32_t av = svld1_f32(pg, a + t * channels + c);
                run = svmul_f32_x(pg, run, av);
                svst1_f32(pg, decay + t * channels + c, run);
            }
        }
    }
#elif defined(__ARM_NEON)
    /* Double-width unroll: 8 channels/iter (2 NEON register groups) hides
     * FMA/MUL latency and doubles memory-level parallelism.  The A57's
     * 3-4 cycle FMA/MUL latency is perfectly hidden at this width. */
    size_t n_vec8 = channels >> 3;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t vi = 0; vi < n_vec8; ++vi) {
        size_t c = vi << 3;
        float32x4_t run0 = vdupq_n_f32(1.0f);
        float32x4_t run1 = vdupq_n_f32(1.0f);
        for (size_t t = 0; t < seq; ++t) {
            run0 = vmulq_f32(run0, vld1q_f32(a + t * channels + c));
            run1 = vmulq_f32(run1, vld1q_f32(a + t * channels + c + 4));
            vst1q_f32(decay + t * channels + c, run0);
            vst1q_f32(decay + t * channels + c + 4, run1);
        }
    }
    /* Remaining groups of 4 (at most 1) */
    for (size_t c = n_vec8 << 3; c + 4 <= channels; c += 4) {
        float32x4_t run = vdupq_n_f32(1.0f);
        for (size_t t = 0; t < seq; ++t) {
            run = vmulq_f32(run, vld1q_f32(a + t * channels + c));
            vst1q_f32(decay + t * channels + c, run);
        }
    }
    for (size_t c = (n_vec8 << 3) + (((channels >> 2) & 1) << 2); c < channels; ++c) {
        float run = 1.0f;
        for (size_t t = 0; t < seq; ++t) {
            run *= a[t * channels + c];
            decay[t * channels + c] = run;
        }
    }
#else
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t c = 0; c < channels; ++c) {
        float run = 1.0f;
        for (size_t t = 0; t < seq; ++t) {
            run *= a[t * channels + c];
            decay[t * channels + c] = run;
        }
    }
#endif
}

/* ---------------------------------------------------------------------------
 * 2. Chunkwise gated scan  (the recurrence the NPU cannot express at all)
 *
 *   s[t][c] = g[t][c] * s[t-1][c] + x[t][c]
 *
 * This is the outer, sequential half of the chunkwise formulation, deliberately kept
 * separate from the per-chunk dense math so the mapping ADR can move the inner matmuls
 * to GPU/NPU while this stays on the CPU (PLAN.md section 3.1). state[] carries across
 * calls, which is exactly the cross-invocation continuity the NOE toolchain has no
 * mechanism for.
 * ------------------------------------------------------------------------- */
void gdn_gated_scan_f32(const float *restrict g, const float *restrict x,
                        float *restrict s, float *restrict state, size_t seq,
                        size_t channels) {
#ifdef __ARM_FEATURE_SVE
    {
        unsigned vl = (unsigned)svcntw();
        size_t n_vec = (channels + vl - 1) / vl;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (size_t vi = 0; vi < n_vec; ++vi) {
            size_t c = vi * vl;
            svbool_t pg = svwhilelt_b32((unsigned)c, (unsigned)channels);
            svfloat32_t acc = svld1_f32(pg, state + c);
            for (size_t t = 0; t < seq; ++t) {
                svfloat32_t gv = svld1_f32(pg, g + t * channels + c);
                svfloat32_t xv = svld1_f32(pg, x + t * channels + c);
                acc = svmla_f32_x(pg, xv, acc, gv); /* acc = x + acc*g, one FMA */
                svst1_f32(pg, s + t * channels + c, acc);
            }
            svst1_f32(pg, state + c, acc);
        }
    }
#elif defined(__ARM_NEON)
    /* Double-width unroll: 8 channels/iter (2 NEON register groups). */
    size_t n_vec8 = channels >> 3;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t vi = 0; vi < n_vec8; ++vi) {
        size_t c = vi << 3;
        float32x4_t acc0 = vld1q_f32(state + c);
        float32x4_t acc1 = vld1q_f32(state + c + 4);
        for (size_t t = 0; t < seq; ++t) {
            acc0 = vfmaq_f32(vld1q_f32(x + t * channels + c), acc0,
                             vld1q_f32(g + t * channels + c));
            acc1 = vfmaq_f32(vld1q_f32(x + t * channels + c + 4), acc1,
                             vld1q_f32(g + t * channels + c + 4));
            vst1q_f32(s + t * channels + c, acc0);
            vst1q_f32(s + t * channels + c + 4, acc1);
        }
        vst1q_f32(state + c, acc0);
        vst1q_f32(state + c + 4, acc1);
    }
    /* Remaining groups of 4 (at most 1) */
    for (size_t c = n_vec8 << 3; c + 4 <= channels; c += 4) {
        float32x4_t acc = vld1q_f32(state + c);
        for (size_t t = 0; t < seq; ++t) {
            acc = vfmaq_f32(vld1q_f32(x + t * channels + c), acc,
                            vld1q_f32(g + t * channels + c));
            vst1q_f32(s + t * channels + c, acc);
        }
        vst1q_f32(state + c, acc);
    }
    for (size_t c = (n_vec8 << 3) + (((channels >> 2) & 1) << 2); c < channels; ++c) {
        float a2 = state[c];
        for (size_t t = 0; t < seq; ++t) {
            a2 = x[t * channels + c] + a2 * g[t * channels + c];
            s[t * channels + c] = a2;
        }
        state[c] = a2;
    }
#else
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t c = 0; c < channels; ++c) {
        float acc = state[c];
        for (size_t t = 0; t < seq; ++t) {
            acc = x[t * channels + c] + acc * g[t * channels + c];
            s[t * channels + c] = acc;
        }
        state[c] = acc;
    }
#endif
}

/* ---------------------------------------------------------------------------
 * 3. Causal depthwise Conv1D, kernel width 4  (KleidiAI's equivalent is SME2-only)
 *
 *   out[t][c] = sum_{j=0..K-1} w[j][c] * in[t-(K-1)+j][c]
 *
 * Causal by construction: output t reads no input beyond t. hist[] holds the K-1
 * previous timesteps per channel so decode can advance one token at a time -- the
 * conv-state analogue of a KV cache, 3 * channels floats.
 * ------------------------------------------------------------------------- */
#define GDN_CONV_K 4

void gdn_causal_dwconv1d_f32(const float *restrict in, const float *restrict w,
                             float *restrict out, float *restrict hist, size_t seq,
                             size_t channels) {
#ifdef __ARM_FEATURE_SVE
    {
        unsigned vl = (unsigned)svcntw();
        size_t n_vec = (channels + vl - 1) / vl;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (size_t vi = 0; vi < n_vec; ++vi) {
            size_t c = vi * vl;
            svbool_t pg = svwhilelt_b32((unsigned)c, (unsigned)channels);
            svfloat32_t h0 = svld1_f32(pg, hist + 0 * channels + c);
            svfloat32_t h1 = svld1_f32(pg, hist + 1 * channels + c);
            svfloat32_t h2 = svld1_f32(pg, hist + 2 * channels + c);
            svfloat32_t w0 = svld1_f32(pg, w + 0 * channels + c);
            svfloat32_t w1 = svld1_f32(pg, w + 1 * channels + c);
            svfloat32_t w2 = svld1_f32(pg, w + 2 * channels + c);
            svfloat32_t w3 = svld1_f32(pg, w + 3 * channels + c);
            for (size_t t = 0; t < seq; ++t) {
                svfloat32_t cur = svld1_f32(pg, in + t * channels + c);
                svfloat32_t acc = svmul_f32_x(pg, h0, w0);
                acc = svmla_f32_x(pg, acc, h1, w1);
                acc = svmla_f32_x(pg, acc, h2, w2);
                acc = svmla_f32_x(pg, acc, cur, w3);
                svst1_f32(pg, out + t * channels + c, acc);
                h0 = h1;
                h1 = h2;
                h2 = cur;
            }
            svst1_f32(pg, hist + 0 * channels + c, h0);
            svst1_f32(pg, hist + 1 * channels + c, h1);
            svst1_f32(pg, hist + 2 * channels + c, h2);
        }
    }
#elif defined(__ARM_NEON)
    /* Double-width unroll: 8 channels/iter (2 NEON register groups).
     * The conv's 4-deep FMA chain (acc = h0*w0 + h1*w1 + h2*w2 + cur*w3)
     * benefits MORE from ILP than gated_scan's single FMA: two independent
     * chains let the OoO scheduler hide the 4-cycle FMA latency. */
    size_t n_vec8 = channels >> 3;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t vi = 0; vi < n_vec8; ++vi) {
        size_t c = vi << 3;
        float32x4_t h0_0 = vld1q_f32(hist + 0 * channels + c);
        float32x4_t h0_1 = vld1q_f32(hist + 0 * channels + c + 4);
        float32x4_t h1_0 = vld1q_f32(hist + 1 * channels + c);
        float32x4_t h1_1 = vld1q_f32(hist + 1 * channels + c + 4);
        float32x4_t h2_0 = vld1q_f32(hist + 2 * channels + c);
        float32x4_t h2_1 = vld1q_f32(hist + 2 * channels + c + 4);
        float32x4_t w0_0 = vld1q_f32(w + 0 * channels + c);
        float32x4_t w0_1 = vld1q_f32(w + 0 * channels + c + 4);
        float32x4_t w1_0 = vld1q_f32(w + 1 * channels + c);
        float32x4_t w1_1 = vld1q_f32(w + 1 * channels + c + 4);
        float32x4_t w2_0 = vld1q_f32(w + 2 * channels + c);
        float32x4_t w2_1 = vld1q_f32(w + 2 * channels + c + 4);
        float32x4_t w3_0 = vld1q_f32(w + 3 * channels + c);
        float32x4_t w3_1 = vld1q_f32(w + 3 * channels + c + 4);
        for (size_t t = 0; t < seq; ++t) {
            float32x4_t cur0 = vld1q_f32(in + t * channels + c);
            float32x4_t cur1 = vld1q_f32(in + t * channels + c + 4);
            float32x4_t acc0 = vmulq_f32(h0_0, w0_0);
            float32x4_t acc1 = vmulq_f32(h0_1, w0_1);
            acc0 = vfmaq_f32(acc0, h1_0, w1_0);
            acc1 = vfmaq_f32(acc1, h1_1, w1_1);
            acc0 = vfmaq_f32(acc0, h2_0, w2_0);
            acc1 = vfmaq_f32(acc1, h2_1, w2_1);
            acc0 = vfmaq_f32(acc0, cur0, w3_0);
            acc1 = vfmaq_f32(acc1, cur1, w3_1);
            vst1q_f32(out + t * channels + c, acc0);
            vst1q_f32(out + t * channels + c + 4, acc1);
            h0_0 = h1_0; h1_0 = h2_0; h2_0 = cur0;
            h0_1 = h1_1; h1_1 = h2_1; h2_1 = cur1;
        }
        vst1q_f32(hist + 0 * channels + c, h0_0);
        vst1q_f32(hist + 0 * channels + c + 4, h0_1);
        vst1q_f32(hist + 1 * channels + c, h1_0);
        vst1q_f32(hist + 1 * channels + c + 4, h1_1);
        vst1q_f32(hist + 2 * channels + c, h2_0);
        vst1q_f32(hist + 2 * channels + c + 4, h2_1);
    }
    /* Remaining groups of 4 (at most 1) */
    for (size_t c = n_vec8 << 3; c + 4 <= channels; c += 4) {
        float32x4_t h0 = vld1q_f32(hist + 0 * channels + c);
        float32x4_t h1 = vld1q_f32(hist + 1 * channels + c);
        float32x4_t h2 = vld1q_f32(hist + 2 * channels + c);
        float32x4_t w0 = vld1q_f32(w + 0 * channels + c);
        float32x4_t w1 = vld1q_f32(w + 1 * channels + c);
        float32x4_t w2 = vld1q_f32(w + 2 * channels + c);
        float32x4_t w3 = vld1q_f32(w + 3 * channels + c);
        for (size_t t = 0; t < seq; ++t) {
            float32x4_t cur = vld1q_f32(in + t * channels + c);
            float32x4_t acc = vmulq_f32(h0, w0);
            acc = vfmaq_f32(acc, h1, w1);
            acc = vfmaq_f32(acc, h2, w2);
            acc = vfmaq_f32(acc, cur, w3);
            vst1q_f32(out + t * channels + c, acc);
            h0 = h1;
            h1 = h2;
            h2 = cur;
        }
        vst1q_f32(hist + 0 * channels + c, h0);
        vst1q_f32(hist + 1 * channels + c, h1);
        vst1q_f32(hist + 2 * channels + c, h2);
    }
    for (size_t c = (n_vec8 << 3) + (((channels >> 2) & 1) << 2); c < channels; ++c) {
        float hh[GDN_CONV_K - 1];
        for (int j = 0; j < GDN_CONV_K - 1; ++j) hh[j] = hist[(size_t)j * channels + c];
        for (size_t t = 0; t < seq; ++t) {
            float cur = in[t * channels + c];
            float acc = hh[0] * w[0 * channels + c] + hh[1] * w[1 * channels + c] +
                        hh[2] * w[2 * channels + c] + cur * w[3 * channels + c];
            out[t * channels + c] = acc;
            hh[0] = hh[1];
            hh[1] = hh[2];
            hh[2] = cur;
        }
        for (int j = 0; j < GDN_CONV_K - 1; ++j) hist[(size_t)j * channels + c] = hh[j];
    }
#else
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t c = 0; c < channels; ++c) {
        float h[GDN_CONV_K - 1];
        for (int j = 0; j < GDN_CONV_K - 1; ++j) h[j] = hist[(size_t)j * channels + c];
        for (size_t t = 0; t < seq; ++t) {
            float cur = in[t * channels + c];
            float acc = h[0] * w[0 * channels + c] + h[1] * w[1 * channels + c] +
                        h[2] * w[2 * channels + c] + cur * w[3 * channels + c];
            out[t * channels + c] = acc;
            h[0] = h[1];
            h[1] = h[2];
            h[2] = cur;
        }
        for (int j = 0; j < GDN_CONV_K - 1; ++j) hist[(size_t)j * channels + c] = h[j];
    }
#endif
}

/* ===========================================================================
 * 4a. fp16-state gated scan  (mixed precision: fp16 storage, fp32 accumulate)
 *
 *   s[t][c] = g[t][c] * s[t-1][c] + x[t][c]
 *
 * Identical arithmetic to gdn_gated_scan_f32; only the state load/store is
 * narrowed to fp16.  The fp32 accumulator runs in registers across the entire
 * seq loop, so the narrowing cost is O(channels), not O(channels × seq).
 * ==========================================================================*/
void gdn_gated_scan_f16(const float *restrict g, const float *restrict x,
                        float *restrict s, __fp16 *restrict state, size_t seq,
                        size_t channels) {
#ifdef __ARM_FEATURE_SVE
    {
        unsigned vl = (unsigned)svcntw();
        size_t n_vec = (channels + vl - 1) / vl;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (size_t vi = 0; vi < n_vec; ++vi) {
            size_t c = vi * vl;
            svbool_t pg = svwhilelt_b32((unsigned)c, (unsigned)channels);
            float tmp[64];
            for (unsigned j = 0; j < vl && c + j < channels; ++j)
                tmp[j] = (float)state[c + j];
            svfloat32_t acc = svld1_f32(pg, tmp);
            for (size_t t = 0; t < seq; ++t) {
                svfloat32_t gv = svld1_f32(pg, g + t * channels + c);
                svfloat32_t xv = svld1_f32(pg, x + t * channels + c);
                acc = svmla_f32_x(pg, xv, acc, gv);
                svst1_f32(pg, s + t * channels + c, acc);
            }
            svst1_f32(pg, tmp, acc);
            for (unsigned j = 0; j < vl && c + j < channels; ++j)
                state[c + j] = (__fp16)tmp[j];
        }
    }
#elif defined(__ARM_NEON)
    /* Double-width unroll: 8 channels/iter. */
    size_t n_vec8 = channels >> 3;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t vi = 0; vi < n_vec8; ++vi) {
        size_t c = vi << 3;
        float16x4_t s16_0 = vld1_f16(state + c);
        float16x4_t s16_1 = vld1_f16(state + c + 4);
        float32x4_t acc0 = vcvt_f32_f16(s16_0);
        float32x4_t acc1 = vcvt_f32_f16(s16_1);
        for (size_t t = 0; t < seq; ++t) {
            acc0 = vfmaq_f32(vld1q_f32(x + t * channels + c), acc0,
                             vld1q_f32(g + t * channels + c));
            acc1 = vfmaq_f32(vld1q_f32(x + t * channels + c + 4), acc1,
                             vld1q_f32(g + t * channels + c + 4));
            vst1q_f32(s + t * channels + c, acc0);
            vst1q_f32(s + t * channels + c + 4, acc1);
        }
        vst1_f16(state + c, vcvt_f16_f32(acc0));
        vst1_f16(state + c + 4, vcvt_f16_f32(acc1));
    }
    for (size_t c = n_vec8 << 3; c + 4 <= channels; c += 4) {
        float16x4_t s16 = vld1_f16(state + c);
        float32x4_t acc = vcvt_f32_f16(s16);
        for (size_t t = 0; t < seq; ++t) {
            acc = vfmaq_f32(vld1q_f32(x + t * channels + c), acc,
                            vld1q_f32(g + t * channels + c));
            vst1q_f32(s + t * channels + c, acc);
        }
        vst1_f16(state + c, vcvt_f16_f32(acc));
    }
    for (size_t c = (n_vec8 << 3) + (((channels >> 2) & 1) << 2); c < channels; ++c) {
        float a = (float)state[c];
        for (size_t t = 0; t < seq; ++t) {
            a = x[t * channels + c] + a * g[t * channels + c];
            s[t * channels + c] = a;
        }
        state[c] = (__fp16)a;
    }
#else
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t c = 0; c < channels; ++c) {
        float acc = (float)state[c];
        for (size_t t = 0; t < seq; ++t) {
            acc = x[t * channels + c] + acc * g[t * channels + c];
            s[t * channels + c] = acc;
        }
        state[c] = (__fp16)acc;
    }
#endif
}

/* ===========================================================================
 * 4b. fp16-output cumulative decay  (mixed precision: fp16 output, fp32 accum)
 *
 *   decay[t][c] = prod_{i<=t} a[i][c]
 *
 * The running product accumulates in fp32 registers.  Only the STORE to the
 * decay[] array is narrowed to fp16, halving its memory footprint and the
 * bandwidth of the subsequent read by the delta-rule consumer.
 * ==========================================================================*/
void gdn_cumdecay_f16(const float *restrict a, __fp16 *restrict decay, size_t seq,
                      size_t channels) {
#ifdef __ARM_FEATURE_SVE
    {
        unsigned vl = (unsigned)svcntw();
        size_t n_vec = (channels + vl - 1) / vl;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (size_t vi = 0; vi < n_vec; ++vi) {
            size_t c = vi * vl;
            svbool_t pg = svwhilelt_b32((unsigned)c, (unsigned)channels);
            svfloat32_t run = svdup_f32(1.0f);
            for (size_t t = 0; t < seq; ++t) {
                svfloat32_t av = svld1_f32(pg, a + t * channels + c);
                run = svmul_f32_x(pg, run, av);
                float tmp[64];
                svst1_f32(pg, tmp, run);
                for (unsigned j = 0; j < vl && c + j < channels; ++j)
                    decay[t * channels + c + j] = (__fp16)tmp[j];
            }
        }
    }
#elif defined(__ARM_NEON)
    /* Double-width unroll: 8 channels/iter. */
    size_t n_vec8 = channels >> 3;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t vi = 0; vi < n_vec8; ++vi) {
        size_t c = vi << 3;
        float32x4_t run0 = vdupq_n_f32(1.0f);
        float32x4_t run1 = vdupq_n_f32(1.0f);
        for (size_t t = 0; t < seq; ++t) {
            run0 = vmulq_f32(run0, vld1q_f32(a + t * channels + c));
            run1 = vmulq_f32(run1, vld1q_f32(a + t * channels + c + 4));
            vst1_f16(decay + t * channels + c, vcvt_f16_f32(run0));
            vst1_f16(decay + t * channels + c + 4, vcvt_f16_f32(run1));
        }
    }
    for (size_t c = n_vec8 << 3; c + 4 <= channels; c += 4) {
        float32x4_t run = vdupq_n_f32(1.0f);
        for (size_t t = 0; t < seq; ++t) {
            run = vmulq_f32(run, vld1q_f32(a + t * channels + c));
            vst1_f16(decay + t * channels + c, vcvt_f16_f32(run));
        }
    }
    for (size_t c = (n_vec8 << 3) + (((channels >> 2) & 1) << 2); c < channels; ++c) {
        float run = 1.0f;
        for (size_t t = 0; t < seq; ++t) {
            run *= a[t * channels + c];
            decay[t * channels + c] = (__fp16)run;
        }
    }
#else
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t c = 0; c < channels; ++c) {
        float run = 1.0f;
        for (size_t t = 0; t < seq; ++t) {
            run *= a[t * channels + c];
            decay[t * channels + c] = (__fp16)run;
        }
    }
#endif
}

/* ===========================================================================
 * 5a. bf16-state gated scan  (mixed precision: bf16 storage, fp32 accumulate)
 *
 * Same structure as the fp16 variant, but state is stored as bf16 (uint16_t).
 * bf16 has the same exponent range as fp32 (8 bits) but only 7 mantissa bits,
 * so it preserves range at the cost of precision — the opposite trade-off from
 * fp16 (5 exp / 10 mantissa).  For GDN state in [−1, 1], fp16's extra mantissa
 * precision is usually preferable, but we implement both and let measurement
 * decide.
 *
 * Uses software bf16 conversion (f32_to_bf16_rne / bf16_to_f32) so it runs on
 * any Armv8-A core without hardware bf16 support.
 * ==========================================================================*/
void gdn_gated_scan_bf16(const float *restrict g, const float *restrict x,
                         float *restrict s, uint16_t *restrict state, size_t seq,
                         size_t channels) {
#ifdef __ARM_FEATURE_SVE
    {
        unsigned vl = (unsigned)svcntw();
        size_t n_vec = (channels + vl - 1) / vl;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (size_t vi = 0; vi < n_vec; ++vi) {
            size_t c = vi * vl;
            svbool_t pg = svwhilelt_b32((unsigned)c, (unsigned)channels);
            float tmp[64];
            for (unsigned j = 0; j < vl && c + j < channels; ++j)
                tmp[j] = bf16_to_f32(state[c + j]);
            svfloat32_t acc = svld1_f32(pg, tmp);
            for (size_t t = 0; t < seq; ++t) {
                svfloat32_t gv = svld1_f32(pg, g + t * channels + c);
                svfloat32_t xv = svld1_f32(pg, x + t * channels + c);
                acc = svmla_f32_x(pg, xv, acc, gv);
                svst1_f32(pg, s + t * channels + c, acc);
            }
            svst1_f32(pg, tmp, acc);
            for (unsigned j = 0; j < vl && c + j < channels; ++j)
                state[c + j] = f32_to_bf16_rne(tmp[j]);
        }
    }
#elif defined(__ARM_NEON)
    /* Double-width unroll: 8 channels/iter. */
    size_t n_vec8 = channels >> 3;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t vi = 0; vi < n_vec8; ++vi) {
        size_t c = vi << 3;
        float32x4_t acc0 = vcvtq_f32_from_bf16(vld1_u16(state + c));
        float32x4_t acc1 = vcvtq_f32_from_bf16(vld1_u16(state + c + 4));
        for (size_t t = 0; t < seq; ++t) {
            acc0 = vfmaq_f32(vld1q_f32(x + t * channels + c), acc0,
                             vld1q_f32(g + t * channels + c));
            acc1 = vfmaq_f32(vld1q_f32(x + t * channels + c + 4), acc1,
                             vld1q_f32(g + t * channels + c + 4));
            vst1q_f32(s + t * channels + c, acc0);
            vst1q_f32(s + t * channels + c + 4, acc1);
        }
        vst1_u16(state + c, vcvtq_bf16_from_f32(acc0));
        vst1_u16(state + c + 4, vcvtq_bf16_from_f32(acc1));
    }
    for (size_t c = n_vec8 << 3; c + 4 <= channels; c += 4) {
        float32x4_t acc = vcvtq_f32_from_bf16(vld1_u16(state + c));
        for (size_t t = 0; t < seq; ++t) {
            acc = vfmaq_f32(vld1q_f32(x + t * channels + c), acc,
                            vld1q_f32(g + t * channels + c));
            vst1q_f32(s + t * channels + c, acc);
        }
        vst1_u16(state + c, vcvtq_bf16_from_f32(acc));
    }
    for (size_t c = (n_vec8 << 3) + (((channels >> 2) & 1) << 2); c < channels; ++c) {
        float a = bf16_to_f32(state[c]);
        for (size_t t = 0; t < seq; ++t) {
            a = x[t * channels + c] + a * g[t * channels + c];
            s[t * channels + c] = a;
        }
        state[c] = f32_to_bf16_rne(a);
    }
#else
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t c = 0; c < channels; ++c) {
        float acc = bf16_to_f32(state[c]);
        for (size_t t = 0; t < seq; ++t) {
            acc = x[t * channels + c] + acc * g[t * channels + c];
            s[t * channels + c] = acc;
        }
        state[c] = f32_to_bf16_rne(acc);
    }
#endif
}

/* ===========================================================================
 * 5b. bf16-output cumulative decay  (mixed precision: bf16 output, fp32 accum)
 * ==========================================================================*/
void gdn_cumdecay_bf16(const float *restrict a, uint16_t *restrict decay, size_t seq,
                       size_t channels) {
#ifdef __ARM_FEATURE_SVE
    {
        unsigned vl = (unsigned)svcntw();
        size_t n_vec = (channels + vl - 1) / vl;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (size_t vi = 0; vi < n_vec; ++vi) {
            size_t c = vi * vl;
            svbool_t pg = svwhilelt_b32((unsigned)c, (unsigned)channels);
            svfloat32_t run = svdup_f32(1.0f);
            for (size_t t = 0; t < seq; ++t) {
                svfloat32_t av = svld1_f32(pg, a + t * channels + c);
                run = svmul_f32_x(pg, run, av);
                float tmp[64];
                svst1_f32(pg, tmp, run);
                for (unsigned j = 0; j < vl && c + j < channels; ++j)
                    decay[t * channels + c + j] = f32_to_bf16_rne(tmp[j]);
            }
        }
    }
#elif defined(__ARM_NEON)
    /* Double-width unroll: 8 channels/iter. */
    size_t n_vec8 = channels >> 3;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t vi = 0; vi < n_vec8; ++vi) {
        size_t c = vi << 3;
        float32x4_t run0 = vdupq_n_f32(1.0f);
        float32x4_t run1 = vdupq_n_f32(1.0f);
        for (size_t t = 0; t < seq; ++t) {
            run0 = vmulq_f32(run0, vld1q_f32(a + t * channels + c));
            run1 = vmulq_f32(run1, vld1q_f32(a + t * channels + c + 4));
            vst1_u16(decay + t * channels + c, vcvtq_bf16_from_f32(run0));
            vst1_u16(decay + t * channels + c + 4, vcvtq_bf16_from_f32(run1));
        }
    }
    for (size_t c = n_vec8 << 3; c + 4 <= channels; c += 4) {
        float32x4_t run = vdupq_n_f32(1.0f);
        for (size_t t = 0; t < seq; ++t) {
            run = vmulq_f32(run, vld1q_f32(a + t * channels + c));
            vst1_u16(decay + t * channels + c, vcvtq_bf16_from_f32(run));
        }
    }
    for (size_t c = (n_vec8 << 3) + (((channels >> 2) & 1) << 2); c < channels; ++c) {
        float run = 1.0f;
        for (size_t t = 0; t < seq; ++t) {
            run *= a[t * channels + c];
            decay[t * channels + c] = f32_to_bf16_rne(run);
        }
    }
#else
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t c = 0; c < channels; ++c) {
        float run = 1.0f;
        for (size_t t = 0; t < seq; ++t) {
            run *= a[t * channels + c];
            decay[t * channels + c] = f32_to_bf16_rne(run);
        }
    }
#endif
}

/* ===========================================================================
 * GDN-2 decoupled-gating scan (bead ob-y3f)
 *
 * GDN-2 replaces GDN-1's single scalar gate with two channel-wise gates:
 *   - b[t][c]: erase gate (key axis) — modulates the decay
 *   - w[t][c]: write gate (value axis) — modulates the input
 *
 * Channel-wise recurrence (our abstraction level for microbenchmarking):
 *   GDN-1: s[t] = x[t] + g[t] * s[t-1]              (1 FMA/element)
 *   GDN-2: s[t] = w[t]*x[t] + g[t]*b[t]*s[t-1]      (2 extra muls/element)
 *
 * The extra cost is 2 elementwise multiplies per timestep: one for w*x,
 * one for g*b before the FMA. On a bandwidth-bound kernel (which these are,
 * per FINDINGS.md §5a), the dominant overhead is the 2 extra streams (b, w),
 * not the arithmetic.
 *
 * See NVLabs/GatedDeltaNet-2 fused_recurrent_gdn2.py for the full matrix
 * recurrence; this is the channel-wise projection for like-for-like comparison.
 * ==========================================================================*/
void gdn2_gated_scan_f32(const float *restrict g, const float *restrict b_gate,
                         const float *restrict w_gate, const float *restrict x,
                         float *restrict s, float *restrict state,
                         size_t seq, size_t channels) {
#ifdef __ARM_FEATURE_SVE
    {
        unsigned vl = (unsigned)svcntw();
        size_t n_vec = (channels + vl - 1) / vl;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (size_t vi = 0; vi < n_vec; ++vi) {
            size_t c = vi * vl;
            svbool_t pg = svwhilelt_b32((unsigned)c, (unsigned)channels);
            svfloat32_t acc = svld1_f32(pg, state + c);
            for (size_t t = 0; t < seq; ++t) {
                size_t off = t * channels + c;
                svfloat32_t gv = svld1_f32(pg, g + off);
                svfloat32_t bv = svld1_f32(pg, b_gate + off);
                svfloat32_t wv = svld1_f32(pg, w_gate + off);
                svfloat32_t xv = svld1_f32(pg, x + off);
                svfloat32_t gb = svmul_f32_x(pg, gv, bv);       /* g*b */
                svfloat32_t wx = svmul_f32_x(pg, wv, xv);        /* w*x */
                acc = svmla_f32_x(pg, wx, acc, gb);              /* acc = w*x + acc*(g*b) */
                svst1_f32(pg, s + off, acc);
            }
            svst1_f32(pg, state + c, acc);
        }
    }
#elif defined(__ARM_NEON)
    size_t n_vec8 = channels >> 3;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t vi = 0; vi < n_vec8; ++vi) {
        size_t c = vi << 3;
        float32x4_t acc0 = vld1q_f32(state + c);
        float32x4_t acc1 = vld1q_f32(state + c + 4);
        for (size_t t = 0; t < seq; ++t) {
            size_t off = t * channels + c;
            float32x4_t g0 = vld1q_f32(g + off);
            float32x4_t g1 = vld1q_f32(g + off + 4);
            float32x4_t b0 = vmulq_f32(g0, vld1q_f32(b_gate + off));
            float32x4_t b1 = vmulq_f32(g1, vld1q_f32(b_gate + off + 4));
            float32x4_t w0 = vmulq_f32(vld1q_f32(w_gate + off), vld1q_f32(x + off));
            float32x4_t w1 = vmulq_f32(vld1q_f32(w_gate + off + 4), vld1q_f32(x + off + 4));
            acc0 = vfmaq_f32(w0, acc0, b0);
            acc1 = vfmaq_f32(w1, acc1, b1);
            vst1q_f32(s + off, acc0);
            vst1q_f32(s + off + 4, acc1);
        }
        vst1q_f32(state + c, acc0);
        vst1q_f32(state + c + 4, acc1);
    }
    for (size_t c = n_vec8 << 3; c + 4 <= channels; c += 4) {
        float32x4_t acc = vld1q_f32(state + c);
        for (size_t t = 0; t < seq; ++t) {
            size_t off = t * channels + c;
            float32x4_t gb = vmulq_f32(vld1q_f32(g + off), vld1q_f32(b_gate + off));
            float32x4_t wx = vmulq_f32(vld1q_f32(w_gate + off), vld1q_f32(x + off));
            acc = vfmaq_f32(wx, acc, gb);
            vst1q_f32(s + off, acc);
        }
        vst1q_f32(state + c, acc);
    }
    for (size_t c = (n_vec8 << 3) + (((channels >> 2) & 1) << 2); c < channels; ++c) {
        float a2 = state[c];
        for (size_t t = 0; t < seq; ++t) {
            size_t off = t * channels + c;
            a2 = w_gate[off] * x[off] + a2 * (g[off] * b_gate[off]);
            s[off] = a2;
        }
        state[c] = a2;
    }
#else
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t c = 0; c < channels; ++c) {
        float acc = state[c];
        for (size_t t = 0; t < seq; ++t) {
            size_t off = t * channels + c;
            acc = w_gate[off] * x[off] + acc * (g[off] * b_gate[off]);
            s[off] = acc;
        }
        state[c] = acc;
    }
#endif
}
