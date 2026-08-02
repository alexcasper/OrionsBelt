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
 * Build (any of these work):
 *   aarch64-linux-gnu-gcc -O3 -march=armv8.2-a+sve   -c gdn_sve.c   # SVE1 floor
 *   aarch64-linux-gnu-gcc -O3 -march=armv9.2-a+sve2+i8mm+bf16 -c gdn_sve.c
 *   clang --target=aarch64-linux-gnu -O3 -mcpu=cortex-a720 -c gdn_sve.c
 */

#include <stddef.h>

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif
#ifdef __ARM_NEON
#include <arm_neon.h>
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
    for (size_t c = 0; c < channels; c += svcntw()) {
        svbool_t pg = svwhilelt_b32((unsigned)c, (unsigned)channels);
        svfloat32_t run = svdup_f32(1.0f);
        for (size_t t = 0; t < seq; ++t) {
            svfloat32_t av = svld1_f32(pg, a + t * channels + c);
            run = svmul_f32_x(pg, run, av);
            svst1_f32(pg, decay + t * channels + c, run);
        }
    }
#elif defined(__ARM_NEON)
    size_t c = 0;
    for (; c + 4 <= channels; c += 4) {
        float32x4_t run = vdupq_n_f32(1.0f);
        for (size_t t = 0; t < seq; ++t) {
            run = vmulq_f32(run, vld1q_f32(a + t * channels + c));
            vst1q_f32(decay + t * channels + c, run);
        }
    }
    for (; c < channels; ++c) {
        float run = 1.0f;
        for (size_t t = 0; t < seq; ++t) {
            run *= a[t * channels + c];
            decay[t * channels + c] = run;
        }
    }
#else
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
    for (size_t c = 0; c < channels; c += svcntw()) {
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
#elif defined(__ARM_NEON)
    size_t c = 0;
    for (; c + 4 <= channels; c += 4) {
        float32x4_t acc = vld1q_f32(state + c);
        for (size_t t = 0; t < seq; ++t) {
            /* acc = x + acc*g -- vfmaq_f32(addend, a, b) = addend + a*b */
            acc = vfmaq_f32(vld1q_f32(x + t * channels + c), acc,
                            vld1q_f32(g + t * channels + c));
            vst1q_f32(s + t * channels + c, acc);
        }
        vst1q_f32(state + c, acc);
    }
    for (; c < channels; ++c) {
        float a2 = state[c];
        for (size_t t = 0; t < seq; ++t) {
            a2 = x[t * channels + c] + a2 * g[t * channels + c];
            s[t * channels + c] = a2;
        }
        state[c] = a2;
    }
#else
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
    for (size_t c = 0; c < channels; c += svcntw()) {
        svbool_t pg = svwhilelt_b32((unsigned)c, (unsigned)channels);
        svfloat32_t h0 = svld1_f32(pg, hist + 0 * channels + c); /* t-3 */
        svfloat32_t h1 = svld1_f32(pg, hist + 1 * channels + c); /* t-2 */
        svfloat32_t h2 = svld1_f32(pg, hist + 2 * channels + c); /* t-1 */
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
            h0 = h1; /* shift the ring: no cross-lane ops, just register renaming */
            h1 = h2;
            h2 = cur;
        }
        svst1_f32(pg, hist + 0 * channels + c, h0);
        svst1_f32(pg, hist + 1 * channels + c, h1);
        svst1_f32(pg, hist + 2 * channels + c, h2);
    }
#elif defined(__ARM_NEON)
    size_t c = 0;
    for (; c + 4 <= channels; c += 4) {
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
    for (; c < channels; ++c) {
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
