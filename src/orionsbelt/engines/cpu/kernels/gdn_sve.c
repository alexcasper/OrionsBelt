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
#include <stdint.h>
#include <string.h>

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

/* ===========================================================================
 * MIXED-PRECISION STATE VARIANTS (bead ob-8qt.4)
 *
 * The recurrent state is the only persistent data structure in GDN decode.
 * Narrowing it from fp32 to bf16 or fp16 halves memory footprint and I/O
 * traffic for state load/store — the two streams that execute on every
 * decode step. At seq=1 (decode), state I/O is ~40% of total traffic; halving
 * it saves ~20% per-token. At seq=64 (prefill), state is ~1% of traffic, so
 * the benefit is concentrated at decode — exactly where bandwidth matters most.
 *
 * CRITICAL CONSTRAINT: all accumulation stays fp32. A decay product of 0.5^64
 * ≈ 5e-20 underflows both bf16 and fp16. Only the *storage format* changes;
 * the *arithmetic format* does not. Mixed precision (narrow state, wide
 * accumulate), not uniform narrowing.
 *
 * API: narrow-format arrays use uint16_t* to be unambiguous about the on-wire
 * representation, independent of compiler __bf16/__fp16 type support.
 * =========================================================================== */

/* --- bf16 conversion (software, portable, no ISA dependency) ---
 *
 * bf16 shares fp32's 8-bit exponent (same range) with only 7 mantissa bits
 * (vs fp32's 23). Conversion is round-to-nearest-even on the low 16 bits.
 * No overflow/underflow handling needed — bf16 covers the same exponent range.
 */
static inline uint16_t f32_to_bf16_sw(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    uint32_t lsb = (bits >> 16) & 1;
    uint32_t rounding_bias = 0x7FFFu + lsb;
    return (uint16_t)((bits + rounding_bias) >> 16);
}

static inline float bf16_to_f32_sw(uint16_t b) {
    uint32_t bits = (uint32_t)b << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

/* --- fp16 (IEEE half) conversion ---
 *
 * fp16 has 5-bit exponent (range ±65504) and 10-bit mantissa. On AArch64
 * the scalar __fp16 type generates a single FCVT instruction (base ISA,
 * mandatory on all cores including Cortex-A57 at Armv8.0-A). NEON FCVTN/FCVTL
 * are also base A64 ASIMD. The __ARM_FEATURE_FP16_VECTOR_ARITHMETIC macro
 * is for fp16 *arithmetic* (fmul/fadd on half regs), which is separate from
 * conversion and not needed here.
 */
static inline uint16_t f32_to_f16_sw(float f) {
    __fp16 h = (__fp16)f;
    uint16_t bits;
    memcpy(&bits, &h, sizeof(bits));
    return bits;
}

static inline float f16_to_f32_sw(uint16_t h) {
    __fp16 hp;
    memcpy(&hp, &h, sizeof(hp));
    return (float)hp;
}

#ifdef __ARM_NEON
/* --- NEON 4-lane bf16 helpers (software conversion in vector registers) ---
 *
 * bf16 has no NEON instruction support until Armv8.6-A (__ARM_FEATURE_BF16).
 * These helpers do the widening/narrowing using integer NEON ops, which are
 * available on every AArch64 core. The FMA loop in between runs in fp32. */
static inline float32x4_t vld1q_bf16_to_f32(const uint16_t *p) {
    uint16x4_t b = vld1_u16(p);
    uint32x4_t expanded = vshll_n_u16(b, 16);   /* zero-extend: bf16 → fp32 bit pattern */
    return vreinterpretq_f32_u32(expanded);
}

static inline void vst1q_f32_to_bf16(float32x4_t v, uint16_t *p) {
    uint32x4_t bits = vreinterpretq_u32_f32(v);
    /* Round-to-nearest-even: add bias of 0x7FFF plus the bit that determines rounding direction */
    uint32x4_t lsb = vandq_u32(vshrq_n_u32(bits, 16), vdupq_n_u32(1));
    uint32x4_t rounding = vaddq_u32(vdupq_n_u32(0x7FFF), lsb);
    uint32x4_t rounded = vaddq_u32(bits, rounding);
    uint16x4_t result = vmovn_u32(vshrq_n_u32(rounded, 16));
    vst1_u16(p, result);
}

/* --- NEON 4-lane fp16 helpers (native FCVTN/FCVTL) --- */
static inline float32x4_t vld1q_f16_to_f32(const uint16_t *p) {
    float16x4_t h = vreinterpret_f16_u16(vld1_u16(p));
    return vcvt_f32_f16(h);                      /* FCVTL: widen 4 × fp16 → 4 × fp32 */
}

static inline void vst1q_f32_to_f16(float32x4_t v, uint16_t *p) {
    float16x4_t h = vcvt_f16_f32(v);             /* FCVTN: narrow 4 × fp32 → 4 × fp16 */
    vst1_u16(p, vreinterpret_u16_f16(h));
}
#endif /* __ARM_NEON */

/* ---------------------------------------------------------------------------
 * 1b. Gated cumulative decay — bf16 output variant
 *
 *   decay[t][c] = prod_{i<=t} a[i][c], stored as bf16
 *
 * Identical to gdn_cumdecay_f32 except decay[] is written as bf16 (uint16_t),
 * halving output write traffic and downstream read traffic. The running
 * product accumulator stays fp32 throughout. Input a[] remains fp32.
 * Bytes: sizeof(float)*seq*ch (read a) + sizeof(uint16_t)*seq*ch (write decay)
 *      = 6*seq*ch, vs 8*seq*ch for the fp32 variant — 25% less I/O.
 * ------------------------------------------------------------------------- */
void gdn_cumdecay_bf16(const float *restrict a, uint16_t *restrict decay,
                       size_t seq, size_t channels) {
#if defined(__ARM_NEON)
    size_t c = 0;
    for (; c + 4 <= channels; c += 4) {
        float32x4_t run = vdupq_n_f32(1.0f);
        for (size_t t = 0; t < seq; ++t) {
            run = vmulq_f32(run, vld1q_f32(a + t * channels + c));
            vst1q_f32_to_bf16(run, decay + t * channels + c);
        }
    }
    for (; c < channels; ++c) {
        float run = 1.0f;
        for (size_t t = 0; t < seq; ++t) {
            run *= a[t * channels + c];
            decay[t * channels + c] = f32_to_bf16_sw(run);
        }
    }
#else
    for (size_t c = 0; c < channels; ++c) {
        float run = 1.0f;
        for (size_t t = 0; t < seq; ++t) {
            run *= a[t * channels + c];
            decay[t * channels + c] = f32_to_bf16_sw(run);
        }
    }
#endif
}

/* ---------------------------------------------------------------------------
 * 1c. Gated cumulative decay — fp16 output variant
 *
 * Same as bf16 variant but with IEEE half-precision output. Note: fp16's
 * 5-bit exponent means values below ~6e-5 flush to zero. For decay products
 * with gates in (0.9, 0.99) over 64 steps this is not an issue (min ~0.001),
 * but if gate values are smaller (e.g. 0.5), the cumulative product underflows
 * fp16. The accumulator is fp32 regardless.
 * ------------------------------------------------------------------------- */
void gdn_cumdecay_f16(const float *restrict a, uint16_t *restrict decay,
                      size_t seq, size_t channels) {
#if defined(__ARM_NEON)
    size_t c = 0;
    for (; c + 4 <= channels; c += 4) {
        float32x4_t run = vdupq_n_f32(1.0f);
        for (size_t t = 0; t < seq; ++t) {
            run = vmulq_f32(run, vld1q_f32(a + t * channels + c));
            vst1q_f32_to_f16(run, decay + t * channels + c);
        }
    }
    for (; c < channels; ++c) {
        float run = 1.0f;
        for (size_t t = 0; t < seq; ++t) {
            run *= a[t * channels + c];
            decay[t * channels + c] = f32_to_f16_sw(run);
        }
    }
#else
    for (size_t c = 0; c < channels; ++c) {
        float run = 1.0f;
        for (size_t t = 0; t < seq; ++t) {
            run *= a[t * channels + c];
            decay[t * channels + c] = f32_to_f16_sw(run);
        }
    }
#endif
}

/* ---------------------------------------------------------------------------
 * 2b. Chunkwise gated scan — bf16 state variant
 *
 *   s[t][c] = g[t][c] * s[t-1][c] + x[t][c], with state persisted as bf16
 *
 * The persistent state[] array (carried across chunk invocations) is stored
 * as bf16. It is widened to fp32 on load at the start, accumulated in fp32
 * through the entire chunk, and narrowed back to bf16 on store at the end.
 * Per-token inputs (g[], x[]) and per-token outputs (s[]) stay fp32.
 *
 * This is where narrowing matters most: at decode (seq=1), state read+write
 * is 2 of 5 memory streams, so halving it saves ~20% of per-token traffic.
 * The accumulated quantization error per chunk boundary is bounded by bf16's
 * ~0.4% relative error, which compounds over chunks but stays well within the
 * correctness tolerances in docs/METHODOLOGY.md for bf16 state.
 * ------------------------------------------------------------------------- */
void gdn_gated_scan_bf16(const float *restrict g, const float *restrict x,
                         float *restrict s, uint16_t *restrict state,
                         size_t seq, size_t channels) {
#if defined(__ARM_NEON)
    size_t c = 0;
    for (; c + 4 <= channels; c += 4) {
        float32x4_t acc = vld1q_bf16_to_f32(state + c);
        for (size_t t = 0; t < seq; ++t) {
            acc = vfmaq_f32(vld1q_f32(x + t * channels + c), acc,
                            vld1q_f32(g + t * channels + c));
            vst1q_f32(s + t * channels + c, acc);
        }
        vst1q_f32_to_bf16(acc, state + c);
    }
    for (; c < channels; ++c) {
        float a2 = bf16_to_f32_sw(state[c]);
        for (size_t t = 0; t < seq; ++t) {
            a2 = x[t * channels + c] + a2 * g[t * channels + c];
            s[t * channels + c] = a2;
        }
        state[c] = f32_to_bf16_sw(a2);
    }
#else
    for (size_t c = 0; c < channels; ++c) {
        float a2 = bf16_to_f32_sw(state[c]);
        for (size_t t = 0; t < seq; ++t) {
            a2 = x[t * channels + c] + a2 * g[t * channels + c];
            s[t * channels + c] = a2;
        }
        state[c] = f32_to_bf16_sw(a2);
    }
#endif
}

/* ---------------------------------------------------------------------------
 * 2c. Chunkwise gated scan — fp16 state variant
 *
 * Same pattern as bf16 but with IEEE half-precision state. fp16 has higher
 * mantissa precision than bf16 (10 bits vs 7) but narrower exponent range
 * (±65504 vs ±3.4e38). For typical GDN state magnitudes (O(1) to O(100)),
 * fp16's extra mantissa bits give ~2× better accuracy than bf16 at the cost
 * of potential overflow on extreme values. The accumulator is fp32 regardless.
 * ------------------------------------------------------------------------- */
void gdn_gated_scan_f16(const float *restrict g, const float *restrict x,
                        float *restrict s, uint16_t *restrict state,
                        size_t seq, size_t channels) {
#if defined(__ARM_NEON)
    size_t c = 0;
    for (; c + 4 <= channels; c += 4) {
        float32x4_t acc = vld1q_f16_to_f32(state + c);
        for (size_t t = 0; t < seq; ++t) {
            acc = vfmaq_f32(vld1q_f32(x + t * channels + c), acc,
                            vld1q_f32(g + t * channels + c));
            vst1q_f32(s + t * channels + c, acc);
        }
        vst1q_f32_to_f16(acc, state + c);
    }
    for (; c < channels; ++c) {
        float a2 = f16_to_f32_sw(state[c]);
        for (size_t t = 0; t < seq; ++t) {
            a2 = x[t * channels + c] + a2 * g[t * channels + c];
            s[t * channels + c] = a2;
        }
        state[c] = f32_to_f16_sw(a2);
    }
#else
    for (size_t c = 0; c < channels; ++c) {
        float a2 = f16_to_f32_sw(state[c]);
        for (size_t t = 0; t < seq; ++t) {
            a2 = x[t * channels + c] + a2 * g[t * channels + c];
            s[t * channels + c] = a2;
        }
        state[c] = f32_to_f16_sw(a2);
    }
#endif
}
