/* Mixed-precision GDN state variants — fp16 and bf16 state with fp32 accumulator.
 *
 * Bead ob-8qt.4. The persistent recurrent state (S matrix and decay buffer) is
 * narrowed to fp16 or bf16, halving resident memory from 48 MiB to 24 MiB across
 * the 24 GDN layers of Qwen3.5-4B. The accumulator is ALWAYS fp32 — a decay of
 * 0.5^64 ≈ 5e-20 underflows fp16 outright.
 *
 * Design: widen on load, compute in fp32, narrow on store. The conversion is
 * O(channels); the scan is O(channels × seq), so for seq ≥ 2 the overhead is
 * < 50% and negligible for prefill (seq=64). The persistent state between calls
 * is fp16/bf16 — that is the memory win.
 *
 * The three existing fp32 kernels (gdn_cumdecay, gdn_gated_scan, gdn_causal_dwconv1d)
 * are unchanged. These functions are the state-narrowed wrappers.
 *
 * Build:
 *   aarch64-linux-gnu-gcc -O3 -march=armv8.6-a+sve2+i8mm+bf16 -c gdn_sve_f16.c
 *   aarch64-linux-gnu-gcc -O3 -march=armv8.2-a+fp16           -c gdn_sve_f16.c
 *   aarch64-linux-gnu-gcc -O3 -march=armv8-a                   -c gdn_sve_f16.c
 */

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

/* The existing fp32 kernels (declared in gdn_sve.c) */
extern void gdn_gated_scan_f32(const float *restrict g, const float *restrict x,
                               float *restrict s, float *restrict state,
                               size_t seq, size_t channels);
extern void gdn_cumdecay_f32(const float *restrict a, float *restrict decay,
                             size_t seq, size_t channels);

/* Maximum SVE vector length in fp32 elements (2048 bits / 32 = 64).
 * Used for stack buffers in the SVE path. Oversized for 128-bit cores
 * (Cortex-A720: 4 elements) but 256 bytes on stack is nothing. */
#define MAX_SVE_F32 64

/* ---------------------------------------------------------------------------
 * bf16 ↔ float conversion helpers.
 *
 * GCC 11 treats __bf16 as a storage-only type and does not allow direct casts
 * to/from float. We use a bit-manipulation approach: bf16 is the upper 16 bits
 * of a float32, so widening is a left-shift by 16 and narrowing is a right-shift
 * with round-to-nearest-even.
 * ------------------------------------------------------------------------- */
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
    /* Round to nearest even: add bias, then handle rounding bit */
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16;
    unsigned short bits = (unsigned short)u;
    __bf16 b;
    memcpy(&b, &bits, sizeof(b));
    return b;
}

/* ===========================================================================
 * 1a. Gated scan — fp16 state, fp32 accumulator
 *
 *   s[t] = g[t] * s[t-1] + x[t]
 *
 * State (persistent across calls) is _Float16. All math is fp32.
 * ===========================================================================*/
void gdn_gated_scan_f16(const float *restrict g, const float *restrict x,
                        float *restrict s, _Float16 *restrict state,
                        size_t seq, size_t channels) {
#ifdef __ARM_FEATURE_SVE
    for (size_t c = 0; c < channels; c += svcntw()) {
        svbool_t pg = svwhilelt_b32((unsigned)c, (unsigned)channels);

        /* Widen fp16 state → fp32 (once per chunk, not per timestep) */
        float buf[MAX_SVE_F32];
        unsigned width = svcntw();
        for (unsigned i = 0; i < width && c + i < channels; i++)
            buf[i] = (float)state[c + i];
        svfloat32_t acc = svld1_f32(pg, buf);

        /* Hot loop — identical fp32 FMA as the pure-fp32 kernel */
        for (size_t t = 0; t < seq; ++t) {
            svfloat32_t gv = svld1_f32(pg, g + t * channels + c);
            svfloat32_t xv = svld1_f32(pg, x + t * channels + c);
            acc = svmla_f32_x(pg, xv, acc, gv);
            svst1_f32(pg, s + t * channels + c, acc);
        }

        /* Narrow fp32 → fp16 state (once per chunk) */
        svst1_f32(pg, buf, acc);
        for (unsigned i = 0; i < width && c + i < channels; i++)
            state[c + i] = (_Float16)buf[i];
    }
#elif defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16)
    size_t c = 0;
    for (; c + 4 <= channels; c += 4) {
        /* Widen fp16 → fp32 using NEON half-precision conversion */
        float16x4_t state_h = vld1_f16((const __fp16 *)(state + c));
        float32x4_t acc = vcvt_f32_f16(state_h);
        for (size_t t = 0; t < seq; ++t) {
            acc = vfmaq_f32(vld1q_f32(x + t * channels + c), acc,
                            vld1q_f32(g + t * channels + c));
            vst1q_f32(s + t * channels + c, acc);
        }
        /* Narrow fp32 → fp16 */
        float16x4_t acc_h = vcvt_f16_f32(acc);
        vst1_f16((__fp16 *)(state + c), acc_h);
    }
    for (; c < channels; ++c) {
        float acc = (float)state[c];
        for (size_t t = 0; t < seq; ++t) {
            acc = x[t * channels + c] + acc * g[t * channels + c];
            s[t * channels + c] = acc;
        }
        state[c] = (_Float16)acc;
    }
#else
    for (size_t c = 0; c < channels; ++c) {
        float acc = (float)state[c];
        for (size_t t = 0; t < seq; ++t) {
            acc = x[t * channels + c] + acc * g[t * channels + c];
            s[t * channels + c] = acc;
        }
        state[c] = (_Float16)acc;
    }
#endif
}

/* ===========================================================================
 * 1b. Gated scan — bf16 state, fp32 accumulator
 * ===========================================================================*/
void gdn_gated_scan_bf16(const float *restrict g, const float *restrict x,
                         float *restrict s, __bf16 *restrict state,
                         size_t seq, size_t channels) {
#ifdef __ARM_FEATURE_SVE
    for (size_t c = 0; c < channels; c += svcntw()) {
        svbool_t pg = svwhilelt_b32((unsigned)c, (unsigned)channels);
        float buf[MAX_SVE_F32];
        unsigned width = svcntw();
        for (unsigned i = 0; i < width && c + i < channels; i++)
            buf[i] = _bf16_to_f32(state[c + i]);
        svfloat32_t acc = svld1_f32(pg, buf);
        for (size_t t = 0; t < seq; ++t) {
            svfloat32_t gv = svld1_f32(pg, g + t * channels + c);
            svfloat32_t xv = svld1_f32(pg, x + t * channels + c);
            acc = svmla_f32_x(pg, xv, acc, gv);
            svst1_f32(pg, s + t * channels + c, acc);
        }
        svst1_f32(pg, buf, acc);
        for (unsigned i = 0; i < width && c + i < channels; i++)
            state[c + i] = _f32_to_bf16(buf[i]);
    }
#elif defined(__ARM_NEON) && defined(__ARM_FEATURE_BF16)
    size_t c = 0;
    for (; c + 4 <= channels; c += 4) {
        bfloat16x4_t state_h = vld1_bf16(state + c);
        float32x4_t acc = vcvt_f32_bf16(state_h);
        for (size_t t = 0; t < seq; ++t) {
            acc = vfmaq_f32(vld1q_f32(x + t * channels + c), acc,
                            vld1q_f32(g + t * channels + c));
            vst1q_f32(s + t * channels + c, acc);
        }
        bfloat16x4_t acc_h = vcvt_bf16_f32(acc);
        vst1_bf16(state + c, acc_h);
    }
    for (; c < channels; ++c) {
        float acc = _bf16_to_f32(state[c]);
        for (size_t t = 0; t < seq; ++t) {
            acc = x[t * channels + c] + acc * g[t * channels + c];
            s[t * channels + c] = acc;
        }
        state[c] = _f32_to_bf16(acc);
    }
#else
    for (size_t c = 0; c < channels; ++c) {
        float acc = _bf16_to_f32(state[c]);
        for (size_t t = 0; t < seq; ++t) {
            acc = x[t * channels + c] + acc * g[t * channels + c];
            s[t * channels + c] = acc;
        }
        state[c] = _f32_to_bf16(acc);
    }
#endif
}

/* ===========================================================================
 * 2a. Cumulative decay — fp16 output, fp32 accumulator
 *
 *   decay[t] = prod_{i<=t} a[i]
 *
 * The prefix product accumulator is ALWAYS fp32 (constraint from ob-8qt.4:
 * 0.5^64 ≈ 5e-20 underflows fp16). The OUTPUT decay buffer (persistent) is
 * narrowed to fp16 to halve its resident size.
 * ===========================================================================*/
void gdn_cumdecay_f16(const float *restrict a, _Float16 *restrict decay,
                      size_t seq, size_t channels) {
#ifdef __ARM_FEATURE_SVE
    for (size_t c = 0; c < channels; c += svcntw()) {
        svbool_t pg = svwhilelt_b32((unsigned)c, (unsigned)channels);
        svfloat32_t run = svdup_f32(1.0f);
        for (size_t t = 0; t < seq; ++t) {
            svfloat32_t av = svld1_f32(pg, a + t * channels + c);
            run = svmul_f32_x(pg, run, av);
            /* Narrow fp32 result → fp16 for the persistent output */
            float buf[MAX_SVE_F32];
            svst1_f32(pg, buf, run);
            for (unsigned i = 0; i < svcntw() && c + i < channels; i++)
                decay[t * channels + c + i] = (_Float16)buf[i];
        }
    }
#elif defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16)
    size_t c = 0;
    for (; c + 4 <= channels; c += 4) {
        float32x4_t run = vdupq_n_f32(1.0f);
        for (size_t t = 0; t < seq; ++t) {
            run = vmulq_f32(run, vld1q_f32(a + t * channels + c));
            float16x4_t run_h = vcvt_f16_f32(run);
            vst1_f16((__fp16 *)(decay + t * channels + c), run_h);
        }
    }
    for (; c < channels; ++c) {
        float run = 1.0f;
        for (size_t t = 0; t < seq; ++t) {
            run *= a[t * channels + c];
            decay[t * channels + c] = (_Float16)run;
        }
    }
#else
    for (size_t c = 0; c < channels; ++c) {
        float run = 1.0f;
        for (size_t t = 0; t < seq; ++t) {
            run *= a[t * channels + c];
            decay[t * channels + c] = (_Float16)run;
        }
    }
#endif
}

/* ===========================================================================
 * 2b. Cumulative decay — bf16 output, fp32 accumulator
 * ===========================================================================*/
void gdn_cumdecay_bf16(const float *restrict a, __bf16 *restrict decay,
                       size_t seq, size_t channels) {
#ifdef __ARM_FEATURE_SVE
    for (size_t c = 0; c < channels; c += svcntw()) {
        svbool_t pg = svwhilelt_b32((unsigned)c, (unsigned)channels);
        svfloat32_t run = svdup_f32(1.0f);
        for (size_t t = 0; t < seq; ++t) {
            svfloat32_t av = svld1_f32(pg, a + t * channels + c);
            run = svmul_f32_x(pg, run, av);
            float buf[MAX_SVE_F32];
            svst1_f32(pg, buf, run);
            for (unsigned i = 0; i < svcntw() && c + i < channels; i++)
                decay[t * channels + c + i] = _f32_to_bf16(buf[i]);
        }
    }
#elif defined(__ARM_NEON) && defined(__ARM_FEATURE_BF16)
    size_t c = 0;
    for (; c + 4 <= channels; c += 4) {
        float32x4_t run = vdupq_n_f32(1.0f);
        for (size_t t = 0; t < seq; ++t) {
            run = vmulq_f32(run, vld1q_f32(a + t * channels + c));
            bfloat16x4_t run_h = vcvt_bf16_f32(run);
            vst1_bf16(decay + t * channels + c, run_h);
        }
    }
    for (; c < channels; ++c) {
        float run = 1.0f;
        for (size_t t = 0; t < seq; ++t) {
            run *= a[t * channels + c];
            decay[t * channels + c] = _f32_to_bf16(run);
        }
    }
#else
    for (size_t c = 0; c < channels; ++c) {
        float run = 1.0f;
        for (size_t t = 0; t < seq; ++t) {
            run *= a[t * channels + c];
            decay[t * channels + c] = _f32_to_bf16(run);
        }
    }
#endif
}
