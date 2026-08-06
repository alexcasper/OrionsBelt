/* Gated DeltaNet CPU micro-kernels: public API.
 *
 * See gdn_sve.c for the full implementation, ISA-floor rationale, and
 * the layout decision (vectorize across channels, walk the sequence
 * sequentially). This header exists so that every consumer — bench_gdn.c,
 * test_gdn_sve.c, test_gdn2_scan.c, test_gdn_mixed.c — shares a single
 * source of truth for function signatures.
 *
 * Why this matters: a previous bug (Session 15, commit 20b50c7) passed the
 * same pointer for w_gate and x in gdn2_gated_scan_f32 because bench_gdn.c
 * re-declared the function without a header to check against. A shared header
 * makes signature changes a compile error instead of silent UB.
 */
#ifndef GDN_SVE_H
#define GDN_SVE_H

#include <stddef.h>
#include <stdint.h>   /* uint16_t for bf16 state */

/* ---------------------------------------------------------------------------
 * fp32 kernels (the primary path)
 * ------------------------------------------------------------------------- */

/* Cumulative decay (inclusive prefix product along the sequence axis).
 *   decay[t][c] = prod_{i<=t} a[i][c]
 * Layout: a, decay are [seq][channels] row-major. */
void gdn_cumdecay_f32(const float *restrict a, float *restrict decay,
                      size_t seq, size_t channels);

/* Chunkwise gated scan (the recurrence the NPU cannot express).
 *   s[t][c] = g[t][c] * s[t-1][c] + x[t][c]
 * state[] carries across calls (cross-invocation continuity). */
void gdn_gated_scan_f32(const float *restrict g, const float *restrict x,
                        float *restrict s, float *restrict state,
                        size_t seq, size_t channels);

/* Causal depthwise Conv1D, kernel width 4.
 *   out[t][c] = sum_{j=0..3} w[j][c] * in[t-3+j][c]
 * hist[] holds the K-1=3 previous timesteps per channel (3*channels floats),
 * carried across calls for single-token decode. */
void gdn_causal_dwconv1d_f32(const float *restrict in, const float *restrict w,
                             float *restrict out, float *restrict hist,
                             size_t seq, size_t channels);

/* ---------------------------------------------------------------------------
 * Mixed-precision variants (bead ob-8qt.4)
 *
 * Narrow storage (fp16/bf16 state/output), wide compute (fp32 accumulate).
 * The fp32 accumulator runs in registers across the entire seq loop, so the
 * narrowing cost is O(channels), not O(channels × seq).
 * ------------------------------------------------------------------------- */

/* fp16-output cumulative decay */
void gdn_cumdecay_f16(const float *restrict a, __fp16 *restrict decay,
                      size_t seq, size_t channels);

/* fp16-state gated scan */
void gdn_gated_scan_f16(const float *restrict g, const float *restrict x,
                        float *restrict s, __fp16 *restrict state,
                        size_t seq, size_t channels);

/* bf16-output cumulative decay (state stored as uint16_t via software conversion) */
void gdn_cumdecay_bf16(const float *restrict a, uint16_t *restrict decay,
                       size_t seq, size_t channels);

/* bf16-state gated scan */
void gdn_gated_scan_bf16(const float *restrict g, const float *restrict x,
                         float *restrict s, uint16_t *restrict state,
                         size_t seq, size_t channels);

/* ---------------------------------------------------------------------------
 * GDN-2 decoupled-gating scan (bead ob-y3f)
 *
 *   GDN-1: s[t] = x[t] + g[t] * s[t-1]
 *   GDN-2: s[t] = w[t]*x[t] + g[t]*b[t]*s[t-1]
 *
 * Two extra channel-wise gates: b_gate (erase), w_gate (write).
 * 4 FLOPs/element (2 MUL + 1 FMA), 5 streams (g, b, w, x, out).
 */
void gdn2_gated_scan_f32(const float *restrict g, const float *restrict b_gate,
                         const float *restrict w_gate, const float *restrict x,
                         float *restrict s, float *restrict state,
                         size_t seq, size_t channels);

#endif /* GDN_SVE_H */
