// SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
// SPDX-License-Identifier: Apache-2.0
//
// Origin: src/orionsbelt/engines/cpu/kernels/gdn_delta_matmul.c (OrionsBelt repository)
//
// KleidiAI micro-kernel: NEON GEMV for the M=1 decode-path delta-rule matmul.
//
//   C[j] = sum_k  A[k] * B[k][j]     (matrix-vector product, M=1)
//
// At decode (M=1), the recurrent state S changes every chunk, so KleidiAI's
// packed-GEMM repack cost (7-126 us on Cortex-A76) exceeds the matmul itself.
// Hand-NEON without packing wins 2.6-3.0x at M=1 (measured).  This kernel is
// the extracted standalone form of that fast path.
//
// The "1x4" in the name encodes M=1 (single output row) and 4-wide NEON
// accumulation.  The "neon" suffix indicates fp32 NEON FMA vectorisation —
// no int8 dot-product (SDOT) instructions are used, because the delta-rule
// operands are fp32.

#ifndef KAI_GDN_GEMV_F32_F32_F32_1X4_NEON_H
#define KAI_GDN_GEMV_F32_F32_F32_1X4_NEON_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/// GEMV (M=1) for the GDN delta-rule matmul.
///
/// Computes C[j] = sum_{k=0}^{K-1} A[k] * B[k][j] for j = 0 .. N-1.
///
/// @param a Input vector (single row of A), length K.
/// @param b RHS matrix B, layout [K][N], row-major.
/// @param c Output vector (single row of C), length N.
/// @param k Reduction dimension (head_dim, typically 128).
/// @param n Output dimension (head_dim * n_heads).
void kai_run_gdn_gemv_f32_f32_f32_1x4_neon(const float *a,
                                            const float *b, float *c,
                                            size_t k, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* KAI_GDN_GEMV_F32_F32_F32_1X4_NEON_H */
