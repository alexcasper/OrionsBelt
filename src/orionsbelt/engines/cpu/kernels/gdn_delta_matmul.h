// SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
// SPDX-License-Identifier: Apache-2.0

/* Gated DeltaNet delta-rule matmul: beta = alpha . S
 *
 * Bead ob-8qt.1. See gdn_delta_matmul.c for the dual-path dispatch rationale.
 */
#ifndef GDN_DELTA_MATMUL_H
#define GDN_DELTA_MATMUL_H

#include <stddef.h>

/* A is row-major [M×K], B is row-major [K×N], C is row-major [M×N] (C = A @ B).
 * No bias, no clamp -- the delta-rule update has neither. */
void gdn_delta_rule_matmul(const float *restrict A, const float *restrict B,
                           float *restrict C, size_t M, size_t K, size_t N);

#endif /* GDN_DELTA_MATMUL_H */
