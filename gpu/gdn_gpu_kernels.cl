/*
 * OpenCL compute kernels for the Gated DeltaNet (GDN) operators.
 *
 * Bead ob-q44.  Implements the three GDN primitives that the CPU SVE/NEON
 * kernels in src/orionsbelt/engines/cpu/kernels/gdn_sve.c provide, plus the
 * full per-token delta-rule recurrent update that is the actual GDN decode
 * algorithm.
 *
 * LAYOUT — same design decision as the CPU kernels:
 *   The sequence axis is inherently sequential (each step depends on the
 *   previous).  We parallelise across the CHANNEL / HEAD axis and walk the
 *   sequence with a plain for-loop inside each work-item.  No cross-lane
 *   communication, no prefix-scan network.
 *
 * HARDWARE TARGET:
 *   Developed and validated on the RK3588's Mali-G610 MP4 (Valhall, OpenCL 3.0).
 *   The same shader logic transfers to the Orion O6's Immortalis-G720 (Arm 5th
 *   gen GPU) — only peak performance is O6-gated (ADR 0005).
 *
 * KERNEL INDEX
 *   1. gdn_cumdecay   — inclusive prefix product along the sequence axis
 *   2. gdn_gated_scan — the sequential recurrence s[t] = g[t]*s[t-1] + x[t]
 *   3. gdn_causal_dwconv1d — causal depthwise Conv1D, kernel width 4
 *   4. gdn_delta_rule_decode — full per-token delta-rule update on matrix state
 */

/* ========================================================================
 * 1.  Gated cumulative decay
 *
 *   decay[t][c] = prod_{i=0..t} a[i][c]
 *
 * Layout: a and decay are (seq, channels) in row-major order.
 * Each work-item owns one channel and walks all timesteps.
 * ====================================================================== */
__kernel void gdn_cumdecay(
    __global const float *a,      /* [seq * channels] */
    __global float *decay,         /* [seq * channels] */
    const uint seq,
    const uint channels)
{
    uint c = get_global_id(0);
    if (c >= channels) return;

    float run = 1.0f;
    for (uint t = 0; t < seq; t++) {
        run *= a[t * channels + c];
        decay[t * channels + c] = run;
    }
}

/* ========================================================================
 * 2.  Chunkwise gated scan — the recurrence the NPU cannot express
 *
 *   s[t][c] = g[t][c] * s[t-1][c] + x[t][c]
 *   state[c] = s[seq-1][c]   (carried to the next invocation)
 *
 * Layout: g, x, s are (seq, channels) row-major; state is [channels].
 * ====================================================================== */
__kernel void gdn_gated_scan(
    __global const float *g,       /* [seq * channels] */
    __global const float *x,       /* [seq * channels] */
    __global float *s,             /* [seq * channels] */
    __global float *state,         /* [channels] — read on entry, written on exit */
    const uint seq,
    const uint channels)
{
    uint c = get_global_id(0);
    if (c >= channels) return;

    float acc = state[c];
    for (uint t = 0; t < seq; t++) {
        uint off = t * channels + c;
        acc = fma(acc, g[off], x[off]);   /* acc = x + acc * g */
        s[off] = acc;
    }
    state[c] = acc;
}

/* ========================================================================
 * 3.  Causal depthwise Conv1D, kernel width 4
 *
 *   out[t][c] = w0[c]*hist0 + w1[c]*hist1 + w2[c]*hist2 + w3[c]*in[t][c]
 *
 * hist[3 * channels] carries the last 3 timesteps across invocations.
 * ====================================================================== */
__kernel void gdn_causal_dwconv1d(
    __global const float *in,      /* [seq * channels] */
    __global const float *w,       /* [4 * channels] */
    __global float *out,           /* [seq * channels] */
    __global float *hist,          /* [3 * channels] — state, read+write */
    const uint seq,
    const uint channels)
{
    uint c = get_global_id(0);
    if (c >= channels) return;

    float h0 = hist[0 * channels + c];
    float h1 = hist[1 * channels + c];
    float h2 = hist[2 * channels + c];
    float w0 = w[0 * channels + c];
    float w1 = w[1 * channels + c];
    float w2 = w[2 * channels + c];
    float w3 = w[3 * channels + c];

    for (uint t = 0; t < seq; t++) {
        uint off = t * channels + c;
        float cur = in[off];
        out[off] = h0 * w0 + h1 * w1 + h2 * w2 + cur * w3;
        h0 = h1;
        h1 = h2;
        h2 = cur;
    }
    hist[0 * channels + c] = h0;
    hist[1 * channels + c] = h1;
    hist[2 * channels + c] = h2;
}

/* ========================================================================
 * 4.  Full per-token delta-rule recurrent update (the actual GDN decode)
 *
 * For each token t, per value-head h:
 *
 *   S_h *= exp(g_h)                              // decay  (scalar × matrix)
 *   kv  = S_h^T · k_h                             // retrieve (mat-vec, dim_v)
 *   delta = (v_h - kv) * beta_h                   // correction (elementwise)
 *   S_h += k_h ⊗ delta                            // write (rank-1 update)
 *   out_h = S_h^T · q_h                           // read (mat-vec, dim_v)
 *
 * State S_h is a (head_k_dim × head_v_dim) matrix, row-major.
 * Each work-group processes one head for one token; work-items tile the
 * matrix rows for the mat-vec and rank-1 update phases.
 *
 * Dimensions (Qwen3.5-0.8B defaults):
 *   num_heads = 16, head_k_dim = 128, head_v_dim = 128
 *
 * Buffers:
 *   S       [num_heads * head_k_dim * head_v_dim]  — recurrent state (persistent)
 *   k_seq   [seq * num_heads * head_k_dim]         — key vectors per token
 *   v_seq   [seq * num_heads * head_v_dim]         — value vectors per token
 *   q_seq   [seq * num_heads * head_k_dim]         — query vectors per token
 *   beta_s  [seq * num_heads]                      — write gates
 *   g_s     [seq * num_heads]                      — decay gates (pre-exp)
 *   out_s   [seq * num_heads * head_v_dim]         — output per token
 *
 * One work-group per (head, token-pair).  Since the recurrence is sequential
 * across tokens, we launch one token at a time from the host, or use a
 * single work-item per head that loops over tokens (chosen here for
 * simplicity — the mat-vec parallelism is within the matrix, not across
 * tokens).
 *
 * Launch: global_size = (head_v_dim, num_heads), local_size = (head_v_dim, 1)
 * Each work-item computes one element of the output vector and one row of
 * the state update.
 * ====================================================================== */
__kernel void gdn_delta_rule_decode(
    __global float *S,             /* [num_heads * hkd * hvd] — persistent state */
    __global const float *k_seq,   /* [num_heads * hkd] — key for this token */
    __global const float *v_seq,   /* [num_heads * hvd] — value for this token */
    __global const float *q_seq,   /* [num_heads * hkd] — query for this token */
    const float beta,              /* write gate for this head/token */
    const float decay,             /* exp(g) — decay factor for this head/token */
    __global float *out,           /* [num_heads * hvd] — output for this token */
    const uint hkd,               /* head_k_dim */
    const uint hvd,               /* head_v_dim */
    const uint num_heads)
{
    uint h = get_global_id(1);     /* head index */
    uint row = get_global_id(0);   /* row index in [0, hvd) — value dimension */

    if (h >= num_heads || row >= hvd) return;

    __global float *S_h = S + (size_t)h * hkd * hvd;       /* [hkd × hvd] */
    __global const float *k_h = k_seq + (size_t)h * hkd;   /* [hkd] */
    __global const float *q_h = q_seq + (size_t)h * hkd;   /* [hkd] */
    __global float *out_h = out + (size_t)h * hvd;          /* [hvd] */

    /* Phase 1: decay — multiply entire row by decay factor */
    for (uint j = 0; j < hkd; j++)
        S_h[(size_t)j * hvd + row] *= decay;

    /* Phase 2: retrieve — kv = S_h^T · k_h  (dot product of column row with k)
     * S_h is (hkd × hvd) row-major, so S_h[j*hvd + row] is element (j, row).
     * kv = sum_j S_h[j, row] * k_h[j] — this work-item accumulates its row's contribution.
     * Actually: kv[row] = sum_j S_h[j, row] * k_h[j]
     * Each work-item computes one element of kv[row]. */
    float kv = 0.0f;
    for (uint j = 0; j < hkd; j++)
        kv += S_h[(size_t)j * hvd + row] * k_h[j];

    /* Phase 3: correction — delta = (v - kv) * beta */
    float delta = (v_seq[(size_t)h * hvd + row] - kv) * beta;

    /* Phase 4: write — rank-1 update: S_h[j, row] += k_h[j] * delta */
    for (uint j = 0; j < hkd; j++)
        S_h[(size_t)j * hvd + row] = fma(k_h[j], delta, S_h[(size_t)j * hvd + row]);

    /* Phase 5: read — out[row] = sum_j S_h[j, row] * q_h[j] */
    float o = 0.0f;
    for (uint j = 0; j < hkd; j++)
        o += S_h[(size_t)j * hvd + row] * q_h[j];

    out_h[row] = o;
}
