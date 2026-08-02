# Findings

Empirical results from porting a Gated DeltaNet model to CIX P1 / Arm silicon. Reusable by
anyone attempting the same thing — including the negative results, which are the point.

---

## 1. NOE Compiler operator coverage for Gated DeltaNet (2026-08-02)

**Bead `ob-t3b.1`. Run on an x86 host with no Orion O6 board attached.**

### Headline

> **Every arithmetic operator Gated DeltaNet needs is natively supported by the CIX NOE
> Compiler. The sequential recurrence that ties them together is not.**

The NPU can compute GDN's per-chunk math. It cannot express the chunk-to-chunk scan. Concretely:
`Scan` is rejected outright, and `Loop` is accepted *only* when its trip count is a compile-time
constant — in which case it is fully unrolled and no loop survives into the IR. A `Loop` whose trip
count is a runtime input is rejected.

This was measured, not assumed, and it is the empirical basis for the layer-to-engine mapping in
[`PLAN.md`](../PLAN.md) §3.1: **CPU hosts the GDN recurrence; accelerators take the dense math.**

### Method

Six hand-authored minimal ONNX graphs (plus one follow-up), one operator family each, so a failure
is attributable to exactly one operator — see [`artifacts/npu_op_probe/`](../artifacts/npu_op_probe/).
Each was verified locally first (`onnx.checker` full check, executes under onnxruntime with finite
outputs, causal conv verified genuinely causal), so a rejection is a NOE coverage gap and not a
malformed input.

Driven through `cixparse` (the frontend that lowers a framework graph to AIPU IR) from
`cixbuilder-6.1.3753.3`, NOE SDK `26_q2` release. Generator: [`scripts/npu_op_probe.py`](../scripts/npu_op_probe.py);
runner: [`scripts/run_op_probe_audit.py`](../scripts/run_op_probe_audit.py). Logs and configs for
every probe are committed under [`artifacts/npu_op_probe/audit/`](../artifacts/npu_op_probe/audit/).

### Results

| Probe | ONNX ops | Verdict | Resulting AIPU IR |
|---|---|---|---|
| causal depthwise Conv1D | `Conv` (groups=C, pads=[3,0]) | ✅ supported | `ArmDepthwiseConv` **+ inserted `ArmReshape`/`ArmTranspose` pairs** |
| gated decay | `Log`, `CumSum`, `Exp` | ✅ supported | `ArmLog`, **`ArmCumulate`**, `ArmExp` |
| delta-rule state update | `MatMul`, `Sub`, `Add`, `Transpose` | ✅ supported | `ArmMatMul` ×3, `ArmEltwise` ×2 |
| elementwise gate chain | `Sigmoid`, `Softplus`, `Neg`, `Exp`, `Mul` | ✅ supported | native |
| chunk recurrence via `Scan` | `Scan` | ❌ **rejected** | — |
| chunk recurrence via `Loop`, **const** trip count | `Loop` | ⚠️ **unrolled, not supported** | 4 × `ArmMul` — **no loop in IR** |
| chunk recurrence via `Loop`, **runtime** trip count | `Loop` | ❌ **rejected** | — |

Verbatim evidence:

```
# Scan
[W] [Parser]: Meet unsupported op type Scan in Node(chunkwise_scan)!    (rc=255)

# Loop with a runtime trip count
[D] [Parser]: Loop(dynamic_loop) max_count/cond_in is non-const, the infer shape is unreliable.
[E] [Parser]: Graph(dynamic_loop_body_subgraph) is not DAG!
```

### Why the "Loop works" result is a trap

`Loop` with a constant trip count returns rc=0, which looks like support. The IR shows what
actually happened — the body was replicated once per iteration:

```
ArmMul, decay_state     ArmMul, decay_state_0
ArmMul, decay_state_1   ArmMul, decay_state_2
```

Four iterations, four multiplies, no loop construct. That is **static unrolling**, and it only
works when the iteration count is known at compile time. Two consequences:

1. **It cannot express variable-length decode.** The number of chunks depends on sequence length,
   which is a runtime property. The runtime-trip-count probe above is the direct proof.
2. **It does not scale even for fixed lengths.** At 262K context with a chunk size of 64, one
   unrolled GDN layer is ~4,096 replicated bodies; across 24 GDN layers that is ~98,000 nodes in a
   single graph. Not a practical compilation target.

Anyone benchmarking this platform who sees `Loop` compile should check the IR before concluding the
recurrence is supported. This is exactly the silent-success case that looks like a win and performs
like a wall.

### Secondary observation: layout-conversion overhead on the causal conv

The depthwise Conv1D is supported but the parser wraps it in `ArmReshape` + `ArmTranspose` on both
input and output, converting between NCHW and its preferred NHWC layout. GDN applies this
convolution in **every one of the 24 linear layers**, so that conversion is paid 24× per forward
pass. Whether it is elided later in the pipeline is not yet established — worth measuring before
assuming the conv is free.

### What this does and does not establish

**Does:** the NPU frontend has native operators for all of GDN's arithmetic — including `CumSum`
as `ArmCumulate`, which was the operator most at risk given that ONNX has no `CumProd` and the
gated decay must be expressed as `exp(cumsum(log(a)))`. And it has no mechanism for a
runtime-length sequential scan.

**Does not:** this is frontend lowering only. Parsing to `Arm*` IR does not prove an operator
executes *on the NPU* rather than falling back to CPU at runtime, nor say anything about achieved
throughput. Per-op engine assignment and performance require the full `cixbuild` pipeline and
on-device execution — the board-gated half of this work, tracked in `ob-8xc`.

### Consequence for the design

The mapping hypothesis in `PLAN.md` §3.1 was argued from workload shape and Arm-IP relevance
*before* this audit. It now has empirical support:

- **GDN recurrence → CPU.** Not a preference. The NPU has no construct that expresses it.
- **Per-chunk dense math → NPU/GPU is viable.** `ArmMatMul`, `ArmCumulate`, and
  `ArmDepthwiseConv` all exist, so a design that keeps the *scan* on CPU while offloading the
  *inner* per-chunk math is supported by the toolchain. This is why the CPU scan kernel
  (`ob-8qt.1`) is specified to keep those two layers separable.
- **The open cost question is dispatch latency**, not operator support — 16 boundary crossings per
  token in a 3:1 stack (`ob-t3b.3`).

### Reproducing

```bash
# host: x86 Linux, Python 3.10 (the cixbuilder wheel is cp310 — 3.11 will refuse it)
uv venv --python 3.10 noe310
uv pip install --python noe310/bin/python cixbuilder-6.1.3753.3-cp310-none-linux_x86_64.whl
uv pip install --python noe310/bin/python tensorflow-cpu   # cixparse imports tf unconditionally

python3 scripts/npu_op_probe.py --out artifacts/npu_op_probe
python3 scripts/run_op_probe_audit.py \
    --cixparse noe310/bin/cixparse --probe-dir artifacts/npu_op_probe
```

Two environment notes that cost time and are not in the vendor docs:

- The `cixbuilder` wheel is tagged **`cp310`** — Python 3.10 only. Confirms the SDK docs and
  refutes community reports of 3.11/3.12 support.
- **`cixparse` imports `tensorflow` at startup even for ONNX input**, and fails with
  `ModuleNotFoundError` without it. `tensorflow-cpu` satisfies it. The SDK archive ships no host
  `requirements.txt`.

---

## 2. No linear-attention model exists for this platform (2026-08-02)

See [`CLAIM_VERIFICATION.md`](./CLAIM_VERIFICATION.md) §3.1 for the full check. In summary: the CIX
AI Model Hub (`26_Q1`/`master`) ships 38 LLMs, all conventional full-attention transformers —
Qwen1.5 through Qwen3 including 30B-A3B, Llama, Phi, InternLM, ERNIE, MiniCPM, DeepSeek distills.
No Qwen3.5, Qwen3-Next, Mamba, RWKV, or any recurrent-state architecture.

MoE is supported (Qwen3-30B-A3B is present) and Qwen3 is present, so neither sparsity nor recency
is the barrier — the absent thing is the architecture class. Combined with §1 above, the reason is
now clear rather than speculative: **the toolchain cannot express a variable-length recurrence, so
no linear-attention model could have been shipped through it in the first place.**

The SDK does ship a vendor `cix-llama-cpp` build (1.3.1) for the board, which is the most likely
existing inference path — and a candidate baseline, though its GDN support is unverified.

---

## 3. Toolchain and KleidiAI coverage for Armv9.2 GDN kernels (2026-08-02)

Investigating what compiler targets Cortex-A720, and whether Arm's own micro-kernel library
already contains anything we need for the CPU-hosted GDN scan (`ob-8qt.1`).

### 3.1 Compiler and target flags

**Clang/LLVM ≥17 with `-mcpu=cortex-a720`.** That single flag is sufficient — verified by dumping
clang 18's predefined feature macros for that target:

```
__ARM_FEATURE_SVE2 1              __ARM_FEATURE_MATMUL_INT8 1
__ARM_FEATURE_SVE2_BITPERM 1      __ARM_FEATURE_SVE_MATMUL_INT8 1
__ARM_FEATURE_SVE 1               __ARM_FEATURE_SVE_MATMUL_FP32 1
__ARM_FEATURE_BF16 1              __ARM_FEATURE_SVE_BF16 1
__ARM_FEATURE_DOTPROD 1           __ARM_FEATURE_FP16_FML 1
```

Equivalent explicit form: `-march=armv9.2-a+sve2+i8mm+bf16`. Arm Compiler for Linux (ACfL) is
Arm's own LLVM-based toolchain and is a defensible choice for an Arm submission.

**GCC caveat (corrected 2026-08-02).** GCC **13 does not know `-mcpu=cortex-a720`** — it was added
in GCC 14. GCC 13 rejects it outright (`unknown value 'cortex-a720' for '-mcpu'`, suggesting
`cortex-a72`, which would be badly wrong to accept silently). On GCC 13 use the **arch-level flag**
`-march=armv9.2-a+sve2+i8mm+bf16`, which is accepted and gives the same ISA. Verified by compiling
our kernels both ways. So: clang ≥17 or **GCC ≥14** for the CPU-name flag; GCC 13 needs the
`-march` form.

Cross-compiling from x86 needs an aarch64 sysroot. Note the DSP SDK archive ships
`ext/mirror/cix_sysroot.tgz`, which is a CIX-matched sysroot and therefore a better match than a
generic Debian one.

**Cortex-A720 has no SME.** `__ARM_FEATURE_SME` is absent under `-mcpu=cortex-a720` while a
`-march=armv9.2-a+sme` control does define it, so the absence is meaningful. Corroborated by
LLVM's own `AArch64Processors.td` feature list for the core, which contains `FeatureSVE`,
`FeatureSVE2`, `FeatureSVEBitPerm`, `FeatureMatMulInt8`, `FeatureBF16`, `FeatureDotProd` and
**no SME or SME2 entry**.

> **Method note.** An earlier attempt to test this by assembling SME instructions under
> `-mcpu=cortex-a720` was discarded as invalid: clang's integrated assembler accepted SME
> instructions even without `+sme`, i.e. it does not gate `.s` input on `-mcpu` features. The
> negative control is what caught it. Feature macros and the LLVM processor definition are the
> sound tests.

### 3.2 The SVE2 width trap

**SVE2 on Cortex-A720 is 128-bit** — the same width as NEON (Arm's TRM documents SVE with a
128-bit vector length). So SVE2 buys **predication** (`whilelt`/`svmla_x`, which gives clean tail
handling for a chunked scan without a scalar epilogue) and gather/scatter — **not** more lanes.
Anyone expecting a free speedup from "scalable vectors" on this core will be disappointed, and any
claim we make about SVE2 must be framed as predication and instruction-selection wins rather than
vector-width wins.

### 3.3 KleidiAI has nothing for the recurrence

Inventoried at KleidiAI `98872b0`. It ships exactly two micro-kernel families:

| Family | Count | ISA targets | Usable on Cortex-A720? |
|---|---:|---|---|
| `matmul` | 185 | neon, dotprod, i8mm, sme/sme2/mopa, a few sve | **109 yes** (neon/dotprod/i8mm), 76 no (SME family) |
| `dwconv` | 4 | **sme2 only** | **None** |

A repo-wide search for `cumsum`, `cumulative`, `prefix.sum`, `scan_`, `recurren`, `ssm`,
`state.space`, `mamba`, `deltanet`, and `linear.attention` returns **no matching kernel** — the two
textual hits are incidental substring matches (`ShapesSmallKC`, a Winograd SME path), not
implementations.

So, concretely, for the three primitives GDN needs on the CPU:

| GDN primitive | KleidiAI status |
|---|---|
| delta-rule small matmuls | ✅ **covered** — 109 A720-usable `matmul` kernels, incl. i8mm/dotprod quantized paths |
| causal depthwise Conv1D | ❌ **exists but SME2-only**, so unusable on Cortex-A720 |
| gated cumulative decay (prefix product) | ❌ **no primitive of any kind** |
| chunkwise sequential scan | ❌ **no primitive of any kind** |

### 3.4 What this means for the contribution

This sharpens the project's claim considerably, and it is now specific rather than rhetorical.
Arm's own optimized-kernel library covers the dense matmuls we need and **nothing else** in the GDN
stack, and its one depthwise-conv family targets an extension this CPU does not implement.

The contribution is therefore three named micro-kernels that do not exist anywhere today for a
non-SME Armv9.2 core:

1. causal depthwise Conv1D for SVE2 + NEON (KleidiAI's is SME2-only)
2. gated cumulative decay / prefix product with predicated tails
3. the chunkwise gated delta-rule scan

All three are shaped like KleidiAI micro-kernels and are **upstreamable to it** — which is a
concrete Potential Impact (20 pts) argument rather than an aspirational one: the deliverable is
reusable by anyone running a linear-attention model on any non-SME Armv9 CPU, not just on the O6.

It also composes with §1: the NPU cannot express the recurrence *and* Arm's kernel library has no
recurrence primitive, so on this platform the sequential scan has no existing home at all. That is
the gap.

---

## 4. The three CPU kernels: implemented and numerically verified (2026-08-02)

**Bead `ob-8qt.1`. Written and verified with no Orion O6 board** — QEMU emulates SVE2, so
correctness is checkable today; only *performance* needs real silicon.

Source: [`src/orionsbelt/engines/cpu/kernels/gdn_sve.c`](../src/orionsbelt/engines/cpu/kernels/gdn_sve.c).
Verify with [`scripts/verify_cpu_kernels.sh`](../scripts/verify_cpu_kernels.sh).

### The layout decision that makes all three easy

A prefix scan across *vector lanes* needs a log-depth Hillis-Steele shuffle network, which is
where these kernels look intimidating. **We never do that.** GDN's sequence axis is inherently
sequential, so we vectorize across the **channel/head** axis and walk the sequence with a plain
scalar loop. Every kernel then reduces to independent lane-wise FMAs with **no cross-lane
communication anywhere** — and the 2048-channel width of Qwen3.5-4B's linear layers (16 key heads
× 128) gives far more parallelism than a 4-lane vector can absorb.

This is why "the recurrence is hard for accelerators" and "the recurrence is easy on a CPU" are
both true. The dependency is along a dimension the CPU was going to iterate anyway.

### The three kernels

| Kernel | Recurrence | Implementation |
|---|---|---|
| `gdn_cumdecay_f32` | `decay[t] = decay[t-1] * a[t]` | one predicated `svmul` per step |
| `gdn_gated_scan_f32` | `s[t] = g[t]*s[t-1] + x[t]` | one predicated `svmla` per step; `state[]` carries across calls |
| `gdn_causal_dwconv1d_f32` | 4-tap depthwise, causal | 4 FMAs per step; `hist[]` is a 3-timestep ring, the conv-state analogue of a KV cache |

Design notes worth keeping:

- **The decay accumulator is fp32 even when surrounding state is fp16.** A decay of 0.5 compounded
  over 64 steps is ~5e-20, which underflows fp16 entirely. Computed as a direct product rather
  than `exp(cumsum(log a))`: at chunk length 64 the direct form is both cheaper and accurate, and
  avoids two transcendentals per element.
- **`state[]` and `hist[]` are explicit caller-owned buffers**, which is precisely the
  cross-invocation state continuity the NOE toolchain has no mechanism for (§1). On the CPU it is
  just a pointer.
- The conv ring buffer shifts by **register renaming** (`h0=h1; h1=h2; h2=cur`), so there are no
  cross-lane or memory ops in the inner loop.
- Written **vector-length-agnostic**, so they widen for free on a core with longer vectors even
  though Cortex-A720 is 128-bit.
- The scan is deliberately the *outer* sequential half only, kept separate from per-chunk dense
  math so the mapping ADR can offload the inner matmuls without touching this (PLAN.md §3.1).

### Verification results

Checked against an independently written scalar reference at 2048 channels and **2051** (a
deliberately awkward width that exercises the SVE2 predicated tail):

| Check | Result |
|---|---|
| `gated_scan` vs precision-matched reference | **bit-identical** (`max_abs = 0.0`) |
| `gated_scan` carried state | **bit-identical** |
| `causal_dwconv1d` vs matched reference | `max_abs = 5.96e-08` — one fp32 ULP, from `svmla` FMA contraction |
| conv history state | **bit-identical** |
| `gated_scan` vs double reference | `max_abs = 1.19e-07`, `max_rel = 3.1e-06` — honest fp32 accumulation quality over 64 steps |
| **Causality** | perturbing the last input changes `t = T-1` and leaks **exactly 0.0** into all earlier outputs |
| SVE @ 256-bit and 512-bit | identical results — confirms genuine vector-length agnosticism |
| NEON path (`-march=armv8-a`) | scan bit-identical; conv within one ULP, as SVE |

### The ISA floor is SVE1, not SVE2

Worth stating plainly because the original filename said otherwise: **these kernels need only
base SVE.** Every intrinsic used — `svcntw`, `svdup_f32`, `svld1_f32`, `svmul_f32_x`,
`svmla_f32_x`, `svst1_f32`, `svwhilelt_b32` — is SVE1, and the guard is `__ARM_FEATURE_SVE`
rather than `__ARM_FEATURE_SVE2`. Nothing in an fp32 prefix-product, gated scan, or 4-tap
depthwise convolution requires SVE2's additions, which are mostly integer and DSP oriented.

Verified across the full matrix, all producing identical results:

| Target | ISA | Result |
|---|---|---|
| `-march=armv8.2-a+sve` @128/256/512-bit | SVE1 | PASS |
| `-mcpu=neoverse-v1` | SVE1 (256-bit) | PASS |
| `-mcpu=a64fx` | SVE1 (512-bit) | PASS |
| `-march=armv9-a`, `-mcpu=cortex-a710` | SVE2 | PASS |
| `-mcpu=neoverse-v2` | SVE2 (Graviton4) | PASS |
| `-march=armv8-a` | no SVE | PASS (scalar fallback) |

Two consequences:

1. **The Edge AI hedge is safe.** ADR 0002 named AWS Graviton as the fallback if no phone is
   suitable — Graviton3 is SVE1-only, so an SVE2-dependent kernel would have quietly broken that
   escape route. It doesn't.
2. **The reusability claim widens.** These are usable on any SVE-capable AArch64 core, and
   degrade to scalar elsewhere — not just on Armv9 parts.

**Where SVE2 and i8mm actually start to matter:** the quantized delta-rule matmuls (`ob-8qt.2`,
via KleidiAI) and any future int8/bf16 state variant, where SVE2's widening integer multiply-
accumulates earn their place. For the fp32 recurrence primitives, SVE1 is sufficient and claiming
otherwise would overstate the requirement.

### The variant that was actually missing was NEON, not SVE1

Having established SVE1 suffices, the natural follow-up is which variant is worth writing at all.
Measured answer: **NEON**, and the gap was real.

Disassembling the no-SVE build showed GCC 13 at `-O3` emitting **zero** NEON vector operations for
all three kernels — purely scalar FP. That is not a compiler failing so much as a consequence of
the loop structure: the inner loop carries a serial dependency (`acc = x + acc*g`), so it cannot be
vectorized, and GCC does not interchange the loops to vectorize across channels instead. The SVE
path only gets that for free because we hand-wrote the channel-wise vectorization.

Explicit NEON paths were therefore added (`vfmaq_f32` over `float32x4_t`, with a scalar tail for
channel counts not divisible by 4), guarded `__ARM_FEATURE_SVE` → `__ARM_NEON` → scalar. The inner
loop now compiles to `fmla v1.4s, v0.4s, v2.4s` with 128-bit `ldr q`/`str q` accesses, and produces
results identical to the SVE path (scan bit-identical, conv within the same one-ULP FMA difference).

Why this matters more than an SVE1/SVE2 split:

- **Apple silicon has no SVE at all**, so without a NEON path it would have run 4-wide-scalar.
  ADR 0002 keeps Apple silicon as a supplementary hedge measurement, and that is now worth having.
- **Most deployed Armv8 Android devices have no SVE either.** The Edge AI framing is about breadth
  of deployable hardware, and NEON is the floor that actually reaches it.
- It makes the "reusable by anyone" claim true rather than aspirational: SVE where available, NEON
  everywhere else, scalar as a correctness reference.

So the final dispatch ladder is **SVE (1 or 2, any vector length) → NEON → scalar**, with all
three verified to agree.

> **Method note.** The first test run reported failures at a 1e-5 relative tolerance. That was the
> *test* being wrong, not the kernels: the reference accumulated in `double` while the kernel used
> `float`, and both quantities cross zero, so relative error near zero is meaningless. Re-running
> against a precision-matched reference gave bit-identical results. Worth recording because the
> naive version of this test would have sent us hunting a nonexistent bug.

### What is verified and what is not

**Verified:** numerical correctness, causality, predicated-tail handling, vector-length
agnosticism, and scalar-fallback equivalence.

**Not verified:** anything about speed. QEMU emulates the ISA but tells you nothing useful about
Cortex-A720 cycle counts, cache behaviour, or memory bandwidth. No performance claim can be made
until the kernels run on real silicon (`ob-41j`, `ob-c9k`).

Also still open: these are fp32. The i8mm/dotprod paths for the delta-rule matmuls are separate
(`ob-8qt.2`, reusing KleidiAI's 109 A720-usable matmul micro-kernels rather than reimplementing),
and a bf16/fp16 state variant is worth measuring since it halves state traffic — the dominant cost
in GDN decode.


---

## 5a. Jetson Nano A57: real-silicon optimization results (2026-08-03)

**Beads `ob-8ms.3`, `ob-8qt.4`–`ob-8qt.7`. All measurements on jetson-j2 (2nd Jetson Nano unit).**

Section 5 stated "no performance results yet — awaiting real silicon." That wait is over. The kernels
now run on a Cortex-A57 with a full optimization stack, and the numbers tell a clear story about
what limits GDN on edge-class Arm silicon.

### Device

| Property | Value |
|---|---|
| SoC | NVIDIA Tegra X1 (jetson-j2) |
| CPU | Cortex-A57 (Armv8.0-A), 4 cores |
| Frequency | 1479 MHz (governor: performance) |
| ISA | NEON only — no SVE, no dotprod, no i8mm, no bf16 |
| L1 D-cache | 48 KB per core |
| L2 cache | 1 MB shared |
| DRAM | LPDDR4 64-bit @ 1600 MHz → 12.8 GiB/s peak |
| Thermals | 43–51 °C across all runs |

This is the most constrained device in the fleet: Armv8.0 with only NEON. Every optimization below
must work on this ISA floor — no SVE predication, no hardware bf16, no dotprod. That constraint is a
feature: it proves the kernels are genuinely portable, not silently relying on a newer ISA.

### Baseline (single-threaded NEON)

The kernels as written in §4, compiled at `-O3 -mcpu=cortex-a57`, single-threaded:

| Kernel | 4B p50 (μs) | 4B GiB/s | 0.8B p50 (μs) | 0.8B GiB/s |
|---|---:|---:|---:|---:|
| cumdecay | 2108 | 0.93 | 593 | 1.65 |
| gated_scan | 3893 | 0.76 | 1096 | 1.35 |
| causal_dwconv1d | 1961 | 1.05 | 518 | 1.99 |

Single-threaded, the kernels achieve <2 GiB/s — under 15% of DRAM peak. The bottleneck is the FMA
dependency chain in the inner loop: each time step's result depends on the previous, so the pipeline
stalls for 4 cycles per step regardless of how wide the vector path is.

### Optimization stack

Four portable optimizations were applied sequentially, each building on the last:

#### 1. Mixed-precision state (ob-8qt.4)

The persistent GDN recurrent state (the "KV-cache equivalent") is halved from fp32 to fp16 or bf16,
while the accumulation loop stays fp32. This halves the persistent memory footprint: 48 MB → 24 MB
across 24 layers for Qwen3.5-4B.

**Critical precision constraint:** the decay accumulator must be fp32. A gate of 0.5 compounded
over 64 steps is ~5e-20, which underflows fp16. Only the *storage* is narrowed; the *arithmetic* is
identical to the fp32 kernel.

Software bf16 conversion (`f32_to_bf16_rne` with round-to-nearest-even) is used because Armv8.0
has no hardware bf16. NEON-vectorized conversion intrinsics were added in ob-8qt.5 to eliminate the
scalar bottleneck.

#### 2. OpenMP parallelization (ob-8qt.5)

The channel loop in each kernel is embarrassingly parallel — each channel's recurrence is
independent. With `-fopenmp` and `#pragma omp parallel for schedule(static)`, channel groups are
distributed across the 4 A57 cores.

This required restructuring the NEON loops from pointer-stride form
(`for (c = 0; c + 4 <= channels; c += 4)`) to counting form (`for (vi = 0; vi < n_vec; ++vi)`)
for OpenMP canonical-loop compatibility. The SVE paths were wrapped in block scopes to declare
`vl`/`n_vec` before the pragma.

#### 3. NEON double-width unrolling — scan & decay (ob-8qt.6)

The NEON path was widened from 4 channels/iter to 8 channels/iter (two independent `float32x4_t`
register groups). This creates two independent FMA dependency chains per iteration, which the A57's
out-of-order scheduler can interleave to hide the 4-cycle FMA/MUL latency.

With two chains and 4-cycle latency, the scheduler can keep 2 FMAs in flight — throughput of 0.5
FMA/cycle, up from 0.25 FMA/cycle with a single chain. A further doubling to 16-wide would need 4
chains for full 1.0 FMA/cycle utilization but would exhaust all 32 NEON registers.

#### 4. NEON double-width unrolling — conv (ob-8qt.7)

The same technique applied to the depthwise conv kernel. The conv benefits **more** from unrolling
than gated_scan/cumdecay because its dependency chain is 4-deep (four chained FMAs: `acc = h0*w0 +
h1*w1 + h2*w2 + cur*w3`) rather than 1-deep. With a single chain, 3 out of every 4 cycles are wasted
waiting for the chain to resolve; with two chains, the scheduler can interleave the two independent
4-deep chains.

### Cumulative results

**Qwen3.5-4B (C=4096, T=64):**

| Stage | cumdecay | gated_scan | conv |
|---|---:|---:|---:|
| Baseline (1 thread, NEON 4-wide) | 2108 μs | 3893 μs | 1961 μs |
| + OpenMP (4 cores) | 829 μs (2.5×) | 1501 μs (2.6×) | 912 μs (2.1×) |
| + 8-wide unroll (scan/decay) | 526 μs (4.0×) | 1003 μs (3.9×) | 901 μs (2.2×) |
| + 8-wide unroll (conv) | 510 μs (**4.1×**) | 1010 μs (**3.9×**) | 590 μs (**3.3×**) |
| Achieved bandwidth | 3.9 GiB/s | 2.9 GiB/s | 3.6 GiB/s |

**Qwen3.5-0.8B (C=2048, T=64):**

| Stage | cumdecay | gated_scan | conv |
|---|---:|---:|---:|
| Baseline | 593 μs | 1096 μs | 518 μs |
| + OpenMP | 335 μs (1.8×) | 454 μs (2.4×) | 321 μs (1.6×) |
| + 8-wide unroll (scan/decay) | 201 μs (2.9×) | 257 μs (4.3×) | 321 μs (1.6×) |
| + 8-wide unroll (conv) | 210 μs (**2.8×**) | 280 μs (**3.9×**) | 195 μs (**2.7×**) |

OpenMP's sublinear scaling (2.1–2.6× on 4 cores for 4B) reflects the A57's 1 MB shared L2 and the
~3 MiB working set that does not fit in cache — threads compete for DRAM bandwidth.

### Mixed-precision: what helps and what doesn't

At the final optimization level, the fp16/bf16 state variants were benchmarked:

| Variant | 4B p50 (μs) | vs fp32 | 0.8B p50 (μs) | vs fp32 |
|---|---:|---:|---:|---:|
| cumdecay (fp32 output) | 510 | — | 210 | — |
| cumdecay_f16 (fp16 output) | 352 | **1.45×** | 164 | **1.28×** |
| cumdecay_bf16 (bf16 output) | 360 | **1.42×** | 162 | **1.30×** |
| gated_scan (fp32 state) | 1010 | — | 280 | — |
| gated_scan_f16 | 1010 | **1.00×** | 297 | 0.94× |
| gated_scan_bf16 | 1010 | **1.00×** | 284 | 0.99× |

**cumdecay** benefits from half-precision output because the output array (`decay[]`) is
O(channels × seq) and dominates memory traffic — halving it directly halves bandwidth pressure.

**gated_scan** does not benefit because only the *state* is narrowed (O(channels)), not the scan
output `s[]` which is O(channels × seq) and stays fp32. The narrowing cost is negligible against the
inner loop's memory traffic, and the state load/store happens once per chunk boundary — not per time
step.

This is a useful design data point: **mixed precision helps the prefix-product but not the scan** on
this architecture. On a device with hardware bf16 or fp16 compute (Armv8.2-A+), the scan inner loop
itself could run at half precision with an fp32 master accumulator, which would help — but that
requires ISA extensions the A57 lacks.

### What limits each kernel

| Kernel | Arithmetic intensity | Bottleneck | Evidence |
|---|---|---|---|
| gated_scan | 2 FLOP / 12 bytes = 0.17 | FMA latency (4-cycle chain) | 0.52 GFLOP/s achieved vs 47.3 peak = 1.1%; 2.9 GiB/s vs 12.8 peak = 23% |
| cumdecay | 1 FLOP / 8 bytes = 0.13 | Memory bandwidth | 3.9 GiB/s = 30% of DRAM peak; MUL has lower latency than FMA |
| conv | 8 FLOP / 4 bytes = 2.0 | Compute (FMA throughput) | 3.6 GFLOP/s = 7.7% of peak; highest GiB/s because it reuses loaded weights 4× |

The conv is the most compute-intensive (8 FLOPs per element loaded, because 4 weight taps are reused
per input element), which is why the double-width unroll gave the largest relative speedup on it
(1.5–1.6×). The gated_scan and cumdecay are memory-latency-bound: the FMA dependency chain serializes
access, preventing the memory system from being saturated.

### Numerical fidelity of mixed precision

| Check | fp16 | bf16 |
|---|---|---|
| Single-call max_abs vs fp32 | 1.07 × 10⁻⁴ | 8.49 × 10⁻⁴ |
| 8-chunk drift worst | 4.07 × 10⁻⁴ | (not measured; expected similar) |
| Drift trend | Plateaus at chunk 2 — no compounding | — |

fp16 has lower max_abs (10 mantissa bits vs bf16's 7) because GDN state values stay in [−1, 1], where
fp16's extra mantissa precision matters more than bf16's wider exponent range. The drift plateaus
after the second chunk, confirming the narrowing error does not compound over successive chunks — the
fp32 accumulator absorbs the per-chunk rounding.

### Reproducing

```bash
# Build (on the device, native gcc)
./scripts/build_device_bench.sh    # outputs dist/bench_gdn_<variant>

# Set performance governor
for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance | sudo tee "$c"
done

# Run
./dist/bench_gdn_jetson_a57 --repeats 30 --csv > results/raw/jetson-j2-conv-unroll.csv

# Correctness
aarch64-linux-gnu-gcc -O3 -fopenmp -mcpu=cortex-a57 -static \
    src/orionsbelt/engines/cpu/kernels/test_gdn_sve.c \
    src/orionsbelt/engines/cpu/kernels/gdn_sve.c \
    -o dist/test_gdn_sve_jetson_a57 -lm && ./dist/test_gdn_sve_jetson_a57
```

Raw CSV files for each optimization stage are committed under `results/raw/jetson-j2*.csv`.

---

## 5. Device microbenchmark: ready to run, awaiting real silicon (2026-08-02)

**Bead `ob-8ms.2`.** The maintainer has Armv8 devices available, which unblocks real Arm
measurements without the Orion O6. This section records the apparatus; **it contains no
performance results yet**, deliberately.

### What exists

[`src/orionsbelt/engines/cpu/kernels/bench_gdn.c`](../src/orionsbelt/engines/cpu/kernels/bench_gdn.c)
is a dependency-free microbenchmark that links statically, so one binary copies to any aarch64
device and runs with no toolchain, no Python, and no shared libraries on the target. Built by
[`scripts/build_device_bench.sh`](../scripts/build_device_bench.sh) into four ISA variants
(`armv8a`, `armv8.2dot`, `armv8.6i8mm`, `armv9sve2`) because Arm devices vary enormously.

It times all three GDN kernels at verified Qwen3.5-4B and 0.8B shapes, follows
[`METRICS.md`](./METRICS.md)'s protocol (3 discarded warmups, N timed repeats, p50/p95, never a
single best run, refuses N<5), reports the **dispatch path the compiler actually selected**, and
emits CSV matching [`RESULTS_SCHEMA.md`](./RESULTS_SCHEMA.md). It also derives achieved GiB/s from
an explicit traffic accounting (`bytes_per_call` in the source) so the bandwidth-bound thesis can
be **tested against a device's spec bandwidth rather than assumed**.

Verified: the `armv9sve2` build reports `sve` and the `armv8a` build reports `neon`, so the
dispatch is genuinely selected rather than silently collapsing to one path.

### ⚠ QEMU timings are not measurements

The benchmark runs correctly under QEMU, and QEMU numbers were used only to prove the harness
works. **They are performance-meaningless** — QEMU emulates instruction by instruction, so the
~1.5 GiB/s and up-to-97% run-to-run spreads observed there are emulation artefacts. No QEMU
timing should ever appear in a result table. QEMU's legitimate role in this project is
*correctness* (`scripts/verify_cpu_kernels.sh`), not speed.

### What Armv8 devices can and cannot tell us

| Measurable on Armv8-A | Not measurable without newer/other hardware |
|---|---|
| **NEON kernel throughput** — the path most deployed Arm devices actually run | SVE/SVE2 throughput (absent on most Armv8; needs Armv9 or an SVE-capable Armv8) |
| **Achieved memory bandwidth** vs device spec — the bandwidth-bound thesis | i8mm int8 matmul (Armv8.6-A and later only) |
| **The memory decomposition** — KV cache vs recurrent state, which is architecture-independent and is the project's central claim | CIX NPU operator execution, Immortalis GPU compute |
| **Prefill vs decode asymmetry** | Engine-boundary dispatch latency (`ob-t3b.3`) |
| **big.LITTLE affinity effects**, where the device has asymmetric clusters | Arm Performix standardised reporting on the O6 |

The middle row is the important one: the KV-cache-versus-recurrent-state result does not depend on
the NPU, the GPU, or Armv9 at all. It is a property of the architecture, so **an Armv8 device can
demonstrate the project's central claim end to end** — which is what moves the Edge AI track from a
credible plan to a credible result.
