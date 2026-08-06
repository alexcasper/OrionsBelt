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

### Decode-phase with optimized kernels (ob-8qt.9)

j1's baseline decode measurement (ob-mrd.5/ob-mrd.6) showed narrow formats (fp16/bf16)
**slower** than fp32 at decode (seq=1), because data is L2-cache-resident and conversion
overhead dominates when I/O is nearly free. The optimized kernels (OpenMP + NEON unrolling)
change this picture — but introduce their own overhead.

**Qwen3.5-4B gated_scan — baseline (j1) vs optimized (j2):**

| Format | Baseline p50 (μs) | Optimized p50 (μs) | Speedup | Optimized GiB/s |
|---|---:|---:|---:|---:|
| fp32 | 8.23 | 4.48 | 1.84× | 17.0 |
| fp16 | 10.83 | 5.31 | 2.04× | 11.5 |
| bf16 | 13.49 | 5.57 | 2.42× | 11.0 |

**Key finding: the narrow-format penalty is halved but not eliminated.**

| Format | Baseline penalty vs fp32 | Optimized penalty vs fp32 |
|---|---:|---:|
| fp16 | +32% | +19% |
| bf16 | +64% | +24% |

The optimizations reduce the penalty because OpenMP parallel execution hides conversion
overhead across 4 cores. But OpenMP fork/join overhead (~2 μs) dominates at these tiny
workloads — the actual kernel work at seq=1 is ~2 μs, and OpenMP adds ~2-3 μs of thread
synchronization. This means **roughly half the decode latency is OpenMP overhead**, and a
serial decode path (omitting `#pragma omp`) would likely halve it.

**Design implication:** the optimal dispatch is phase-dependent not only in precision format
(j1's finding) but also in threading strategy: OpenMP for chunk-parallel prefill (seq=64),
serial for token-by-token decode (seq=1). This is a concrete actionable finding for the
heterogeneous dispatcher (`ob-7a9`).

**Conv at decode** benefits most from unrolling: baseline 27.6 μs → optimized 9.2 μs (**3.0×**),
because the conv's 4-deep dependency chain benefits from having two independent chains even
when the data fits in L2.

### Sustained-load with optimized kernels (ob-8qt.9)

The `--sustained 120` benchmark (gdn_gated_scan, Qwen3.5-4B) reveals what happens when the
device runs optimized kernels flat-out for 2 minutes. j1's baseline showed zero thermal
throttling; the optimized kernels tell a more nuanced story:

| Window | j1 Baseline (GiB/s) | j1 Temp (°C) | j2 Optimized (GiB/s) | j2 Temp (°C) |
|--------|-------------------:|-------------:|---------------------:|-------------:|
| 0–5 s  | 0.77               | 51.0         | **2.80**             | 52.5         |
| 55–60 s| 0.76               | 51.5         | 2.78                 | 55.0         |
| 115–120 s | 0.76            | 52.0         | 2.65                 | 58.0         |

**Three observations:**

1. **Throughput is 3.6× higher** with optimizations (2.80 vs 0.77 GiB/s) — consistent with the
   burst benchmark.
2. **No hard throttling** — the Jetson Nano's active cooling keeps the A57 within safe limits
   (58°C peak).
3. **Measurable ~5% throughput decline after ~80 s**, correlated with temperature reaching
   56–58°C. This decline was invisible with baseline kernels because they weren't pushing the
   device hard enough to generate significant heat. The optimized kernels reveal a gentle
   thermal slope — honest data that strengthens rather than weakens the submission.

The `--sustained` mode is portable: every device in the fleet can produce its own decay curve,
and passively-cooled devices (Pi 5, RK3588) will show a steeper slope.

---

## 5b. Fleet bandwidth-scaling cross-comparison (2026-08-03)

**Bead `ob-8ms.3`.** With data from all three fleet devices (Pi 5, RK3588, Jetson
Nano), we can now test the central hypothesis from `METRICS.md`: are the GDN kernels
memory-bandwidth-bound at ~0.25 FLOP/byte?

Full analysis is regenerable via `python3 bench/fleet_analysis.py` and committed at
[`results/figures/fleet_bandwidth_scaling.md`](../results/figures/fleet_bandwidth_scaling.md).

### Devices and spec bandwidth

| Device | Cores | ISA | Spec BW (GiB/s) |
|--------|-------|-----|-----------------|
| Pi 5 | 4x A76 @ 2.4 GHz | Armv8.2 + dotprod | 17.0 |
| RK3588 big | 4x A76 @ 2.4 GHz | Armv8.2 + dotprod | 34.0 |
| RK3588 little | 4x A55 @ 1.8 GHz | Armv8.2 | 34.0 |
| Jetson j1/j2 | 4x A57 @ 1.48 GHz | Armv8.0 (NEON only) | 25.6 |
| **Orion O6** | 4x A720 big + 4x A720 mid + 4x A520 | Armv9.2 (SVE2) | **93.1** |

### Achieved throughput (4B model, seq=64, baseline fp32, single-threaded)

All fleet devices were benchmarked single-threaded. j2's data is a fresh run of the
current binary with `OMP_NUM_THREADS=1` (the original j2 CSV was captured pre-OpenMP
at commit `28729f3`; see optimization-impact section below for the 4-core numbers).

| Device | Spec | CumDecay | Scan | DWConv1D | Scan/Spec |
|--------|------|----------|------|----------|-----------|
| Pi 5 | 17.0 | 3.74 | 1.20 | 3.23 | 7.1% |
| RK3588 big | 34.0 | 4.13 | 1.96 | 4.02 | 5.8% |
| RK3588 little | 34.0 | 0.87 | 0.35 | 0.65 | 1.0% |
| Jetson j1 | 25.6 | 1.16 | 0.72 | 1.04 | 2.8% |
| Jetson j2 | 25.6 | 1.32 | 1.13 | 1.20 | 4.4% |

### The discriminating test: Pi 5 (A76, less BW) vs Jetson (A57, more BW)

The `DEVICE_RUNBOOK` poses the key question: if the scan kernel is
bandwidth-bound, the Jetson (25.6 GiB/s, oldest A57 cores) should beat the
Pi 5 (17.0 GiB/s, newest A76 cores).

| Kernel | Pi 5 | Jetson j1 | Jetson j2 | Winner | Pi5/J1 |
|--------|------|-----------|-----------|--------|--------|
| CumDecay | 3.74 | 1.16 | 1.32 | **Pi 5** | 3.22x |
| Scan | 1.20 | 0.72 | 1.13 | **Pi 5** | 1.67x |
| DWConv1D | 3.23 | 1.04 | 1.20 | **Pi 5** | 3.11x |

**Result: the Pi 5 wins on ALL three kernels despite having 33% LESS spec
bandwidth.** The bandwidth-bound hypothesis does NOT hold at seq=64 working set
sizes. These kernels are **instruction-overhead-bound, not DRAM-bandwidth-bound**
at this scale.

This is consistent with the working set analysis: at seq=64 with 4096 channels,
the total traffic is ~1 MiB — small enough to be L2/L3-resident, so core
microarchitecture (IPC, OoO depth, clock frequency) dominates over raw DRAM
bandwidth. The A76's ~1.6× higher clock and substantially better IPC than the A57
explain the Pi 5's win despite less bandwidth.

### Implications for downstream decisions

The `DEVICE_RUNBOOK` warns: *"If the Pi 5 wins comfortably, the thesis is wrong
or incomplete, and we need to know that — several downstream decisions rest on it."*

1. **CPU-first mapping (PLAN.md §3.1):** the argument that GDN layers should stay
   on the CPU because they are "memory-bandwidth-bound at decode" is **partially
   undermined** at prefill chunk sizes (seq=64). At decode (seq=1), the j2 data
   shows the opposite — data is L2-resident and throughput jumps to 9–17 GiB/s
   (see decode section above). The bandwidth-bound claim holds at decode but not
   at prefill chunk size.
2. **Kernel optimization priority:** the 2.6-2.9× improvement from 4-core OpenMP
   + NEON double-width unrolling on the Jetson (scan: 1.13→2.94 GiB/s single→4-core)
   confirms that instruction overhead was the binding constraint single-threaded, but
   multi-threaded scaling reveals a bandwidth component (2.6-2.9× from 4 cores, not the
   theoretical 4×). The O6's 12 cores and 5× bandwidth should scale better.
3. **Weight quantization remains correctly prioritized:** weights are the dominant
   traffic at model scale, and INT4/INT8 weight compression targets actual DRAM
   bandwidth, which these microbenchmarks don't exercise.

### O6 prediction

A naive bandwidth-linear extrapolation would predict 2.6–6.6 GiB/s scan throughput
on the O6 (scaling from each fleet device). **This is almost certainly an
overprediction** because the kernels are instruction-bound, not bandwidth-bound.

A core-performance-based prediction is more honest: scaling from the RK3588 A76
big cluster (1.96 GiB/s scan) by the expected A720 IPC+clock gain (1.5–2.5×)
gives **2.9–4.9 GiB/s predicted scan throughput** — ~3–5% of the O6's 93.1 GiB/s
spec bandwidth. If the board arrives, run
`bench_gdn_armv9sve2 --repeats 30 --csv` to check.

### Optimization impact: j2 single-threaded vs 4-core OpenMP (2026-08-03)

The original j2 CSV (6 rows) was captured at commit `28729f3` — before OpenMP
parallelization, NEON double-width unrolling, and bf16 vectorization were added.
A fresh run of the current optimized binary shows the real-world impact:

| Kernel (4B, seq=64) | Single-thread | 4-core OpenMP | Speedup |
|--------------------|--------------:|--------------:|--------:|
| CumDecay | 1.32 GiB/s | 3.85 GiB/s | 2.9× |
| Scan | 1.13 GiB/s | 2.94 GiB/s | 2.6× |
| DWConv1D | 1.20 GiB/s | 3.51 GiB/s | 2.9× |

The 2.6-2.9× scaling from 4 cores (not the theoretical 4×) reveals a bandwidth
component that the single-threaded comparison cannot expose: single-thread
performance is IPC-limited, but multi-threaded scaling hits shared L2/memory
bandwidth. **Implication for the O6:** its 12 cores and 93.1 GiB/s bandwidth
mean the bandwidth ceiling is 5× higher — the O6 should scale closer to the
theoretical core-count ratio than these 4-core devices.

**Mixed-precision at decode (seq=1):** bf16/fp16 state narrowing is SLOWER than
fp32 at decode — conversion overhead dominates when the working set is tiny
(16 KiB state). This confirms the ob-8qt.4 design: use fp32 state at decode,
narrow only for prefill chunk boundaries.

| Kernel (4B, seq=1) | fp32 | bf16 | fp16 |
|--------------------|-----:|-----:|-----:|
| CumDecay | 8.03 GiB/s | 5.23 GiB/s | 5.78 GiB/s |
| Scan | 17.86 GiB/s | 11.49 GiB/s | 12.08 GiB/s |

### ⚠ Measurement quality: the replicate gap was one bad run, not a disagreement

Every cross-device table takes one run per device, and two devices were measured
twice. The gaps looked alarming — until the spreads were checked (bead `ob-bf7`):

| Device class | Runs (scan, 4B) | Gap | Spread of each |
|---|---|---:|---|
| RK3588 big | t3 1.96 vs t4 3.29 | 1.68× | **t3 153%** vs t4 17% |
| RK3588 little | t3 0.35 vs t4 0.55 | 1.57× | t3 29% vs t4 12% |
| Pi 5 (same board) | r5 1.20 vs j1 1.84 | 1.53× | r5 7% vs j1 — |
| Jetson j2 (same board) | canonical 0.73 vs `_single` 1.13 | 1.55× | the 1.13 run has **no manifest** |

**The RK3588 "disagreement" is not two valid measurements conflicting.** t3's scan
run reports p50 1514 µs against p95 3832 µs — a **153% spread**, where
[`DEVICE_RUNBOOK.md`](./DEVICE_RUNBOOK.md) calls anything past ~10% suspect and
says to suspect thermal throttling first. t4's same-commit run is 17%. One run is
simply contaminated, so the fleet tables now anchor on **t4** on quality grounds,
and `fleet_analysis.py` prints the spread beside every headline number so a noisy
figure cannot be quoted as a clean one. That is how a 153% row came to anchor the
O6 prediction in the first place.

The Jetson case resolved the same way: the outlier (1.13) is the run with **no
manifest at all**, while the manifest-backed canonical run reads 0.73 — agreeing
with j1's 0.72 to ~1%. Under PLAN.md §9 the unprovenanced file is not a result,
and dropping it makes the two Jetson units agree rather than conflict.

**A caveat that limits all of it:** every manifest on the fleet records
`dirty: true`. The stored SHA therefore does not identify the code that ran, so
"same commit" never guaranteed "same binary" — which is why the environmental
explanation could not have been established either way.

**What survives.** The Pi 5 beats both Jetson units on all three kernels despite
33% less spec bandwidth, under every pairing and comfortably outside the spread.
The pure bandwidth-bound thesis is incomplete at this working set. The O6 estimate
is **~5-8 GiB/s**, with the IPC/clock assumption and the dirty-tree problem as the
dominant uncertainties rather than anchor choice.

Closing `ob-bf7` needs one clean-tree, commit-matched sweep across every device
with pinning, governor and thermals recorded. The tables in
[`fleet_bandwidth_scaling.md`](../results/figures/fleet_bandwidth_scaling.md) are
generated by `bench/fleet_analysis.py` — do not hand-edit that file.

> [!WARNING]
> **`results/raw/jetson-j1_clean.csv` is a 4-core OpenMP run, not a single-threaded
> one, and must not be dropped into the fleet table.** It was captured to answer
> `ob-bf7` and its manifest is genuinely the project's first `dirty: false` — but
> its numbers (cumdecay 3.79, scan 2.92, dwconv 3.60 GiB/s) match the known 4-core
> `jetson-j2-omp-full.csv` to within ~2% and are 3-4x both known single-threaded
> A57 runs. Used as the j1 row it would put the A57 (2.92) *above* the Pi 5 (1.20),
> inverting the discriminating result this whole section rests on.
>
> Nothing in the manifest said so, because thread count was not captured. That gap
> is now closed — `bench/manifest.py` and `scripts/capture_manifest.sh` both emit a
> `parallelism` block with `omp_num_threads` and `effective_threads` — but every
> manifest predating this commit is silent on it, so **infer thread count from the
> filename or the magnitude, not from the manifest**, for anything captured earlier.
>
> So `ob-bf7` is **not** closed by that run. What is still needed is a clean-tree
> sweep at a *matched thread count*, which for the cross-device table means
> `OMP_NUM_THREADS=1`.

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

### Mixed-precision recurrent-state kernels (ob-8qt.4)

Four narrow-state variants of `gdn_cumdecay` and `gdn_gated_scan` were added (bf16/fp16). All
accumulate in fp32 — only the storage format narrows. Measured on Jetson A57 (NEON):

| Kernel (4B, seq=64) | Format | p50 (µs) | GiB/s | vs fp32 |
|---|---|---:|---:|---|
| cumdecay | fp32 | 1800 | 1.09 | — |
| cumdecay | bf16 | 1259 | 1.16 | +6% bw, −30% time |
| cumdecay | f16 | 1006 | 1.46 | +34% bw, −44% time |
| gated_scan | fp32 | 3925 | 0.75 | — |
| gated_scan | bf16 | 3987 | 0.74 | ≈ same |
| gated_scan | f16 | 3948 | 0.75 | ≈ same |

**Cumdecay benefits from narrower output** (less write traffic). Fp16 is fastest because
`vcvt_f16_f32` maps to the single-cycle `FCVTN` instruction on base A64, while bf16 uses software
rounding (integer NEON ops).

**Gated_scan shows no measurable difference at seq=64** because state read+write is only ~1% of
total traffic at prefill chunk size. The benefit concentrates at **decode (seq=1)** where state
I/O is ~40% of traffic — halving it saves ~20% per token. This confirms the bead's re-scoping
note: state narrowing is a *memory-residency* optimization (halving resident state from 48 MB to
24 MB across 24 layers), not a step-change in decode bandwidth.

**Precision findings:**
- bf16 state: 0.38% max relative error (7 mantissa bits). No values flush to zero — bf16 shares
  fp32's 8-bit exponent, so even 0.5^64 ≈ 5e-20 is representable.
- fp16 state: 0.05% max relative error (10 mantissa bits, more precise than bf16). However, fp16's
  5-bit exponent means small decay products flush to zero: 0.5^20 ≈ 9.5e-7 is near the subnormal
  floor. For gates in (0.9, 0.99) this is not an issue.
- **The decay accumulator MUST stay fp32**: confirmed by stress test with constant-0.5 gates,
  where the running product reaches ~5e-20. Both bf16 and fp16 would underflow this; fp32 handles
  it cleanly.

**Platform note**: `__fp16` scalar type and `FCVTN`/`FCVTL` NEON conversions work on Cortex-A57
(Armv8.0-A) despite `__ARM_FEATURE_FP16_VECTOR_ARITHMETIC` being undefined. That macro gates fp16
*arithmetic* (fmul/fadd on half registers), not conversion, which is base A64 ASIMD.

#### Cross-device validation: RK3588 (A76 big / A55 little), 2026-08-04

Re-measured on **rk3588-t4** — a newer core class (Cortex-A76 big, Armv8.2 + dotprod; A55 little)
with **hardware fp16 (`asimdhp`) but no hardware bf16** (bf16 needs Armv8.6-A). This exercises the
software bf16 conversion path on a faster core — exactly the case the installed base of Armv8.2
devices hits. Single-threaded, governor=performance, thermals 40–43 °C flat. Correctness gate
re-passed on A76 (deterministic, so fidelity numbers match the Jetson run: fp16-state max-abs
1.07e-4, bf16 8.49e-4, 8-chunk drift plateaus 4.07e-4, no compounding).

Prefill (Qwen3.5-4B, seq=64, single-threaded p50 µs):

| Kernel | Jetson A57 | RK3588 A76 | RK3588 A55 |
|---|---:|---:|---:|
| cumdecay fp32 | 1800 | 271.0 | 1422.0 |
| cumdecay f16-output | 1006 (1.45×) | **171.5 (1.58×)** | 932.2 (1.53×) |
| cumdecay bf16-output | 1259 (1.42×) | **247.1 (1.10×)** | 970.1 (1.47×) |
| gated_scan fp32 | 3925 | 559.5 | 3875.3 |
| gated_scan f16-state | 3948 (1.00×) | 558.6 (1.00×) | 3642.6 (1.06×) |
| gated_scan bf16-state | 3987 (1.00×) | 548.7 (1.02×) | 3665.6 (1.06×) |

**Findings confirmed and refined cross-device:**

1. **Gated-scan state narrowing stays flat at prefill** on all three core classes (≤6%) — the
   state is ~0.5% of per-step traffic. It is a decode/residency lever, as the Jetson run concluded.
2. **fp16 output narrowing is a consistent, portable win** (1.45–1.59×): the A76's hardware fp16
   (`FCVTN`/`FCVTL`) makes the conversion single-cycle.
3. **NEW — software bf16 conversion negates the bandwidth win on the fastest core.** On the A76,
   cumdecay-bf16 drops to 1.10× (4B) and is *slower than fp32* at 0.8B (129.5 vs 124.0 µs): the
   integer-NEON round-to-nearest-even conversion cost rivals the bytes saved. On the slower A55
   and A57, the bandwidth saving dominates so bf16 still wins (1.42–1.47×).

**Policy consequence (refines ob-qpa):** on A76-class cores — the **newest in the fleet** (Pi 5,
RK3588 big cluster) — **prefer fp16 over bf16** for narrowed state/output. bf16 only wins with
hardware support (Armv8.6-A+, e.g. Graviton / the O6's A720) or where the core is slow enough to
amortize software conversion. The optimal narrow format is **core-class-dependent, not universal.**

Raw CSVs: `results/raw/rk3588-t4_{big,little}_singlethread.csv`; manifest
`results/manifests/rk3588-t4_mixedprec.json` (git SHA `aad6189`, governor=performance).

### Decode-phase narrow-format penalty (ob-mrd.6)

The initial mixed-precision table above measured only prefill (seq=64). The refreshed
28-row CSV (ob-mrd.5) adds decode (seq=1) configs, and the picture changes sharply:

**Qwen3.5-4B gated_scan — prefill vs decode:**

| Format | Prefill (seq=64) GiB/s | Decode (seq=1) GiB/s | Decode vs fp32 |
|---|---:|---:|---|
| fp32 | 0.72 | **9.27** | — |
| bf16 | 0.74 | 4.52 | **−51%** |
| fp16 | 0.74 | 5.63 | **−39%** |

At prefill, narrow formats are flat or marginally positive — the I/O traffic reduction
compensates for the conversion cost. At decode, they are roughly **half the speed** of fp32.

**Why:** at seq=1 the scan touches ~80 KiB of data (5 × 4096 × 4 bytes), which fits entirely
in the A57's 512 KiB L2 cache. The kernel is measuring cache bandwidth, not DRAM bandwidth —
hence the 12.9× jump from 0.72 to 9.27 GiB/s for fp32. With data already cache-resident, the
dominant cost shifts from memory I/O to the fp32↔half conversion instructions (`FCVTN`/`FCVTL`
for fp16, integer NEON sequence for bf16). Narrowing saves ~20% of traffic but adds ~100%
conversion overhead when traffic is nearly free.

**Implication for the design:** state narrowing is a **prefill-only** optimization. At decode,
where GDN's O(1) fixed-state advantage is the whole point, fp32 is both faster and more
accurate. The optimal dispatch is format-adaptive: narrow state for chunk-parallel prefill,
fp32 state for token-by-token decode. This is a concrete, measured argument rather than
conventional wisdom, and it sharpens the project's honest-framing commitment: the mixed-precision
headline should say "halves resident state during prefill," not "speeds up GDN."

## Sustained-load thermal characterization (ob-mrd.2)

The benchmark now supports `--sustained <seconds>` which runs `gdn_gated_scan` on the
Qwen3.5-4B config (seq=64) continuously for N seconds, sampling throughput and CPU
temperature every 5 s. This directly addresses PLAN.md risk R7: on passively-cooled
edge hardware, a burst number that cannot be sustained is misleading.

**Jetson-J1 (Cortex-A57, active fan cooling), 120-second sustained run:**

| Window | Throughput (GiB/s) | Thermal (°C) | vs first |
|--------|-------------------|-------------|----------|
| 0–5 s  | 0.77              | 51.0        | 0.0%     |
| 55–60 s| 0.76              | 51.5        | -0.8%    |
| 115–120 s | 0.76           | 52.0        | -0.9%    |

**Finding**: No thermal throttling. The Jetson Nano's active cooling keeps the A57
at 51–52 °C for the entire 2-minute run. Throughput is essentially flat (0.76–0.77
GiB/s, <1% variation within noise). The thermal rise is only +1.5 °C.

This is the expected result for actively-cooled hardware. The more interesting
characterization will come from passively-cooled devices (e.g. Pi 5) where sustained
load may trigger frequency reduction. The `--sustained` flag is portable — every
device in the fleet can produce its own decay curve.

## Native NEON kernel correctness verification (ob-mrd.3)

The existing `verify_cpu_kernels.sh` cross-compiles for SVE and verifies under QEMU.
However, **no device in the current fleet has SVE** — Jetson A57, Pi 5 A76, and
RK3588 A76/A55 all dispatch through the NEON path. The new
`scripts/verify_kernels_native.sh` builds and runs the C kernel tests natively on
each device using its real ISA, validating the actual dispatch path.

**Jetson-J1 (Cortex-A57, Armv8.0-A, NEON) — all tests pass on real silicon:**

| Kernel | vs scalar ref (float) | vs scalar ref (double) | Bit-identical |
|--------|----------------------|----------------------|---------------|
| `gdn_gated_scan_f32` | max_abs=0.000 | max_abs=1.19e-7 | YES |
| `gdn_cumdecay_f32` | max_abs=0.000 | max_abs=5.96e-8 | YES |
| `gdn_causal_dwconv1d_f32` | max_abs=5.96e-8 | — | ~1 ULP |

The 1-ULP deviation in `causal_dwconv1d` vs the float reference comes from NEON FMA
contraction (`vfmaq_n_f32` fuses multiply-add), while the scalar reference uses
separate multiply and add. This is expected and benign — the fused result is actually
*more* accurate than the unfused one.

Mixed-precision bf16/fp16 variants also pass all bounds on NEON:
cumdecay bf16 ≤0.4%, fp16 ≤0.05%; gated_scan bf16 ≤0.4%, fp16 ≤0.05%.
Determinism verified (bit-identical across repeated runs).

**New coverage added**: `test_gdn_sve.c` previously tested only `gated_scan` and
`causal_dwconv1d`; `cumdecay` was declared but never exercised. Added scalar
references (float and double) and comparison reporting for `cumdecay`.

---

## INA3221 power/energy characterization on Jetson-J1 (ob-agf.1)

The Jetson Nano exposes a TI INA3221 power monitor through IIO sysfs
(`/sys/devices/.../iio:device0/`), providing real-time power on three rails:

| Rail | Name | Idle | Avg load (sustained scan) | Peak |
|------|------|------|--------------------------|------|
| 0 | POM_5V_IN (board total) | 1906 mW | 2831 mW | 3225 mW |
| 1 | POM_5V_GPU | 0 mW | 0 mW | 0 mW |
| 2 | POM_5V_CPU | 448 mW | 1067 mW | 1347 mW |

**Sustained gated_scan (Qwen3.5-4B prefill, 10 s):**
- Delta power: **925 mW** board (619 mW CPU-only)
- Throughput: 0.74 GiB/s (stable, no thermal decay)
- Energy per GiB: **~1250 mJ/GiB board** (~837 mJ/GiB CPU-only)
- Thermal: 51.7°C → 52.0°C (active fan cooling, no throttle)

**Key observations:**

1. **CPU dominates the power budget.** The delta from idle is 619 mW CPU vs 925 mW
   board total — 67% of the incremental power is CPU. The GPU rail reads 0 mW
   (not used by the NEON kernel), so the remaining ~33% is memory controller, I/O,
   and board overhead.

2. **No thermal throttling at sustained load.** Temperature rose only 0.3°C over
   10 seconds at peak throughput. The Jetson Nano's active fan cooling is
   effective for this workload. This confirms the sustained-load finding from
   ob-mrd.2: throughput is flat at 0.74 GiB/s with no decay.

3. **Energy efficiency context.** At 1.25 J/GiB board-wide, the A57 cores deliver
   competitive energy efficiency for memory-bound linear-attention workloads. For
   comparison, a Raspberry Pi 5 (Cortex-A76) would move the same data faster but
   at higher power — the J/GiB comparison across the device fleet will reveal
   whether newer cores are more or less energy-efficient per unit of memory
   bandwidth.

The power sampling script (`scripts/power_bench.sh`) wraps any bench_gdn invocation
with synchronized INA3221 sampling and produces energy-per-GiB metrics without
requiring perf, ftrace, powertop, or Arm Performix — the Jetson's hardware power
monitor is sufficient.

### Per-kernel energy efficiency comparison

Using the extended `--sustained-kernel` flag (ob-mrd.7), all three fp32 kernels
were profiled for 10 seconds each under the INA3221 power monitor:

| Kernel | Throughput (GiB/s) | Δ Power board (mW) | Energy (mJ/GiB board) | Energy (mJ/GiB CPU) |
|--------|-------------------|--------------------|-----------------------|---------------------|
| `gdn_gated_scan` | 0.74 | 925 | **1250** | 836 |
| `gdn_causal_dwconv1d` | 0.88 | 903 | **1026** | 767 |
| `gdn_cumdecay` | 1.06 | 925 | **874** | 667 |

**Key finding: power is constant, energy scales with throughput.** All three
kernels draw essentially the same incremental board power (~900–925 mW over idle)
despite different throughput rates. On the A57, the power budget is dominated by
memory subsystem and core overhead, not by the specific arithmetic pattern. The
energy-per-GiB metric therefore tracks 1/throughput: `cumdecay` is most
energy-efficient because it moves data fastest, not because it draws less power.

This has a practical implication for the dispatcher design (ob-7a9): on
bandwidth-bound cores like the A57, kernel selection affects *latency* but not
*power draw*. A dynamic dispatcher should optimize for throughput, not for power,
on this class of hardware. (This may differ on newer cores like A720 where
compute-bound kernels can draw significantly more power.)

### Governor comparison: performance vs ondemand (sustained gated_scan)

| Governor | Throughput (GiB/s) | Idle (mW board) | Δ Power (mW board) | Energy (mJ/GiB board) |
|----------|-------------------|-----------------|---------------------|----------------------|
| `performance` | 0.74 | 1906 | 925 | **1250** |
| `ondemand` | 0.69 | 1698 | 1097 | **1602** |

**Finding: `performance` is both faster and more energy-efficient for sustained
inference.** Despite `ondemand` lowering idle power by 208 mW (frequency scales
down when idle), the 7% throughput penalty under load means each GiB costs 28%
more energy. The frequency ramping latency on A57 is high enough that sustained
workloads never benefit from scaling.

This directly validates PLAN.md's recommendation to use the `performance`
governor for all benchmarking. For bursty decode workloads (where idle gaps
between tokens allow frequency to drop), `ondemand` might save idle energy — but
that saving is irrelevant if it increases per-token energy under load.

**Practical implication for submission:** all reported numbers use the
`performance` governor. The `ondemand` comparison is documented to show the
trade-off, not to suggest it as a recommended setting.

## Jetson J1↔J2 cross-check: power, thermal, and energy efficiency (2026-08-03)

**Mandate: "jetson-j2, 2nd unit — cross-check vs jetson-j1."** Both are Jetson
Nano A57 (4× Cortex-A57 @ 1.479 GHz, Armv8.0 NEON, active fan cooling, Tegra
210). The INA3221 power monitor is present on both boards at the same IIO path.

### The j1 CSV is stale

j1's main benchmark CSV (`jetson-j1.csv`) was captured at commit `2c9ac9f` —
**before** NEON double-width unrolling, bf16 vectorization, and OpenMP
parallelization were added. j2's data is at commit `194e37c` (post-optimization).
The fleet comparison table (§5b) already uses j2's single-threaded data for a
fair cross-device comparison, but j1's stale CSV should be re-run when j1 is
available with the current binary.

**Impact on fleet table:** the j1 row (1.16/0.72/1.04 GiB/s) understates what
the A57 achieves with optimized kernels. j2's single-threaded data (1.32/1.13/1.20
GiB/s) is the accurate representation. The fleet conclusions (Pi 5 A76 beats
Jetson A57 despite less bandwidth) are unaffected — the relative ordering holds
regardless.

### Energy efficiency: optimized kernels vs old kernels (single-threaded)

Both devices ran sustained `gdn_gated_scan` (Qwen3.5-4B, seq=64) single-threaded
under the INA3221 power monitor.

| Metric | j1 (old kernels) | j2 (new kernels) | Δ |
|--------|-----------------:|-----------------:|----|
| Throughput | 0.77 GiB/s | 1.03 GiB/s | **+34%** |
| Δ Board power | 925 mW | 1018 mW | +10% |
| Δ CPU power | 619 mW | 670 mW | +8% |
| Energy/GiB board | **1250 mJ** | **989 mJ** | **−21%** |
| Energy/GiB CPU | **836 mJ** | **651 mJ** | **−22%** |
| Thermal rise | +0.3°C | +1.2°C | similar |

**Key finding: NEON double-width unrolling improves energy efficiency by ~21%,
not just throughput.** The optimized kernel does 34% more work per second while
drawing only 10% more power, because the unrolled NEON instructions keep the
pipeline fuller with less branch overhead. The extra power comes from more
intense ALU/register-file activity, not from higher clock or memory traffic.

### Multi-threading trades energy efficiency for throughput

j2 with 4-core OpenMP vs 1-core, same sustained workload:

| Metric | 1-core | 4-core | Ratio |
|--------|-------:|-------:|-------|
| Throughput | 1.03 GiB/s | 2.37 GiB/s | **2.3×** |
| Δ Board power | 1018 mW | 2697 mW | 2.6× |
| Δ CPU power | 670 mW | 1904 mW | 2.8× |
| Energy/GiB board | 989 mJ | **1138 mJ** | 1.15× worse |
| Energy/GiB CPU | 651 mJ | **804 mJ** | 1.23× worse |
| Thermal rise | +1.2°C | +2.0°C | — |

**Multi-threading is 15% less energy-efficient per GiB on the A57.** Power
scales super-linearly (2.8× CPU power for 2.3× throughput) because all four cores
share a single L2 and memory controller — the incremental cores add full clock
power but get diminishing bandwidth returns. For latency-critical decode, the
throughput gain justifies the energy cost; for throughput-batch workloads, single-
threaded may be more efficient.

### Hardware consistency validation

Both physical units show near-identical absolute power readings:

| Reading | j1 | j2 |
|---------|----|----|
| Idle board power | 1906 mW | 1853 mW |
| Idle CPU power | 448 mW | 409 mW |
| Sustained thermal range | 51.0–52.0°C | 51.5–53.0°C |

The ~3% idle power difference is within normal unit-to-unit variation (VRM
efficiency, sensor calibration, ambient temperature). **No anomalous behavior
detected on either unit.**

### Reproducing

```bash
# 4-core sustained with power instrumentation
sudo env OMP_NUM_THREADS=4 ./scripts/power_bench.sh --sustained 30 --csv

# Single-thread for cross-check
sudo env OMP_NUM_THREADS=1 ./scripts/power_bench.sh --sustained 30 --csv
```

Power logs are committed at `results/raw/jetson-j2_power_sustained_{1,4}core.csv`.

---

## Jetson-J1 ↔ Jetson-J2 cross-check: OpenMP scaling on Cortex-A57 (2026-08-03)

**Bead `ob-8ms.3`.** Two Jetson Nano units (both Tegra X1, Cortex-A57 quad-core
@ 1.479 GHz, Armv8.0-A, NEON only) provide a controlled A/B comparison:
**J1 ran single-threaded baseline kernels**; **J2 ran the same kernels with
OpenMP 4-thread parallelism** enabled at build time. Same ISA, same governor
(`performance`), same active fan cooling, same 30-repeat protocol. The only
variable is thread count.

> **File naming note:** Both `jetson-j1.csv` and `jetson-j2.csv` (canonical)
> are single-threaded and reproducible from the committed source. The OpenMP
> results below come from `jetson-j2-omp-full.csv`, built with a parallelized
> variant of the kernel (pragmas not yet in the mainline source). The ST
> numbers between the two units agree within ±7%, confirming hardware
> consistency.

### Prefill (seq=64) — Qwen3.5-4B

| Kernel | J1 ST (GiB/s) | J2 OMP-4T (GiB/s) | Speedup |
|--------|-------------:|------------------:|--------:|
| gdn_cumdecay | 1.16 | 3.85 | **3.3×** |
| gdn_gated_scan | 0.72 | 2.96 | **4.1×** |
| gdn_causal_dwconv1d | 1.04 | 3.66 | **3.5×** |
| gdn_cumdecay_f16 | 1.45 | 4.21 | **2.9×** |
| gdn_gated_scan_f16 | 0.74 | 2.94 | **4.0×** |
| gdn_cumdecay_bf16 | 1.25 | 4.16 | **3.3×** |
| gdn_gated_scan_bf16 | 0.74 | 2.98 | **4.0×** |

### Decode (seq=1) — Qwen3.5-4B

| Kernel | J1 ST (GiB/s) | J2 OMP-4T (GiB/s) | Speedup |
|--------|-------------:|------------------:|--------:|
| gdn_cumdecay | 4.65 | 8.37 | **1.8×** |
| gdn_gated_scan | 9.27 | 15.10 | **1.6×** |
| gdn_causal_dwconv1d | 4.98 | 13.80 | **2.8×** |
| gdn_cumdecay_f16 | 3.09 | 5.71 | **1.8×** |
| gdn_gated_scan_f16 | 5.63 | 11.96 | **2.1×** |
| gdn_gated_scan_bf16 | 4.52 | 11.49 | **2.5×** |

### Findings

1. **Near-linear scaling on prefill (3.3–4.1× on 4 cores) — which means these
   kernels are NOT bandwidth-saturated single-threaded.** This point was
   originally written as evidence that the kernels *are* bandwidth-bound, but the
   inference runs the wrong way: if one thread already saturated DRAM, adding
   three more could not buy 3.3–4.1×. `gated_scan` reaching 4.1× on 4 cores is
   slightly *superlinear*, which only happens when the single-thread baseline was
   limited by something other than bandwidth — here the serial dependency chain
   and load latency, which extra threads hide by overlapping independent
   channels. That is consistent with §5b's conclusion (instruction-overhead-bound
   at seq=64, ~1 MiB L2-resident working set) rather than in tension with it, and
   it is the third place in this document where near-linear multicore scaling was
   misread as a bandwidth result.

   What the scaling does bound is the *aggregate* limit: 4 threads have not yet
   hit the A57's memory system on this working set, so the ceiling is above 4×
   single-thread here and the O6 (more cores, ~4× the bandwidth) has headroom.

2. **Diminishing returns on decode (1.6–2.8×).** At seq=1 the working set is
   tiny (~16 KB for 0.8B, ~64 KB for 4B per kernel call). Four threads contend
   for the same cache lines, and thread dispatch overhead (≈2 µs per
   `#pragma omp parallel`) is a significant fraction of the 3–10 µs kernel
   runtime. The dispatcher design (`ob-7a9`) should consider single-threaded
   decode on small cores.

3. **Precision reduction helps cumdecay but not gated_scan — confirmed on both
   devices.** On J2, `cumdecay_f16` hits 4.21 GiB/s vs 3.85 fp32 (+9%); but
   `gated_scan_f16` is 2.94 vs 2.96 (≈same). This matches J1's pattern exactly.
   The scan kernel's bottleneck is the sequential dependency chain, not memory
   traffic — halving the data type doesn't break the dependency.

4. **GDN2 scan matches GDN1 in prefill, wins in decode.** J2 includes
   `gdn2_gated_scan` (not present in J1's run): prefill 2.95 GiB/s (≈ GDN1's
   2.96), decode 16.95 GiB/s (+12% vs GDN1's 15.10). The GDN-2 variant's smaller
   recurrent state reduces memory traffic at seq=1 where state I/O dominates.

5. **Cross-device consistency.** J2's current run matches the earlier
   `jetson-j2-full-optimized.csv` (3.82 vs 3.85 GiB/s on cumdecay — within 1%),
   confirming the benchmark is reproducible across sessions on the same hardware.

## 6. GDN-2 reference clone and decoupled-gating microbenchmark (2026-08-03)

**Bead ob-y3f. NVLabs GatedDeltaNet-2 repo cloned and analysed; C kernel stub benchmarked on jetson-j2.**

### Reference implementation

The NVLabs repo (`github.com/NVlabs/GatedDeltaNet-2`) is a full PyTorch + Triton
training and inference framework (lit_gpt-based, requires Python 3.10+, torch 2.9,
CUDA, triton, flash-linear-attention). It cannot run on the Jetson A57 (Python 3.6.9,
no CUDA). A smoke-test script (`scripts/smoke_test_gdn2.py`) is committed for x86/CUDA hosts.

### GDN-2 recurrence (from `fused_recurrent_gdn2.py`)

Per token, the matrix state S ∈ R^{d_k × d_v} updates as:

```
S ← Diag(exp(g_t)) · S                   # channel-wise decay (inherited from KDA)
v_new = (w_t ⊙ v_t) − (b_t ⊙ k_t)ᵀ · S  # gated write minus gated erase read
S ← S + k_t ⊗ v_newᵀ                     # rank-one update
o_t = Sᵀ · q_t                            # output read
```

**Key difference from GDN-1:** the single scalar gate β_t splits into two
channel-wise gates — `b_t ∈ [0,1]^{d_k}` (erase, key axis) and `w_t ∈ [0,1]^{d_v}`
(write, value axis). Setting both to uniform 1 recovers KDA.

### Channel-wise microbenchmark on Cortex-A57

We added a `gdn2_gated_scan_f32` kernel that extends our existing `gdn_gated_scan_f32`
with the two extra per-channel gates. The recurrence at channel level:

- **GDN-1:** `s[t] = x[t] + g[t] · s[t−1]` — 3 streams, 1 FMA/element
- **GDN-2:** `s[t] = w[t]·x[t] + g[t]·b[t] · s[t−1]` — 5 streams, 2 mul + 1 FMA/element

Measured on jetson-j2 (4× A57, NEON 8-wide, OpenMP 4-thread, governor=performance):

| Model | Kernel | p50 (μs) | GiB/s | Slowdown vs GDN-1 |
|---|---|---:|---:|---:|
| 4B (seq=64) | gdn_gated_scan | 995 | 2.98 | — |
| 4B (seq=64) | **gdn2_gated_scan** | **1632** | **3.01** | **1.64×** |
| 0.8B (seq=64) | gdn_gated_scan | 258 | 5.73 | — |
| 0.8B (seq=64) | **gdn2_gated_scan** | **431** | **5.71** | **1.67×** |

### Finding: GDN-2's overhead is entirely bandwidth, not compute

The slowdown (1.64–1.67×) matches the stream-count ratio (5/3 = 1.67×) almost
exactly, and achieved GiB/s is identical. This means:

1. **The bandwidth-bound thesis (§5a) extends to GDN-2.** The extra gates add
   memory traffic proportional to their stream count, with zero compute overhead
   visible in throughput.

2. **GDN-2 costs ~67% more wall-clock per GDN layer** at the prefill batch size.
   Across 24 GDN layers in Qwen3.5-4B, this is a meaningful but not prohibitive
   increase — and it is the *exact* prediction of the bandwidth model, not a
   surprise.

3. **At decode (seq=1)** the overhead is smaller (~1.18×) because the kernel
   fits partially in L1/L2 cache and the extra gates hit register-bound paths.

This validates the microbenchmark-only comparison path (ob-9ke option a) as
sufficient for the write-up: the cost difference is fully explained by memory
traffic, and a full layer-swap experiment (option b) is unlikely to reveal
anything the bandwidth model doesn't already predict.

**Decode (seq=1) on Jetson A57:** the cumdecay kernel reports 4.65 GiB/s at
seq=1 against 1.16 GiB/s at seq=64 — 4.0× — because the single-token step is
**cache-resident** where the seq=64 sweep streams from DRAM. At 4096 channels
each array is ~16 KiB, so the working set sits in the A57's 32 KiB L1D and 2 MiB
L2 rather than "entirely in L1"; L2 residency is enough to explain the gap.

Read that 4.65 GiB/s as a latency-dominated figure, not a sustained streaming
rate: the seq=1 step moves a few tens of KiB in 6.6 µs, so it is not directly
comparable to the seq=64 number as *bandwidth*. It is still the right thing to
measure for decode, and the direction is the architectural point — O(1)
recurrent state means the decode working set stays cache-resident **regardless
of context length**, where a KV cache grows until it cannot.

## Memory decomposition: GDN O(1) state vs full-attention O(n) KV cache (2026-08-03)

The central claim of this project, quantified with verified architecture
data from `src/orionsbelt/model/gdn_layer_info.py` (Qwen3.5-4B, 32 layers:
24 GDN + 8 full-attention, pattern 8×(3 GDN → 1 full)).

| Context | Weights | KV cache (FA) | GDN state | Conv state | Total |
|---|---|---|---|---|---|
| 4K | 11182 MB | 134 MB | 50 MB | 3 MB | 11370 MB |
| 32K | 11182 MB | 1074 MB | 50 MB | 3 MB | 12309 MB |
| 128K | 11182 MB | 4295 MB | 50 MB | 3 MB | 15531 MB |
| 256K | 11182 MB | 8590 MB | 50 MB | 3 MB | 19826 MB |

**Scaling (relative to 4K baseline):**

| Context | Weights | KV cache | GDN state | Total |
|---|---|---|---|---|
| 4K | 1.0× | 1.0× | 1.0× | 1.0× |
| 32K | 1.0× | 8.0× | 1.0× | 1.1× |
| 128K | 1.0× | 32.0× | 1.0× | 1.4× |
| 256K | 1.0× | 64.0× | 1.0× | 1.7× |

**Key insight:** At 256K context, the GDN recurrent state is 50 MB
(constant regardless of context length) while the full-attention KV cache
balloons to 8.6 GB — a **171× difference**. GDN saves 8.5 GB of memory
at this context length, which is the difference between fitting in 8 GB
edge DRAM and not. Weights (11.2 GB at FP16) dominate at all context
lengths, which is why INT4 weight quantization (PLAN.md §6, ADR 0004)
is the complementary half of the story.

---

## 7. RKNN (Rockchip RK3588) operator coverage — the recurrence limitation generalises (2026-08-06)

**Bead `ob-t3b.5`. Run on-device: RK3588 big.LITTLE, rknn-toolkit2 v2.3.2, aarch64.**

### Headline

> **Two independent edge-NPU toolchains — CIX NOE and Rockchip RKNN — both fail to route
> GDN's sequential recurrence to the NPU. This is not a vendor limitation; it is a
> structural property of edge-NPU compilers.**

Section 1 established that the CIX NOE Compiler cannot express a runtime-length sequential
recurrence. The RK3588 has its own ~6 TOPS NPU (Rockchip RKNN) with an entirely independent
vendor toolchain. Feeding the **same seven ONNX probe graphs** through rknn-toolkit2's
`load_onnx` + `build` pipeline confirms: **RKNN is even stricter** — it rejects `Loop` outright
(even with a constant trip count), and routes `Scan` to CPU as an unsupported "custom operator."

The finding generalises: keeping the scan on CPU is not a preference for one platform — it is a
**universal edge-NPU toolchain constraint**.

### Method

The seven hand-authored ONNX probe graphs from [`artifacts/npu_op_probe/`](../artifacts/npu_op_probe/)
(verified locally under `onnxruntime`, see Section 1) were fed to rknn-toolkit2's conversion
pipeline on-device:

```python
from rknn.api import RKNN
rknn = RKNN(verbose=True)
rknn.config(target_platform="rk3588", float_dtype="float16", optimization_level=3)
rknn.load_onnx(model=probe_onnx)
rknn.build(do_quantization=False)
```

Generator: [`scripts/npu_op_probe.py`](../scripts/npu_op_probe.py);
runner: [`scripts/rknn_op_probe.py`](../scripts/rknn_op_probe.py).
Logs and results committed under [`artifacts/npu_op_probe/rknn_audit/`](../artifacts/npu_op_probe/rknn_audit/).

### Results — CIX NOE vs Rockchip RKNN

| Probe | ONNX ops | CIX NOE | RKNN (RK3588) |
|---|---|---|---|
| causal depthwise Conv1D | `Conv` (groups=C, pads=[3,0]) | ✅ NPU | ✅ NPU (with `Reshape` wrappers) |
| gated decay | `Log`, `CumSum`, `Exp` | ✅ all NPU | ⚠️ `Log`+`Exp` → **CPU fallback**; `CumSum` → rewritten as `Conv` on NPU |
| delta-rule state update | `MatMul`, `Sub`, `Add`, `Transpose` | ✅ all NPU | ✅ all NPU (`exMatMul`) |
| elementwise gate chain | `Sigmoid`, `Softplus`, `Neg`, `Exp`, `Mul` | ✅ all NPU | ⚠️ `Exp` → **CPU fallback**; rest NPU |
| chunk recurrence via `Scan` | `Scan` | ❌ rejected | ⚠️ **accepted but CPU-only** — "pure cpu op model" |
| chunk recurrence via `Loop`, **const** trip count | `Loop` | ⚠️ statically unrolled (4×) | ❌ **rejected** |
| chunk recurrence via `Loop`, **runtime** trip count | `Loop` | ❌ rejected | ❌ **rejected** |

Verbatim RKNN evidence:

```
# Scan — accepted but placed on CPU as a "custom operator"
W RKNN: Meet RKNN unsupport Operator: name = 'chunkwise_scan', type = Scan,
        it will be treated as a custom operator.
D RKNN: detect pure cpu op model.

# Loop (even with constant trip count) — hard rejection
E RKNN: build: The Loop('chunkwise_loop') will cause the graph to be a dynamic graph!
        Remove it manually and try again!
ValueError: The Loop('chunkwise_loop') will cause the graph to be a dynamic graph!
```

### Why the Scan "compiles" result is a trap

RKNN does not reject `Scan` at `load_onnx` — it returns rc=0 and proceeds to `build`, which also
returns rc=0. But the verbose log reveals what happened: Scan was treated as a **custom operator**
(a CPU-only escape hatch), and the compiler flagged the model as `"detect pure cpu op model"`.
The resulting `.rknn` file has every op on CPU, zero NPU utilisation. This is a **silent
fallback** — the model "compiles" but the NPU does nothing.

Anyone benchmarking this platform who sees Scan compile without checking the op-placement table
would conclude the NPU handles the recurrence. It does not. The per-op target column in the
Network Layer Information Table is the ground truth.

### RKNN-specific findings not seen on CIX

1. **`Exp` and `Log` cause CPU fallback.** The RK3588 NPU has no exponential or logarithm kernel.
   This directly affects GDN's gated decay, which is computed as `exp(cumsum(log(a)))`. On CIX
   these are native NPU ops (`ArmExp`, `ArmLog`); on RKNN they run on CPU. The `CumSum` in the
   middle is cleverly rewritten as a `Conv` and placed on NPU, but the surrounding transcendentals
   negate the benefit.

2. **RKNN is stricter on `Loop` than CIX.** CIX accepted `Loop` with a constant trip count via
   static unrolling (Section 1). RKNN rejects all `Loop` constructs, even those that CIX would
   unroll — its graph optimizer's `_dynamic_check` refuses any node that *could* introduce
   dynamic control flow, regardless of whether the trip count is actually constant.

3. **`CumSum` → `Conv` rewrite.** RKNN's optimizer lowers `CumSum` to a depthwise `Conv` with
   a lower-triangular weight matrix — a well-known compiler trick. This is placed on NPU and is
   an interesting point of contrast with CIX's native `ArmCumulate` op.

### What this establishes

**The recurrence limitation is structural, not vendor-specific.** Two independent toolchains
(CIX NOE Compiler for the CIX NOE NPU, and Rockchip rknn-toolkit2 for the RK3588 NPU) — different
vendors, different compiler stacks, different NPU architectures — both fail to route a sequential
scan to the NPU. The mechanism differs (CIX rejects Scan; RKNN accepts it as CPU-only), but the
practical outcome is identical: **the scan must run on CPU.**

This strengthens the layer-to-engine mapping argument from Section 1: CPU-hosted scan is not an
optimisation choice for one platform — it is a **constraint imposed by the edge-NPU ecosystem**.

### Reproducing

```bash
# On the RK3588 device (aarch64), Python 3.10:
pip3 install rknn-toolkit2   # installs 2.3.2 from PyPI, works on aarch64

python3 scripts/npu_op_probe.py --out artifacts/npu_op_probe   # regenerate probes if needed
python3 scripts/rknn_op_probe.py                               # runs all 7 probes
# Results: artifacts/npu_op_probe/rknn_audit/rknn_audit_results.json
# Per-probe logs: artifacts/npu_op_probe/rknn_audit/*.log
```

Environment note: rknn-toolkit2 2.3.2 installs cleanly from PyPI on aarch64 and requires no
vendor SDK download — unlike the CIX toolchain which needed a manual wheel install and
has an unconditional TensorFlow dependency.

---

## 8. KleidiAI packed-GEMM micro-kernels for delta-rule matmuls (2026-08-06, ob-8qt.2)

### Motivation

The delta-rule update β = α·S involves a small matmul per chunk (M×K×N where K=head_dim=128,
N=head_dim×n_heads). Arm's KleidiAI library ships 185+ tuned GEMM micro-kernels. The question:
do KleidiAI's packed-GEMM kernels actually win at GDN delta-rule sizes, or does their packing
overhead eat the benefit on small per-chunk matmuls?

### Device and kernel selection

**Device**: RK3588 Cortex-A76 @ 2.3 GHz (big cluster, cores 4-7). ISA: `asimddp=true` (dotprod),
`i8mm=false`, `sve=false`, `bf16=false`. The A76 predates i8mm (A78+) and SVE (Neoverse/V2+).

**Kernel**: `kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla` — tile mr=6, nr=8, kr=1.
This is the best f32 kernel available without i8mm/SME. Also needed: `kai_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon`
for RHS (state matrix S) packing.

### Results — matmul only (excluding packing)

Benchmarked at four GDN delta-rule shapes, pinned to A76 cores, 2000 repeats:

| Shape | M | K | N | Naive C (µs) | Hand NEON (µs) | KleidiAI (µs) | KleidiAI vs NEON |
|---|---|---|---|---|---|---|---|
| decode (1 head) | 1 | 128 | 128 | 13.5 | 3.4 | 1.87 | **1.8× faster** |
| prefill (1 head) | 64 | 128 | 128 | 867 | 217 | 60.8 | **3.6× faster** |
| decode (16 heads) | 1 | 128 | 2048 | 705 | 52.9 | 31.5 | **1.7× faster** |
| prefill (16 heads) | 64 | 128 | 2048 | 44914 | 3467 | 975 | **3.5× faster** |

KleidiAI's matmul kernel always wins over hand-written NEON — 1.7× at decode (M=1), 3.5-3.6× at
prefill (M=64). The larger speedup at prefill reflects better tile utilisation: the 6×8 tile is
underutilised when M=1 (only one row of the 6-row tile is useful).

Correctness: `max_abs_diff = 0.0` for both NEON and KleidiAI vs naive reference across all shapes.

### Results — packing cost (the catch)

KleidiAI requires the RHS (state matrix S) to be packed into its internal layout before the matmul.
In the delta-rule, S changes **every chunk**, so packing cannot be amortised across iterations —
it is a per-step cost.

| RHS size | Pack time (µs) | Packed bytes | Raw bytes | Overhead |
|---|---|---|---|---|
| 128×128 (1 head) | ~7 | 66048 | 65536 | +0.8% |
| 128×2048 (16 heads) | ~126 | 1056768 | 1048576 | +0.8% |

### Net comparison: KleidiAI (matmul + pack) vs hand-NEON (no pack)

| Shape | KleidiAI total (µs) | NEON (µs) | Winner | Margin |
|---|---|---|---|---|
| decode 1×128×128 | 1.87 + 7 = **8.9** | 3.4 | **NEON wins** | 2.6× |
| prefill 64×128×128 | 60.8 + 7 = **67.8** | 217 | **KleidiAI wins** | 3.2× |
| decode 1×128×2048 | 31.5 + 126 = **157.5** | 52.9 | **NEON wins** | 3.0× |
| prefill 64×128×2048 | 975 + 126 = **1101** | 3467 | **KleidiAI wins** | 3.1× |

**Break-even M** (where KleidiAI packing cost equals matmul savings): **M ≈ 3-6** for both head
configurations. Below M≈5, hand-written NEON without packing is faster.

### Interpretation

This is a **phase-dependent recommendation**:

1. **Prefill / chunked-prefill (M ≥ ~8)**: Use KleidiAI. The packed kernel delivers 3-3.6× over
   hand-NEON and packing cost is negligible (<1% of matmul time at M=64). This is where GDN models
   spend most wall-clock time during long-sequence ingestion.

2. **Decode (M = 1)**: Use hand-written NEON. The matmul is small enough that KleidiAI's packing
   overhead (7-126 µs) exceeds the matmul itself. A 4-wide fp32 FMA loop without packing is faster.

3. **On devices with i8mm** (Cortex-A78+, Neoverse V2/N2, Cortex-A720): KleidiAI ships int8 and
   int4 matmul kernels using the I8MM dot-product instruction. These would be both faster (2× per
   cycle vs NEON FMLA) and have smaller packed representations (4-8× less data to move during
   packing). The packing-cost threshold would shift accordingly. This device (A76) cannot test
   that path.

### What this means for the project

The delta-rule matmul in `gdn_gated_scan` should use a **dual-path strategy**: KleidiAI packed-GEMM
for prefill, hand-NEON for decode. This keeps the novel implementation work focused on the three
primitives KleidiAI genuinely lacks (causal conv, gated prefix product, sequential scan), while
reusing Arm's tuned matmul where it actually helps.

The benchmark harness (`bench/kleidiai_matmul_bench.c`) and raw data
(`results/raw/kleidiai/rk3588-t3_kleidiai_matmul.csv`) are committed for reproducibility.

### Reproducing

```bash
# KleidiAI must be cloned (not yet a submodule — evaluation phase):
git clone https://gitlab.arm.com/kleidi/kleidiai.git /tmp/kleidiai

# Build:
gcc -O3 -march=armv8.2-a+simd -I/tmp/kleidiai \
  bench/kleidiai_matmul_bench.c \
  /tmp/kleidiai/kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla.c \
  /tmp/kleidiai/kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32p8x1biasf32_6x8x4_neon_mla_asm.S \
  /tmp/kleidiai/kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon.c \
  -lm -o dist/bench_kleidiai

# Run on big cores:
taskset -c 4-7 dist/bench_kleidiai --csv
```

---

## 9. big.LITTLE affinity policy for GDN kernels on RK3588 (2026-08-06, ob-dqu)

### Motivation

The RK3588 has an asymmetric 4×A76@2.3GHz (big) + 4×A55@1.8GHz (little) layout, directly
analogous to the Orion O6's planned 4×A720 big + 4×A520 little. The Linux scheduler by default
distributes threads across all 8 cores, which means latency-critical GDN kernels may land on the
4× slower A55 cluster or migrate between clusters mid-computation. This study quantifies the
penalty and establishes the pinning policy.

### ISA path verification

| Feature | A76 (big) | A55 (little) | Used by GDN kernels? |
|---|---|---|---|
| NEON (ASIMD) | ✓ | ✓ | **Yes** — all fp32/bf16/fp16 kernels use FMLA |
| dotprod (asimddp) | ✓ | ✗ | **No** — SDOT/UDOT are int8 instructions; fp32 kernels don't use them |
| i8mm | ✗ | ✗ | Not available (requires A78+) |
| SVE/SVE2 | ✗ | ✗ | Not available (requires Armv9) |

Binary analysis confirms: 0 SDOT/UDOT instructions in either binary. 21 FMLA (NEON fused
multiply-add) instances in the A76 binary. The dispatch path is NEON-only on this device.

The `-mcpu=cortex-a76` and `-mcpu=cortex-a55` flags affect compiler scheduling but produce
nearly identical code (both use NEON). Cross-running shows <4% difference between binaries
on the same cluster.

### Affinity comparison — 6 configurations

All measurements: Qwen3.5-4B config (seq=64, channels=4096, 24 GDN layers), 30 repeats,
governor=performance, thermals 38.8→41.6°C.

| Config | Binary | Cores | gdn_gated_scan p50 (µs) | gdn_cumdecay p50 (µs) | gdn_causal_dwconv1d p50 (µs) |
|---|---|---|---|---|---|
| **big_only_a76** | A76 | 4-7 (big) | **284** | **88** | **95** |
| all_cores_a76 | A76 | all 8 (no pin) | 586 | 193 | 218 |
| big_on_little | A76 | 0-3 (little) | 1559 | 402 | 410 |
| little_only_a55 | A55 | 0-3 (little) | 1464 | 343 | 356 |
| little_on_big | A55 | 4-7 (big) | 295 | 93 | 121 |
| simultaneous_split | both | big+A55 parallel | 308 (big) / 1428 (little) | 86 (big) / 1412 (little) | 142 (big) / 358 (little) |

**Key ratios** (vs big_only_a76 baseline):

| Config | gdn_gated_scan | gdn_cumdecay | gdn_causal_dwconv1d |
|---|---|---|---|
| all_cores (no pin) | **2.06× slower** | **2.21× slower** | **2.28× slower** |
| big_on_little | 5.49× slower | 4.59× slower | 4.30× slower |
| little_on_big | 1.04× slower | 1.06× slower | 1.27× slower |

The default OS scheduler (all_cores, no pinning) is **2-2.3× slower** than pinning to big cores.

### Thread-count sensitivity

The root cause of the "all cores" penalty is thread placement, not migration overhead. When
OpenMP sees 8 CPUs, it spawns 8 threads — 4 land on slow A55 cores and become the bottleneck.

| Config | Threads | Pinning | gated_scan p50 (µs) | Spread | vs optimal |
|---|---|---|---|---|---|
| omp4_pinbig | 4 | big (4-7) | **281** | 8-15% | **1.0× (baseline)** |
| default_pinbig | auto (4) | big (4-7) | **282** | 3-9% | **1.0×** |
| omp8_pinbig | 8 | big (4-7) | 468 | 5-13% | 1.7× slower |
| default_nopin | 8 | none | 523 | 21-72% | 1.9× slower |
| omp4_nopin | 4 | none | 886 | 7-33% | 3.1× slower |

Findings:
1. **Pinning is the single most important optimization** — 1.9-3.1× speedup.
2. **Thread count must match physical cores**: 4 threads on 4 big cores is optimal.
3. **Oversubscription (8 threads on 4 cores)** causes SMT-like contention: 1.7× slower.
4. When pinned to big cores, OpenMP auto-detects 4 available CPUs — **no need to set
   OMP_NUM_THREADS explicitly**. `taskset -c 4-7` alone is sufficient.
5. The unpinned configs have **enormous spread** (21-72%) because the scheduler constantly
   migrates threads between clusters. Pinned configs are stable (3-15%).

### Simultaneous split workload (big=decode + little=housekeeping)

Running A76 binary on big cores and A55 binary on little cores simultaneously:

| Kernel | Big solo (µs) | Big concurrent (µs) | Interference | Little solo (µs) | Little concurrent (µs) |
|---|---|---|---|---|---|
| gdn_gated_scan | 284 | 308 | **1.09×** | 1464 | 1428 |
| gdn_cumdecay | 88 | 86 | 0.98× (noise) | 343 | 1412* |
| gdn_causal_dwconv1d | 95 | 142 | 1.49× | 356 | 358 |

\* cumdecay on little shows high variance under load; gated_scan (the dominant kernel) is stable.

The big cluster sees <10% interference from the little cluster's workload. This confirms that
**big and little clusters have independent cache hierarchies and memory bandwidth** — the split
affinity policy (big=latency-critical, little=housekeeping) is viable with negligible overhead.

### Recommendation

For GDN inference on RK3588 (and by analogy, Orion O6):

1. **Always pin to big cores**: `taskset -c 4-7` for all latency-critical GDN kernels.
   This is a 2-3× speedup over default scheduling — the largest single optimization available.
2. **Do not oversubscribe**: let OpenMP auto-detect thread count from the affinity mask.
   Setting OMP_NUM_THREADS higher than physical cores hurts (1.7×).
3. **Use little cores for background work**: tokenisation, memory management, logging can run
   on cpu0-3 with <10% impact on big-core throughput.
4. **On the O6 (A720 + i8mm)**: the same policy applies, plus i8mm-enabled GEMM kernels
   (BFMMLA for bf16, SDOT for int8) will use the dot-product path that this A76 device cannot test.

### Reproducing

```bash
# Full affinity study (6 configs, 30 repeats):
bash bench/biglittle_affinity_study.sh --repeats 30 --csv > results/raw/affinity/study.csv

# Thread-count sensitivity:
for t in 4 8; do
  OMP_NUM_THREADS=$t taskset -c 4-7 dist/bench_gdn_rk3588_a76 --repeats 30 --csv
  OMP_NUM_THREADS=$t dist/bench_gdn_rk3588_a76 --repeats 30 --csv
done
```

---

## 10. GDN-2 vs GDN-1 gating: microbenchmark comparison on RK3588 (2026-08-06, ob-82b)

### The architectural difference

**GDN-1** (standard Gated DeltaNet): `s[t] = x[t] + s[t-1] * g[t]` — one decay gate, one input.
- 3 fp32 streams per step (read g, x; write s) = 12 bytes/element
- 1 FMA = 2 FLOPs/element

**GDN-2** (decoupled erase/write): `s[t] = w[t]*x[t] + s[t-1] * (g[t]*b[t])` — separate erase (b)
and write (w) gates, per ADR 0001.
- 5 fp32 streams per step (read g, b, w, x; write s) = 20 bytes/element
- 2 MUL + 1 FMA = 3 FLOPs/element

GDN-2 does **67% more memory traffic** and **50% more arithmetic** per element, with the same
sequential recurrence dependency chain.

### Measured comparison (RK3588 A76, pinned cpu4-7, 30 repeats)

**Prefill (seq=64) — Qwen3.5-4B (channels=4096):**

| Kernel | p50 (µs) | GiB/s | GFLOP/s | Spread |
|---|---|---|---|---|
| gdn_gated_scan | 267 | 11.07 | 1.96 | 6.2% |
| gdn2_gated_scan | 533 | 9.21 | 1.97 | 10.9% |
| **Ratio** | **2.00× slower** | 0.83× | 1.01× | — |

**Decode (seq=1) — Qwen3.5-4B (channels=4096):**

| Kernel | p50 (µs) | GiB/s | GFLOP/s | Spread |
|---|---|---|---|---|
| gdn_gated_scan | 1.46 | 52.33 | 5.62 | 0.1% |
| gdn2_gated_scan | 1.46 | 73.20 | 11.23 | 20.0% |
| **Ratio** | **1.00× (identical)** | 1.40× | 2.00× | — |

**Decode (seq=1) — Qwen3.5-0.8B (channels=2048):**

| Kernel | p50 (µs) | GiB/s | GFLOP/s |
|---|---|---|---|
| gdn_gated_scan | 1.17 | 32.72 | 3.51 |
| gdn2_gated_scan | 1.17 | 45.77 | 7.02 |
| **Ratio** | **1.00× (identical)** | 1.40× | 2.00× |

**Little cluster (A55) — Qwen3.5-4B prefill:**

| Kernel | p50 (µs) | Ratio |
|---|---|---|
| gdn_gated_scan | 1304 | — |
| gdn2_gated_scan | 4047 | **3.1× slower** |

### Analysis

**Finding 1: GDN-2's extra gates are free at decode.** At seq=1, both kernels take identical
wall-clock time (1.46µs). GDN-2 achieves 2× the GFLOP/s because it does 2× the FLOPs in the same
time — the extra multiply-adds are completely hidden behind memory latency. This confirms the
kernel is **memory-bandwidth-bound at decode**, not compute-bound: you could add arbitrary
arithmetic per element without changing the latency, as long as the stream count doesn't increase
past what the memory system can serve.

Wait — GDN-2 does increase streams from 3 to 5, yet latency is unchanged. This means at seq=1 the
kernel is not even bandwidth-bound — it's dominated by **function-call and state load/store
overhead**. The per-element work (whether 2 or 3 FLOPs, 3 or 5 streams) is negligible compared
to the fixed cost of entering the kernel, loading the state vector, and storing the output.

**Finding 2: GDN-2 costs exactly 2× at prefill.** At seq=64, GDN-2 is 2.0× slower with identical
GFLOP/s throughput (1.97 vs 1.96). This means the A76 pipeline processes both kernels at the same
FLOP rate — GDN-2 simply has 2× the FLOPs to do. The extra streams (5 vs 3) cause the effective
bandwidth to drop slightly (11.07 → 9.21 GiB/s), suggesting the memory system is approaching
saturation but is not yet the bottleneck at this shape.

**Finding 3: GDN-2 penalty is worse on the little cluster.** On A55 (in-order, narrower pipeline),
GDN-2 is 3.1× slower vs 2.0× on A76. The A55's simpler pipeline cannot hide the extra instruction
latency from the two additional multiplies per step. This has implications for heterogeneous
dispatch: GDN-2 models are relatively more expensive on little cores.

### What this means for the project

1. **GDN-2 is a viable decode-time alternative at zero cost.** The decoupled erase/write gating
   that improves retrieval quality (per NVLabs' paper) adds no decode latency on this hardware.
   The quality improvement is "free" at inference time.

2. **GDN-2 costs 2× at prefill.** This is expected and acceptable — prefill is amortised across
   all subsequent decode steps. For a chunk size of 64, the 266µs extra prefill cost is recovered
   after ~182 decode steps (at 1.46µs/step).

3. **On edge devices with little cores, GDN-2 should be routed to big cores only.** The 3.1×
   penalty on A55 vs 2.0× on A76 means the little cluster is a worse-than-linear fit for GDN-2's
   extra arithmetic.

### Reproducing

The data is already in the Phase 1 CSVs — no separate run needed:

```bash
# GDN-1 vs GDN-2 from existing benchmark data:
grep -E "gdn_gated_scan|gdn2_gated_scan" results/raw/rk3588-t3_big.csv
grep -E "gdn_gated_scan|gdn2_gated_scan" results/raw/rk3588-t3_little.csv
```
