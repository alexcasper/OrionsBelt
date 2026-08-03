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

---

## Cross-device kernel bandwidth comparison (2026-08-03)

Benchmark data from the device fleet (bead ob-8ms.3, ob-c9q):

| Device | Core | Clock | Scan kernel (4B) | Cumdecay (4B) | Ratio to Jetson |
|---|---|---|---|---|---|
| Jetson Nano | A57 | 1.48 GHz | 0.72 GiB/s | 1.16 GiB/s | 1.0× |
| Raspberry Pi 5 | A76 | 2.40 GHz | 1.84 GiB/s | 2.93–3.74 GiB/s | ~2.5–2.6× |

**Finding:** The Pi 5 (A76) is ~2.5× faster than the Jetson (A57) on the
gated-scan kernel, consistent with the clock ratio (1.63×) and the wider
A76 execution pipeline. No inversion — the A76's advantage scales as expected
for this memory-bandwidth-bound kernel.

**Decode (seq=1) on Jetson A57:** the cumdecay kernel hits 4.65 GiB/s at
seq=1 vs 1.16 GiB/s at seq=64 — a 4× speedup because the single-token
recurrence fits entirely in L1 cache. This is the architectural property
that makes GDN viable on edge silicon: O(1) recurrent state means the
decode path becomes cache-resident regardless of context length.
