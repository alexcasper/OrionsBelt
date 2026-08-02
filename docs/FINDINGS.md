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

**Clang/LLVM ≥17 or GCC ≥13, with `-mcpu=cortex-a720`.** That single flag is sufficient — verified
by dumping clang 18's predefined feature macros for that target:

```
__ARM_FEATURE_SVE2 1              __ARM_FEATURE_MATMUL_INT8 1
__ARM_FEATURE_SVE2_BITPERM 1      __ARM_FEATURE_SVE_MATMUL_INT8 1
__ARM_FEATURE_SVE 1               __ARM_FEATURE_SVE_MATMUL_FP32 1
__ARM_FEATURE_BF16 1              __ARM_FEATURE_SVE_BF16 1
__ARM_FEATURE_DOTPROD 1           __ARM_FEATURE_FP16_FML 1
```

Equivalent explicit form: `-march=armv9.2-a+sve2+i8mm+bf16`. Arm Compiler for Linux (ACfL) is
Arm's own LLVM-based toolchain and is a defensible choice for an Arm submission.

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
