# Claim verification against primary sources

Bead `ob-ofk` (`t-verify-claims`). `brief.md` mixes primary sources with weak secondary ones
(X posts, blog aggregators, a YouTube video). Judges will check our numbers, so every figure
we intend to quote is traced here to a primary source, corrected, or dropped.

Verified 2026-08-02.

---

## 1. Corrections — brief.md is wrong

### 1.1 The third track is **Edge AI**, not "Mobile AI" ⚠️

`brief.md` repeatedly refers to a "Mobile AI" track. The [official rules](https://arm-ai-optimization-challenge.devpost.com/rules)
list the three category prizes as:

| Prize | Amount |
|---|---|
| Best in Track — **Physical AI** | $1,000 |
| Best in Track — **Cloud AI** | $1,000 |
| Best in Track — **Edge AI** | $1,000 |

There is no Mobile AI track. This matters directly: it changes which track the hedge plan
targets, and "Edge AI" is a *better* fit for an Orion O6 inference box than "Mobile AI" ever
was — the O6 is an edge device, not a phone.

### 1.2 The "60 layers" figure is wrong for the models we care about

`brief.md` claims "3:1 linear:full attention across 60 layers", sourced from an X post. Primary
sources disagree:

| Checkpoint | Layers | Source |
|---|---|---|
| Qwen3.5 dense (default config) | `num_hidden_layers=32` | transformers `Qwen3_5TextConfig` |
| Qwen3.5-4B | 32, as `8 × (3× DeltaNet → FFN → 1× Attention → FFN)` | secondary, consistent with above |
| Qwen3.5-35B-A3B (MoE) | `num_hidden_layers=40` | transformers `Qwen3_5MoeTextConfig` |

For a 32-layer model at 3:1 that is **24 Gated DeltaNet layers and 8 full-attention layers** —
the concrete numbers our profiling and mapping work needs. Do not quote 60.

### 1.3 The 397B-A17B upper bound is unconfirmed

`brief.md` states the family runs "from 0.8B to 397B-A17B". The 0.8B floor is confirmed. The
397B-A17B ceiling appears only in weak sources; the transformers MoE docs document
`Qwen3.5-35B-A3B`, and at least one secondary source describes the family as topping out
around 122B. **Drop this figure** — it is irrelevant to us anyway, since we target the small
dense end.

### 1.4 Minor overstatements

| brief.md | Primary source |
|---|---|
| "over 100GB/s" memory bandwidth | "100GB/s" (128-bit LPDDR5 @ 5500MT/s) |
| License must be "visible in the GitHub About section" | Rules require a public repo under MIT or Apache-2.0; the About-section specificity is not in the rules text |

The About-section detail is worth doing regardless — it costs nothing and makes compliance
obvious to a judge — but it is our practice, not a stated requirement.

### 1.5 GDN-2's RULER gains — **now quantified** (superseded, see below)

*Originally recorded here as abstract-only and qualitative. Superseded 2026-08-02: the full
paper HTML is reachable at `arxiv.org/html/2605.22791`, and Table 3 contains numeric scores.
Extracted by parsing the raw HTML table directly rather than via a summarizer — an intermediate
summarized read dropped a column and misaligned the rest, so these were confirmed cell-by-cell.*

MK-NIAH-1 (multi-key needle-in-a-haystack), 1.3B models, higher is better:

| Setting | Model | 1K | 2K | 4K |
|---|---|---:|---:|---:|
| Recurrent-only | Gated DeltaNet | 58.0 | 37.0 | 27.8 |
| Recurrent-only | **Gated DeltaNet-2** | **72.6** | **51.4** | **37.8** |
| Hybrid | Gated DeltaNet | 91.0 | 78.4 | 44.8 |
| Hybrid | **Gated DeltaNet-2** | **93.0** | **84.6** | **48.0** |

The gain is large in the recurrent-only setting (+14.6 / +14.4 / +10.0) and much smaller once
periodic full attention is present (+2.0 / +6.2 / +3.2). **That second row matters more to us
than the first**, because Qwen3.5 *is* a hybrid — so the honest expected upside of a GDN-2 swap
in our setting is the smaller hybrid delta, not the headline recurrent-only one.

A further useful detail from the paper's channel-structure ablation (MK-NIAH-1 @4K): channel-wise
erase with a scalar write scores 35.2, scalar erase with channel-wise write scores 30.6, and both
channel-wise scores 37.8. **The erase gate carries most of the benefit** — so if a layer swap is
ever attempted under time pressure, channel-wise erase is the higher-value half to implement first.

Caveat on cost: the paper measures only **training** throughput on H100 (38.0 → 36.1 Kt/s, ~5%,
attributed to the added gates) and **never measures single-token decode or inference latency on
any hardware**. It provides no evidence either way about our bandwidth-bound decode regime.

---

## 2. Confirmed claims

### 2.1 Competition (source: [Devpost rules](https://arm-ai-optimization-challenge.devpost.com/rules))

| Claim | Status |
|---|---|
| Deadline 2026-08-14, 16:00 PT | ✅ Confirmed — 4:00pm **PDT** |
| $8,000 total: $3,000 overall / $2,000 runner-up / 3 × $1,000 track | ✅ Confirmed |
| Judging 40 technological / 25 WOW / 20 impact / 15 developer experience | ✅ Confirmed, sums to 100 |
| MIT or Apache-2.0, public open-source repo | ✅ Confirmed |
| Write-up: overview, functionality/output, step-by-step setup and validation | ✅ Confirmed |
| Video optional, under 3 minutes, shows project on intended device | ✅ Confirmed |

### 2.2 Orion O6 hardware (source: [radxa.com](https://radxa.com/products/orion/o6/), [Radxa docs](https://docs.radxa.com/en/orion/o6/app-development/artificial-intelligence))

| Claim | Status |
|---|---|
| Up to 45 TOPS combined (NPU + CPU + GPU) | ✅ Confirmed |
| NPU 28.8 TOPS | ✅ Confirmed |
| INT4 / INT8 / INT16 / FP16 / BF16 / TF32 | ✅ Confirmed |
| 12-core CPU: 4× A720 big @2.8GHz, 4× A720 medium @2.4GHz, 4× A520 little @1.8GHz, 12MB shared L3 | ✅ Confirmed — note the **three distinct frequency tiers**, which matters for affinity work |
| Immortalis G720 MC10 GPU | ✅ Confirmed |
| Up to 64GB, 128-bit LPDDR5 @ 5500MT/s | ✅ Confirmed |
| ~30 tokens/sec on Qwen2-1.5B | ✅ Confirmed — vendor figure, unstated precision and context length, so treat as a rough sanity target rather than a benchmark |
| Models "within ten billion parameters" | ✅ Confirmed as vendor wording |
| CIX Early Bird Program required for latest P1 software | ✅ Confirmed as vendor wording — but see §2.2a: it is **not** required for the NOE Compiler / NPU SDK, which downloads directly |

### 2.2a NPU SDK — **brief.md was wrong, and the news is good** (verified 2026-08-02)

Re-checked against [Radxa's NPU environment-setup page](https://docs.radxa.com/en/orion/o6/app-development/artificial-intelligence/env-setup)
and the [public CIX Linux wiki](https://github.com/cixtech/cix-manifest/wiki):

| brief.md claimed | Primary source says |
|---|---|
| NOE Compiler pins **Python 3.8** | **"The SDK is only compatible with Python 3.10."** Set up via a miniforge 3.10 venv. (Community reports the Python NOE wheel wants 3.11-3.12 and `cix-noe-umd` errors on 3.13+, so treat 3.10 as the documented target and expect version sensitivity.) |
| SDK access requires **CIX Early Bird enrollment + approval** | **No registration or approval mentioned.** Direct download from the Radxa Download Station via `wget`. |
| — | **The SDK runs on an x86 Linux host**, not the board. The board's official OS image already ships the NPU driver. |

Three consequences, and they are significant:

1. **The Python 3.8 constraint does not exist.** The correct floor is **3.10**. Any code written
   to a 3.8 dialect for the NPU toolchain's sake was self-imposed on a false premise.
2. **Risk R2 largely dissolves.** The NOE Compiler is not gated behind an approval queue. There
   may still be an Early Bird program for other CIX material, but it is not on the path to
   compiling models.
3. **NPU compiler work no longer requires the board.** Because `cixbuild` runs on an x86 host, we
   can install the SDK and attempt to compile GDN operators *today*, with no hardware. That moves
   the single most valuable early finding — the NOE op-coverage audit for the gated recurrent scan
   (risk R3, and the core technical contribution) — from "blocked behind two external gates" to
   "actionable now". Only *running* the compiled model needs the board.

The separate `cix-manifest` wiki covers the **Linux BSP/kernel build** (CIX EVB, Radxa Orion O6,
MG MS-R1, BeiQi AI PC) and is fully public — so BSP-level work is ungated too.

### 2.3 Qwen3.5 architecture (source: [transformers docs](https://huggingface.co/docs/transformers/model_doc/qwen3_5))

| Claim | Status |
|---|---|
| 3:1 hybrid: three Gated DeltaNet layers per one Gated Attention layer | ✅ **Confirmed by primary source** — the load-bearing claim holds |
| Small dense variants exist at 0.8B, 2B, 4B, 9B | ✅ Confirmed — docs/archive/PLAN.md's 0.8B–4B target range is viable |
| 262K native context | ✅ Confirmed: "Native context is 262,144 tokens." ~1M requires **YaRN rope scaling** via `rope_scaling`; plain loading gives the native window only |

Concretely useful details found:

- `layer_types` on the text config is a per-layer list of `"linear_attention"` / `"full_attention"`.
  **This is exactly the introspection hook `t-arch-audit` needs** — the hybrid layout is readable
  from config, no modeling-code archaeology required.
- Linear-attention layer shapes: `linear_conv_kernel_dim=4`, `linear_key_head_dim=128`,
  `linear_value_head_dim=128`, `linear_num_key_heads=16`, `linear_num_value_heads=32`.
  **Correction (2026-08-02, from `docs/MODEL_SURVEY.md`):** these are the *default-config* values
  and are **not constant across the family** — `linear_num_value_heads` is 16 at 0.8B/2B and 32 at
  4B/9B. Read the shapes from the chosen checkpoint's own `config.json`, never from the doc default.
- The DeltaNet path is `Qwen3NextGatedDeltaNet` — Qwen3.5's text backbone reuses Qwen3-Next's
  linear-attention decoder, so Qwen3-Next tooling is likely to transfer.
- Dense and MoE checkpoints share the same GDN core (`Qwen3_5MoeGatedDeltaNet` ≡
  `Qwen3_5GatedDeltaNet`) but have very different shapes; weights are not interchangeable.

### 2.4 Q8_0 block-quantized GEMV results (verified 2026-08-08)

Source: our own measurements on Jetson Nano A57 (Cortex-A57, 4×1.48 GHz, governor=performance).

| Claim | Status |
|---|---|
| Q8_0 decode: 5.12 tok/s on 0.8B model (A57) | ✅ Confirmed — FINDINGS §29, CSV `jetson-j1_q80_vs_int8_vs_fp32_08b.csv`, manifest `jetson-j1_q80_a57.json` (sha `d223c19`) |
| 2.97× speedup over FP32 decode (1.72 tok/s → 5.12 tok/s) | ✅ Confirmed — same CSV, same commit, A/B comparison |
| 95% of llama.cpp Q8_0 throughput on the same hardware | ✅ Confirmed — llama.cpp Q8_0 baseline: 5.40 tok/s; 5.12/5.40 = 94.8% (FINDINGS §28, §29; CSV `jetson-j1_llamacpp_vs_orionsbelt_08b.csv`) |
| Q8_0 cosine similarity 1.000000 vs FP32 across all 11 weight matrices | ✅ Confirmed — FINDINGS §30, CSV `jetson-j1_quant_accuracy_08b_4b.csv` (66 rows), manifest sha `c643e34`. Method: per-matmul GEMV with fixed-seed random weights, `--verify-quant` mode |
| INT8 also achieves cos_sim 1.000000 but no speed gain on A57 | ✅ Confirmed — same accuracy CSV; INT8 decode throughput 1.72 tok/s (§29 CSV) |
| INT4 degrades to cos_sim ≈ 0.99998, no speed advantage | ✅ Confirmed — same accuracy CSV; INT4 throughput 1.64 tok/s (§29 CSV) |
| Cache-blocked GEMM: 49–78× prefill speedup | ✅ Confirmed — FINDINGS §25 (ob-8qt.15) |
| llama.cpp Q8_0/Q4_0 is 3.1× faster decode, 2.3× faster prefill than our C loop | ✅ Confirmed — FINDINGS §28 (ob-mrd.15), measured on same A57 device |
| Q8_0 GDN layer cost flat at 73–80 ms (aggregated) across ctx 1–4096 | ✅ Confirmed — FINDINGS §31, CSV `jetson-j1_08b_q80_ctxsweep_e2e_raw.csv`, manifest sha `d63e64a` |
| Q8_0 retains 1.85–2.46× advantage over FP32 across all context lengths | ✅ Confirmed — FINDINGS §31, same CSV. Ratio peaks at ctx=1024, narrows at ctx=4096 where attention dominates |
| Q8_0 pure-GDN throughput ±3% across ctx 1–4096 (O(1) confirmation) | ✅ Confirmed — FINDINGS §31, CSV `jetson-j1_08b_q80_puregdn_ctxsweep_e2e_raw.csv`, manifest sha `72bafcc` |
| INT4 is 16% slower than FP32 at short context on A57 (dequant overhead) | ✅ Confirmed — FINDINGS §32, CSV `jetson-j1_08b_int4_ctxsweep_e2e_raw.csv`, manifest sha `00fe9f1` |

All Q8_0 measurements use `gcc -O3 -fopenmp -mcpu=cortex-a57` (matching
`scripts/build_device_bench.sh`). Thermals ≤53°C throughout; governor
`performance` confirmed in every manifest.

### 2.5 SDOT INT8 and INT4+SDOT hybrid GEMV results (verified 2026-08-09)

Source: our own measurements on RK3588 (Cortex-A76 big cluster, 4×2.4 GHz, governor=performance).

| Claim | Status |
|---|---|
| SDOT INT8 GEMV: 3.48 tok/s on 4B model (A76) | ✅ Confirmed — FINDINGS §33, manifest `rk3588-t4_sdot_4b.json` (sha `d6b77b2`, clean re-run) |
| SDOT INT8 GEMV: 30.51 tok/s on 0.8B model (A76) | ✅ Confirmed — FINDINGS §33, manifest `rk3588-t4_sdot_08b.json` (sha `d6b77b2`, clean re-run) |
| SDOT is 1.92× over NEON INT8 (4B A76) | ✅ Confirmed — FINDINGS §33, 1.82→3.48 tok/s A/B comparison |
| SDOT is 3.09× over NEON INT8 (0.8B A76) | ✅ Confirmed — FINDINGS §33, 9.86→30.51 tok/s |
| SDOT reaches 83% of theoretical bandwidth ceiling (4B) | ✅ Confirmed — FINDINGS §33, 3.48/4.20 tok/s |
| SDOT SNR: 46.7–49.7 dB vs FP32 oracle | ✅ Confirmed — FINDINGS §33, `--verify-int4` extended test suite |
| INT4+SDOT hybrid: 4.52 tok/s on 4B model (A76) | ✅ Confirmed — FINDINGS §34, manifest `rk3588-t4_int4sdot_4b.json` (sha `d6b77b2`, clean re-run) |
| INT4+SDOT hybrid: 36.36 tok/s on 0.8B model (A76) | ✅ Confirmed — FINDINGS §34, manifest `rk3588-t4_int4sdot_08b.json` (sha `d6b77b2`, clean re-run) |
| INT4+SDOT is 1.30× over INT8+SDOT (4B A76) | ✅ Confirmed — FINDINGS §34, 3.48→4.52 tok/s |
| INT4+SDOT is 1.19× over INT8+SDOT (0.8B A76) | ✅ Confirmed — FINDINGS §34, 30.51→36.36 tok/s |
| INT4+SDOT slower on A55 (0.96× vs INT8 SDOT) | ✅ Confirmed — FINDINGS §34, 1.30 vs 1.36 tok/s — compute-bound on little cluster |
| Cumulative optimization stack: ~65× over naive FP32 baseline (4B) | ✅ Confirmed — FINDINGS §34, 0.07→4.52 tok/s |
| INT4+SDOT SNR: 46.7–49.7 dB, identical to NEON | ✅ Confirmed — FINDINGS §34, 12 test cases |

All SDOT/INT4+SDOT measurements on A76 use `gcc -O3 -fopenmp -mcpu=cortex-a76`
(dotprod enabled). Cross-device agreement t3↔t4 (§38, both devices SDOT):
**3.0% (4B), 5.1% (0.8B)** for pure-GDN INT8+SDOT; **7.4% (4B), 3.7% (0.8B)**
for INT4+SDOT (t3 clean re-run at `c880887`). E2e (with attention) INT8+SDOT
gap grows with context (8–27% at ctx 512–4096) due to RAM-bandwidth differences
between the two boards (t3: 32 GB, t4: 8 GB), not compute — §38. Thermals
≤62°C; governor `performance` confirmed in every manifest.

### 2.6 Engine boundary-crossing cost (verified 2026-08-09, cross-validated 2026-08-12)

| Claim | Status |
|---|---|
| 16 crossings/token at 3.36 ms total (5KB hidden state, Mali-G610, RustiCL/Panfrost) | ✅ Confirmed — FINDINGS §39, CSV `rk3588-t4_gpu_boundary_crossing.csv`, manifest `rk3588-t4_gpu_boundary_crossing.json` (sha `7ca7f2a`, dirty=false corrected from false-positive, governor=performance, ~41°C) |
| ~10% of 30 tok/s (33.3 ms) decode budget (RustiCL/Panfrost) | ✅ Confirmed — 3.36/33.3 = 10.1% |
| Latency-dominated: ~0.1 ms dispatch floor on RustiCL/Panfrost (1KB–100KB payloads) | ✅ Confirmed — same CSV, write_blocking rows: 0.102/0.102/0.103/0.108/0.112 ms for 1KB/5KB/10KB/50KB/100KB |
| Heterogeneous offload must deliver >11% speedup to break even (RustiCL/Panfrost) | ✅ Confirmed — crossing tax is 10.1%, so net speedup must exceed this |
| **ARM blob driver: 16 crossings at 0.24 ms (14× faster than RustiCL/Panfrost)** | ✅ Confirmed — FINDINGS §39 cross-validation, CSV `rk3588-t3_gpu_boundary_crossing.csv`, manifest `rk3588-t3_gpu_boundary_crossing.json` (sha `25941cf`, governor=performance, ~41°C) |
| **ARM blob crossing tax: 0.7% of decode budget (vs 10.1% on RustiCL)** | ✅ Confirmed — 0.24/33.3 = 0.72% |
| **ARM blob dispatch floor: ~6 µs per call (vs ~100 µs on RustiCL)** | ✅ Confirmed — same CSV, write_blocking rows: 0.005/0.007/0.007/0.013/0.017 ms for 512B/5KB/10KB/50KB/100KB |
| **Break-even speedup on blob: >0.7%** | ✅ Confirmed — crossing tax is 0.7%, so net speedup must exceed this (negligible barrier) |

Initial measurement on RK3588 t4 via RustiCL/Panfrost driver (open-source). Cross-validated
on RK3588 t3 via ARM proprietary blob (libmali-valhall-g610-g13p0-x11) — the driver type a
production deployment and the O6 will use. The 14× gap is dispatch-overhead-dominated (not
bandwidth): both drivers converge at large payloads (1 MB write: 9367 vs 4602 MiB/s, only 2×
apart). ADR 0005 designates both as valid proxies for the O6's Immortalis-G720. The blob's
0.24 ms / 0.7% figure is the more representative estimate for target hardware.

### 2.7 GPU kernel performance — corrected conclusion (verified 2026-08-11)

| Claim | Status |
|---|---|
| All 4 GDN primitives have OpenCL kernels, bit-exact validated (87/87 tests) | ✅ Confirmed — `gpu/gdn_gpu_kernels.cl` (205 lines), `gpu/gdn_gpu_bench.c` (617 lines), validation harness |
| Mali-G610 matches or beats 4-thread A76 CPU on all three channel-wise kernels | ✅ Confirmed — FINDINGS §13, `results/raw/rk3588-t3_gpu_gdn_kernels.json`, commit `fbda76e`, 4 independent runs of 50 repeats, spread ≤8.7% |
| Scan kernel: GPU 57.5 µs vs CPU 114.9 µs (1.99×) | ✅ Confirmed — same JSON; device-side profiling (CL_QUEUE_PROFILING_ENABLE) |
| CumDecay: GPU 31.9 µs vs CPU 33.0 µs (1.03×) | ✅ Confirmed — same JSON |
| DWConv1D: GPU 38.5 µs vs CPU 50.2 µs (1.30×) | ✅ Confirmed — same JSON |
| Initial conclusion was reversed by kernel-code fix | ✅ Confirmed — commits 4cc1cba/d60220c (matrix notation), old data at 048aa7e showed GPU 0.59–0.81× CPU |
| Validated on two independent driver stacks | ✅ Confirmed — ARM Mali libmali blob + open-source Mesa RustiCL (FINDINGS §13, ob-q44.1) |

> **Caveat:** GPU timing is device-side profiling (excludes host↔device transfer).
> CPU comparison is 4-thread A76 (OMP_NUM_THREADS=4, taskset cores 4–7).
> Full methodology in FINDINGS §13.

### 2.8 Cross-vendor NPU compiler rejection of GDN recurrence (verified 2026-08-02, 2026-08-06)

| Claim | Status |
|---|---|
| CIX NOE Compiler rejects runtime-trip-count `Loop` (cannot express sequential recurrence) | ✅ Confirmed — FINDINGS §1, `scripts/npu_op_probe.py`, NOE SDK 26_q2 / cixbuilder 6.1.3753.3 |
| Rockchip RKNN also rejects runtime `Loop` ("dynamic graph" error) | ✅ Confirmed — FINDINGS §2a, `scripts/rknn_op_probe.py`, RKNN toolkit targeting RK3588 |
| Scan compiles on RKNN (CIX rejects even Scan) — genuine toolchain difference | ✅ Confirmed — FINDINGS §2a, 8 KB compiled model verified in RKNN simulator |
| No current edge NPU compiler handles GDN's variable-length recurrence | ✅ Confirmed — two independent vendors, different silicon, same architectural constraint |

This generalizes beyond a single vendor: NPU compilers require static, parallelizable dataflow
graphs, and GDN's per-token sequential recurrence violates that. The project's CPU kernels
run because OoO pipelines handle sequential dependencies; the NPU's strength (massive
parallelism) is the wrong tool for this workload.

### 2.9 ONNX Runtime CPU EP: GDN via Loop but 16× slower (verified 2026-08-08)

| Claim | Status |
|---|---|
| ORT CPU EP executes GDN recurrence correctly (rel_err 2.3×10⁻⁷ vs NumPy) | ✅ Confirmed — FINDINGS §27, `scripts/ort_gdn_probe.py`, device rk3588-t4 |
| ORT generic Loop: ~49 µs/token vs ~3 µs for fused C kernel (16× overhead) | ✅ Confirmed — same audit, single-head V=128 |
| No Arm-specific tuning for projection matmuls in ORT's GDN path | ✅ Confirmed — ORT uses generic CPU EP, no NEON/SVE optimization for GDN ops |
| Third data point confirming no existing CPU toolchain optimizes GDN for Arm | ✅ Confirmed — NPU (rejected), KleidiAI (matmul only), ORT (generic), llama.cpp (dedicated op but untuned scalar) |

### 2.10 Sustained-load thermal stability (verified 2026-08-09)

| Claim | Status |
|---|---|
| INT8 SDOT: 0.3% throughput decay over 3.4 min sustained (3.46→3.45 tok/s) | ✅ Confirmed — FINDINGS §37, rk3588-t4 A76 4-thread, governor=performance |
| FP32: 0.9% throughput decay over 5 min sustained (1.10→1.09 tok/s) | ✅ Confirmed — same device/section |
| Temperature plateaus at ~52°C within 70s, no throttling (idle ~39°C, ΔT=13°C) | ✅ Confirmed — bigcore thermals logged in FINDINGS §37 table |
| Headline numbers are steady-state sustainable, not burst artifacts | ✅ Confirmed — directly addresses PLAN.md risk R7 |

### 2.11 Context-length scaling: GDN O(1) vs full-attention O(n) (verified 2026-08-09)

| Claim | Status |
|---|---|
| Pure-GDN throughput flat to within <0.2% across ctx=1–4096 | ✅ Confirmed — FINDINGS §17, 0.8B: 27.52→27.47 tok/s (0.18%); 4B: 3.25→3.24 tok/s (within measurement noise) |
| Hybrid model degrades 1.33× (4B) to 1.56× (0.8B) at ctx=4096 | ✅ Confirmed — same section, hybrid tables: 4B 3.30→2.49, 0.8B 28.79→18.46 |
| Full-attention latency grows linearly with context (6.2× for 4B, 8.7× for 0.8B) | ✅ Confirmed — FINDINGS §17, full-attn column: 4B 19→116 ms, 0.8B 2.5→22 ms |
| GDN recurrent state is constant (~1.2 MB for 24 layers) vs KV cache growing to 256 MB at ctx=4096 | ✅ Confirmed — same section, memory analysis |
| Cross-validated on Jetson Nano A57 (second core class) | ✅ Confirmed — FINDINGS §17 A57 ctx-sweep, commit 3d83bdc |
| Cross-validated on two independent RK3588 units (§36, §38) | ✅ Confirmed — t3 and t4 agree within 3–5% after SDOT binary fix |

---

## 3. Finding that changes the technical thesis 🔴

The transformers docs report measured kernel numbers that we should absorb *before* setting
expectations. On an NVIDIA GB10, swapping the slow PyTorch GDN fallback for an optimized kernel:

| Checkpoint | TTFT (prefill) | Decode |
|---|---|---|
| Qwen3.6-27B dense | 1.66 s → 1.11 s (**1.49× faster**) | 4.11 → 4.14 tok/s (flat) |
| Qwen3.6-35B-A3B MoE | 0.73 s → 0.53 s (**1.38× faster**) | 16.3 → 16.7 tok/s (flat) |

With the upstream explanation:

> "Decode is roughly flat because the single-token DeltaNet recurrence is memory-bandwidth-bound;
> the win is on the chunked-prefill core and grows with prompt length."

### What this means for us

**Optimizing GDN kernels buys prefill/TTFT, not decode throughput.** The single-token recurrence
is bandwidth-bound, and no amount of kernel cleverness changes that — on the O6, with 100GB/s of
LPDDR5, it will be *more* bandwidth-bound than on a GB10, not less.

So the two honest claims to build the submission around are:

1. **Prefill / TTFT throughput** — where kernel work genuinely pays, and where the win *grows
   with context length*. This is the right target for the GPU scan kernel (`t-gpu-scan`) and the
   NPU offload.
2. **Decode memory footprint** — O(1) recurrent state versus a KV cache growing linearly with
   context. This is an architectural property, not a kernel optimization, and it is what makes
   long context feasible on a 64GB edge board at all.

What we should **not** promise is a large decode tokens/sec win from GDN kernel optimization.
Physics is against it, and claiming it would be the kind of overstatement that collapses under a
judge's scrutiny. Better to predict the flat decode result, measure it, and explain *why* — that
reads as competence rather than a miss.

This also sharpens `t-harness-mem` from "nice measurement" to **the** measurement: the memory
decomposition is the load-bearing evidence, and prefill scaling is the performance headline.

### A second finding: the gap our project fills

> "The DeltaNet path needs the optional `causal_conv1d` and `fla` packages for its fast kernels —
> without them, the model silently falls back to slower and more memory hungry PyTorch ops."

Upstream already documents an architecture-specific hole: on NVIDIA GB10 (SM121) neither package
ships a build, so the fast path silently degrades. **Arm/Vulkan is the same class of hole, wider.**
That is the concrete, verifiable statement of what this project contributes — not "we optimized a
model" but "the fast GDN kernels do not exist for this silicon, and here is what we built and
measured instead." Upstream's own workaround is a Hub kernel (`Atlas-Inference/gdn`), which is a
useful precedent for how a per-architecture GDN kernel gets packaged.

---

## 3.1 CIX's own model hub contains no linear-attention architecture (verified 2026-08-02)

Checked the [CIX AI Model Hub on ModelScope](https://www.modelscope.cn/models/cix/ai_model_hub/files?version=26_Q1)
via the ModelScope file API, at revisions `26_Q1` and `master` (there is no `26_Q2` model-hub
revision yet, even though the SDK we have is the 26 Q2 release).

`models/Generative_AI/LLM/` holds **38 model entries**. Every one is a conventional
full-attention transformer:

- Qwen family: Qwen1.5-1.8B/4B, Qwen2-0.5B/1.5B/7B, Qwen2.5-0.5B/1.5B/3B/7B, **Qwen3-0.6B /
  1.7B / 4B / 8B / 30B-A3B**
- Llama 2 / 3 / 3.1 / 3.2, Phi-2 / Phi-3-mini / Phi-3.5-mini, Gemma-2-2b, ChatGLM3-6B,
  InternLM2.5 / InternLM3, ERNIE-4.5 (incl. 21B-A3B), MiniCPM3-4B, GLM-Edge-4B,
  DeepSeek-R1 distills

**Absent: Qwen3.5, Qwen3-Next, Mamba, RWKV, any Gated DeltaNet or SSM or linear-attention model
of any kind.** The multimodal directory tells the same story — Qwen2-VL and Qwen2.5-VL/Omni, no
Qwen3.5-VL.

Two details make this more than an "it's just new" observation:

1. **Qwen3-30B-A3B is present**, so MoE architectures are already supported by the toolchain —
   sparsity is not the barrier.
2. **Qwen3 is present**, so the hub is not merely lagging by a generation.

What is missing is specifically the *architecture class*: nothing in CIX's reference collection
carries a recurrent state instead of a KV cache. That is precisely the hole this project targets,
and it is now a checkable, citable fact rather than an inference.

**What this does and does not establish.** It does *not* prove the NOE Compiler cannot compile GDN
operators — that is exactly what the probe graphs in `artifacts/npu_op_probe/` are for, and the
answer might well be "most of it compiles, the sequential scan does not." What it does establish
is that **no reference path exists**: nobody has shipped a linear-attention model for this
silicon, so there is no worked example to copy, no precedent for how the recurrent state is
carried across invocations, and no vendor-tuned kernel to lean on. That is the strongest available
support for the framing in §3 — the contribution is not "we optimized a model" but "we did the
port that had no prior art on this platform."

## 4. Previously unverified — now resolved

All items below were originally open; each has since been resolved elsewhere in this
document or in a linked source. Kept for traceability.

| Item | Resolution | Where |
|---|---|---|
| ~~NOE Compiler Python 3.8 pin~~ | **Resolved.** The correct floor is 3.10, not 3.8. The Radxa NPU SDK page is accessible. | §2.2a |
| ~~GDN-2 numeric RULER scores~~ | **Resolved.** Scores extracted from the paper's Table 3 (MK-NIAH-1). | §1.5 |
| ~~Per-checkpoint layer counts and `layer_types`~~ | **Resolved.** Read from `config.json` for both 0.8B and 4B checkpoints. | [`GDN_LAYER_AUDIT.md`](GDN_LAYER_AUDIT.md) |
| ~~Whether `causal_conv1d` / `fla` have any aarch64 build~~ | **Resolved.** Confirmed absent — no Arm/Vulkan/OpenCL build exists for either package. | [`FINDINGS.md`](FINDINGS.md), [`DEVPOST_SUBMISSION.md`](DEVPOST_SUBMISSION.md) |

## Sources

- [Arm Create: AI Optimization Challenge — Rules](https://arm-ai-optimization-challenge.devpost.com/rules)
- [Radxa Orion O6 product page](https://radxa.com/products/orion/o6/)
- [Radxa Orion O6 — Artificial Intelligence docs](https://docs.radxa.com/en/orion/o6/app-development/artificial-intelligence)
- [Qwen3.5 — transformers docs](https://huggingface.co/docs/transformers/model_doc/qwen3_5)
- [Qwen3.5 MoE — transformers docs](https://huggingface.co/docs/transformers/model_doc/qwen3_5_moe)
- [Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention (arXiv 2605.22791)](https://arxiv.org/abs/2605.22791)
- [Atlas-Inference/gdn Hub kernel](https://huggingface.co/kernels/Atlas-Inference/gdn)
