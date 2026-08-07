# ADR 0002: Select the portable aarch64 hedge target

- **Status:** Superseded by [ADR 0005](./0005-device-fleet-and-bandwidth-study.md) (device fleet selected) and [ADR 0007](./0007-commit-to-edge-ai-track.md) (Edge AI track committed)
- **Date:** 2026-08-02
- **Bead:** `ob-zh4`
- **Deciders:** Claude (agent), pending maintainer confirmation (see "What I need from the maintainer" below)

## Context

Per PLAN.md §1 and §7 (R1, R2), we hold neither the Orion O6 board nor CIX Early Bird
SDK access as of T-12, and both are externally gated with no way to compress the
timeline. The hedge track (epic E3, `ob-8ms`) exists so a submittable entry survives
even if the board never arrives, targeting the **Edge AI** track (confirmed in
[`docs/CLAIM_VERIFICATION.md`](../CLAIM_VERIFICATION.md) §1.1 — there is no "Mobile AI"
track, and the hedge therefore does not have to be a phone).

Two things must be demonstrable on whatever device we pick, per PLAN.md §2.4:

1. **Prefill/TTFT throughput scaling with context length** — the metric where GDN kernel
   work genuinely pays, per the verified upstream finding that optimized GDN kernels
   speed up prefill 1.38–1.49× while leaving decode flat.
2. **The KV-cache-vs-GDN-recurrent-state memory split** — full attention grows linearly
   with context, GDN state stays flat. This is the architectural claim, not a kernel win,
   and it's the thing that makes long context viable on memory-scarce silicon.

Both should run through Arm i8mm/SVE CPU paths, "ideally" (soft requirement) with a
Vulkan or OpenCL GPU compute path. Everything here must also survive PLAN.md's working
agreements (§9): every run gets a manifest, percentiles not single best runs, thermal
state captured, negative results reported honestly rather than hidden.

The O6's CPU is confirmed Armv9.2 (Cortex-A720/A520), which the CIX P1 TRM documents as
supporting SVE2 ([CNX Software](https://www.cnx-software.com/2025/12/13/cix-releases-p1-cpu-trm-and-developer-guides-for-gpu-ai-accelerator-os-and-firmware-bios/);
[Arm Cortex-A720 product page](https://www.arm.com/products/silicon-ip-cpu/cortex-a/cortex-a720)),
and a Mali-derived Immortalis G720 GPU reached via Vulkan/OpenCL. Anything the hedge
produces is worth more if its CPU intrinsics and GPU shader logic carry over to that
target rather than being thrown away at go/no-go (Aug 9).

### Option 1 — Android phone via Termux

**For:**
- The only candidate that is unambiguously "Edge AI" to a judge — a phone SoC with
  big.LITTLE, a real battery/thermal envelope, and a GPU reached the same way (Vulkan)
  the O6 will be reached.
- Termux + llama.cpp Vulkan support is real and actively maintained in 2026: recent
  community reports show automatic Mali GPU acceleration and community 1-click
  installers with "Turnip/Mesa Vulkan GPU acceleration for Snapdragon chips"
  ([llama.cpp discussion #23193](https://github.com/ggml-org/llama.cpp/discussions/23193);
  [sanatani-hackers/Llama.cpp-termux](https://github.com/sanatani-hackers/Llama.cpp-termux)).
- Recent flagship silicon closes the SVE2 gap that used to make phones a poor CPU-path
  match: the original Snapdragon 8 Elite (Oryon, Armv8.7-A) has no SVE2, but **Snapdragon
  8 Elite Gen 5 adds SVE2 and SME1** ([search-derived summary, see Sources](#sources)),
  and several MediaTek Dimensity flagships in this class use the same
  Cortex-X4/A720/A520 Armv9.2 core mix as the CIX P1, which is a genuinely strong
  transfer story if the device in hand is that recent.

**Against (stated honestly, not glossed over):**
- **Thermal throttling is severe, not cosmetic.** Independent measurement: "the iPhone
  16 Pro loses nearly half its throughput within two iterations," and on a Galaxy S24
  Ultra "the thermal governor slammed GPU frequency from 680 MHz down to 231 MHz when
  the GPU hit 78.3°C," with sustained inference on a warm phone dropping 20–30% from
  cold-start ([arXiv:2410.03613](https://arxiv.org/html/2410.03613), device-benchmark
  summary surfaced via search — treat the exact numbers as **secondary-source and
  unverified against the primary paper**, but the qualitative pattern — severe,
  fast-onset throttling — is corroborated by multiple independent write-ups). This
  directly threatens our own working agreement to report stable percentiles rather than
  a lucky best run.
- **RAM ceiling makes 262K context implausible, full stop.** 2026 flagships commonly
  ship 12–16GB ([Smartprix Snapdragon 8 Elite listings](https://us.smartprix.com/mobiles/snapdragon-8-elite-gen5-phones-list) — unverified beyond marketing listings). A
  4B model's growing full-attention KV cache at long context, plus OS overhead, plus
  Termux's own footprint, will not fit alongside a 262K window. This is not fatal — the
  descope ladder (PLAN.md §7, R4) already plans an incremental sweep (4K→32K→128K→262K)
  and permits dropping the top point at T-1 — but it means the phone caps out well
  before the full sweep, likely around 32K–128K depending on model size chosen.
- **Toolchain is hobbyist-grade, not vendor-blessed.** The same search evidence that
  shows Vulkan works also shows it is fragile: "Vulkan was 2x worse than CPU-only in
  some cases" and "some official Qualcomm drivers cannot handle certain quantization
  formats (Q4_K), requiring a switch to the Turnip driver"
  ([llama.cpp PR #9672](https://github.com/ggml-org/llama.cpp/pull/9672);
  [llama.cpp discussion #23193](https://github.com/ggml-org/llama.cpp/discussions/23193)).
  Budget real bring-up time for this in `ob-ng6`.
- **No reliable power/energy instrumentation without root.** Android has no
  vendor-neutral rail-level power API comparable to `powermetrics` on macOS; anything
  we report will be battery-drain estimation, and that must be disclosed as best-effort
  rather than lab-grade in the write-up.
- **SVE2 support is device-dependent, not guaranteed.** If the maintainer's phone is a
  Snapdragon 8 Elite Gen 4 (or older) or a mid-range SoC, there is no SVE2 at all — only
  NEON + (likely) i8mm. That's still a legitimate, honestly-reported partial answer to
  the "i8mm/SVE" ask, just not the whole thing.

### Option 2 — AWS Graviton (or other cloud Arm)

**For:**
- Confirmed to satisfy the CPU-path requirement outright and immediately: Graviton3/4
  (Neoverse V1/V2) support SVE, bfloat16, and i8mm/int8 dot-product instructions as
  documented Arm extensions
  ([Arm Neoverse V1 blog](https://developer.arm.com/community/arm-community-blogs/b/architectures-and-processors-blog/posts/neoverse-v1-platform-a-new-performance-tier-for-arm);
  [AWS Graviton getting-started guide](https://github.com/aws/aws-graviton-getting-started)).
  Graviton2 (Neoverse N1) does **not** have SVE, so the instance family choice matters —
  this must be c7g/c8g/m7g/r7g or newer, not c6g.
- Zero device-ownership risk, provisionable in minutes, fully scriptable in CI, ample
  RAM (r7g/r8g instances scale to hundreds of GB), so the full 262K context sweep is
  trivially in reach with no descoping.
- Most reproducible option for judges to independently re-run: pin an instance type and
  AMI, and the exact same numbers should come back out.

**Against — and this is the disqualifying one, not a minor one:**
- **No credible Vulkan/OpenCL GPU compute path exists on the general Graviton fleet.**
  Standard compute instances (c7g/m7g/r7g) are CPU-only. The one Arm+GPU pairing AWS
  ships, **G5g, pairs Graviton2 (no SVE) with an NVIDIA T4G Tensor Core GPU** — a CUDA
  device, not a Vulkan/OpenCL one, and not Arm-vendor GPU IP at all, so nothing learned
  there transfers to the O6's Immortalis/Mali Vulkan target
  ([NVIDIA technical blog on G5g](https://developer.nvidia.com/blog/aws-launches-first-nvidia-gpu-accelerated-graviton-based-instance-with-amazon-ec2-g5g/)).
  There is no Graviton3/4 + Vulkan-capable-GPU instance type. The GPU leg of the hedge
  is simply absent here, not degraded.
- **Credibility to a judge is the deeper problem, and it is about track fit, not
  polish.** This competition has a *separate* Cloud AI track. An "Edge AI" submission
  whose benchmark numbers come from an AWS EC2 instance reads, to any judge who checks,
  as a category mismatch — the hedge exists specifically because Physical AI (the O6)
  might not materialize, and Cloud AI is already a different track we are not entering.
  Submitting server silicon under Edge AI risks looking like we are gaming the track
  rather than demonstrating an edge result, which cuts against WOW factor and Potential
  Impact (PLAN.md §2.2) far more than a rough or throttled phone number would.

### Option 3 — Apple silicon (M-series)

**For:**
- By far the most mature non-CUDA backend: llama.cpp's Metal path is the best-supported
  GPU backend in the project outside CUDA, and `powermetrics` gives genuinely good
  rail-level power/energy instrumentation, unlike Android.
- Best memory bandwidth of the three by a wide margin, and it scales cleanly across
  tiers (M4 base ≈120GB/s, M4 Pro ≈200–273GB/s, M4 Max ≈409.6GB/s, M2/M3 Ultra
  ≈800GB/s — figures aggregated from a comparison roundup,
  [laptopmedia.com](https://laptopmedia.com/comparisons/apple-m4-vs-m3-pro-max-vs-m2-pro-max-ultra-vs-m1-pro-max-ultra-the-ultimate-benchmark-comparison/),
  **unverified against Apple's own spec sheets** — treat as directionally right, not
  exact). That range would make the *bandwidth-bound decode* finding unusually vivid to
  show, since you can watch it saturate differently by tier.

**Against — and two of these are hard, not soft, misses:**
- **No SVE, at all, ever, on any Apple Silicon chip shipped to date, including the
  latest.** Multiple independent sources agree the M4 "lacks SVE (and SVE2) support...
  the LLVM compiler officially flags the M4 as supporting ARMv8.7a," prioritizing SME
  instead ([Phoronix](https://www.phoronix.com/news/Apple-M4-Added-To-LLVM-Clang);
  [Apple Developer Forums thread](https://developer.apple.com/forums/thread/757704)).
  i8mm *is* supported from M2 onward, so half of the named "i8mm/SVE" ask is available —
  but the SVE half is categorically impossible here, not merely absent from one device
  in hand.
- **Metal is not Vulkan, and the translation layer is leaky.** MoltenVK converts
  SPIR-V to Metal Shading Language, but documented limitations include no 64-bit atomics
  on shared/buffer memory and no cooperative-matrix-to-simdgroup-matrix translation
  ([KhronosGroup/MoltenVK](https://github.com/KhronosGroup/MoltenVK)), and llama.cpp
  itself has open issues around Vulkan-via-MoltenVK shader compilation crashes
  ([llama.cpp issue #15498](https://github.com/ggml-org/llama.cpp/issues/15498)). A
  chunkwise gated-delta-rule scan kernel written and validated against MoltenVK is not
  the same artifact as one validated against the O6's native Vulkan driver — it would
  need re-validating there regardless, undercutting the "work transfers to O6" case
  this ADR is supposed to protect.
- **Weakest Arm-vendor framing of the three, and this is a real Devpost-optics risk for
  an "Arm Create" challenge.** Apple designs fully custom cores outside the Arm
  Neoverse/Cortex partner ecosystem the O6, CIX, and Graviton all sit inside. A demo
  recorded on a MacBook for a challenge framed around Arm-partner silicon invites the
  same "is this actually what you said it is" scrutiny that disqualifies Graviton for
  Edge AI, just on the vendor-alignment axis instead of the track-fit axis.

## Decision

**Primary hedge target: Android phone via Termux**, using the most capable aarch64
Android device the maintainer currently owns.

**Default if no suitable phone is available or bring-up stalls: AWS Graviton3 or
Graviton4** (c7g/c8g family or newer — not c6g/Graviton2, which lacks SVE), used
explicitly as a CPU/SVE2/i8mm-only reproducibility target with the GPU leg and the
"Edge AI" framing both disclosed as compromised in the write-up, not silently dropped.

Apple Silicon is **not** selected as the hedge target. It is ruled out on the one
criterion that is hardest to work around: it cannot show an Arm SVE path at all, on any
shipping chip, which is one of the two literally-named CPU requirements. It remains
usable as a last-resort, clearly-labeled *supplementary* data point (e.g., an
i8mm-only NEON baseline, or a Metal-native — not Vulkan — GPU number for context) if the
maintainer happens to own one and time allows, but it must never stand in as *the*
hedge result.

This decision is genuinely conditional on something only the maintainer knows: **what
aarch64 hardware they physically have access to today.** Absent that confirmation, the
default above (phone, falling back to Graviton) is what `ob-ng6` should proceed with.

### What I need from the maintainer

1. Which Android device(s) are in hand — model, RAM, SoC (Snapdragon generation or
   MediaTek Dimensity model), and whether Termux/root access is already set up. This
   determines whether SVE2 is available at all or whether the CPU story is i8mm-only.
2. Whether root or `adb` access is available, since it changes what power/thermal
   telemetry we can actually pull.
3. As a bring-up safety net for `ob-ng6`: confirmation of AWS account access (for the
   Graviton default) and, if owned, an Apple Silicon Mac model (for the last-resort
   supplementary path only).

If none of this is answered before `ob-ng6` needs to start, proceed on the phone
default and treat "no suitable phone" as discovered-during-bring-up, triggering the
Graviton fallback per the reversal trigger below.

## Alternatives considered

| Option | Why not (as primary) |
|---|---|
| AWS Graviton (or other cloud Arm) | Guarantees the CPU/SVE requirement and is the easiest to automate and reproduce, but has no Vulkan/OpenCL GPU-capable instance pairing at all on Graviton3/4, and reads as a track-fit mismatch for Edge AI given this competition has a separate Cloud AI track. Kept as the explicit fallback because availability and reproducibility are real and the CPU-only claim is still honestly demonstrable there. |
| Apple silicon (M-series) | Best toolchain maturity and memory bandwidth, and good power instrumentation, but categorically cannot demonstrate SVE on any shipping chip, and its Vulkan story only exists through a leaky MoltenVK translation layer whose limitations (no 64-bit atomics, no cooperative-matrix translation) mean GPU kernel work done here would still need re-validating on the O6's native Vulkan driver — the opposite of the "work transfers" goal. Also the weakest Arm-vendor-ecosystem alignment of the three. Kept only as an optional, clearly-labeled supplementary data point. |

What would change my mind: confirmation that the maintainer's only available aarch64
hardware is a Mac, or a phone so old/underpowered it cannot hold even the 0.8B
checkpoint — in either case the ranking above should be revisited with that constraint
made explicit, rather than defaulting silently.

## Consequences

**Accepted costs.**
- Thermal instability on the phone means some runs will need to be reported as
  degraded/throttled rather than clean, with the throttling itself documented as a
  finding (consistent with PLAN.md §9's stance that negative/partial results are worth
  reporting honestly).
- The context sweep on the phone will very likely stop short of 262K; per the descope
  ladder this is pre-accepted (R4), but it should be flagged early rather than
  discovered at T-2.
- Power/energy numbers from the phone are best-effort battery-drain estimates, not
  lab-grade — this must be labeled as such wherever it appears in results or the write-up.
- If the Graviton fallback triggers, the submission loses its GPU/Vulkan hedge leg
  entirely and its "Edge AI" framing weakens; that must be stated plainly in the
  write-up rather than papered over, per this project's own claim-verification ethic.

**Follow-on work (not filed as beads by this agent — no `bd` write commands run under
this task's constraints; recommend the maintainer or a future session file these under
`ob-8ms`):**
- A bring-up smoke test on the actual device in hand, as part of `ob-ng6`, to confirm
  at runtime (not just at compile time) which of SVE2/i8mm/dot-product and which GPU
  backend (Vulkan via Termux) are actually active — this ADR's technical claims about
  any *specific* device are unverified until that smoke test runs.
- If the Graviton fallback is taken, a short explicit paragraph for the write-up
  disclosing the degraded Edge AI framing and the missing GPU leg, so a judge sees an
  honest limitation rather than a hidden one.
- If time allows and a Mac is available, an optional supplementary Metal-native GPU
  number and NEON+i8mm CPU number, explicitly labeled as "not the hedge, for context
  only."

**Reversal cost / trigger.** Cheap to reverse before `ob-ng6` writes device-specific
code: the benchmark harness (E5) is being built hardware-independent by design, so
switching the hedge target mainly means re-running the same harness against a different
device, not rewriting it. The reversal trigger is discovery **during `ob-ng6` bring-up**
(target: within the M0/M1 window, Aug 2–6) that either (a) the phone in hand cannot
sustain even a short benchmark run without throttling to unusable levels within the
first few iterations, or (b) no phone capable of holding the smallest (0.8B) checkpoint
is available. Either triggers an immediate, undebated switch to the AWS Graviton3/4
default — this is pre-agreed now specifically so it is not re-litigated under time
pressure later, matching the spirit of the descope ladder in PLAN.md §7.

## Sources

- [PLAN.md](../archive/PLAN.md) (archived, see [docs/archive/README.md](../archive/README.md)) — §1 two-track structure, §2.4 verified prefill/decode finding, §7 risk register and descope ladder
- [docs/CLAIM_VERIFICATION.md](../CLAIM_VERIFICATION.md) — §1.1 Edge AI track correction
- [Arm Create: AI Optimization Challenge — Rules](https://arm-ai-optimization-challenge.devpost.com/rules)
- [CIX P1 CPU TRM release note (CNX Software)](https://www.cnx-software.com/2025/12/13/cix-releases-p1-cpu-trm-and-developer-guides-for-gpu-ai-accelerator-os-and-firmware-bios/)
- [Arm Cortex-A720 product page](https://www.arm.com/products/silicon-ip-cpu/cortex-a/cortex-a720)
- [llama.cpp discussion #23193 — Termux + Vulkan on Mali-G715](https://github.com/ggml-org/llama.cpp/discussions/23193)
- [llama.cpp PR #9672 — Android build updates](https://github.com/ggml-org/llama.cpp/pull/9672)
- [sanatani-hackers/Llama.cpp-termux](https://github.com/sanatani-hackers/Llama.cpp-termux)
- [arXiv:2410.03613 — mobile LLM inference performance study](https://arxiv.org/html/2410.03613) (thermal-throttling figures surfaced via search summary — re-verify against the primary paper before quoting exact numbers in the write-up)
- [aws/aws-graviton-getting-started](https://github.com/aws/aws-graviton-getting-started)
- [Arm Neoverse V1 platform blog](https://developer.arm.com/community/arm-community-blogs/b/architectures-and-processors-blog/posts/neoverse-v1-platform-a-new-performance-tier-for-arm)
- [NVIDIA blog — EC2 G5g Graviton2 + T4G](https://developer.nvidia.com/blog/aws-launches-first-nvidia-gpu-accelerated-graviton-based-instance-with-amazon-ec2-g5g/)
- [Phoronix — Apple M4 added to LLVM, ISA capabilities](https://www.phoronix.com/news/Apple-M4-Added-To-LLVM-Clang)
- [Apple Developer Forums — does M4 support SVE?](https://developer.apple.com/forums/thread/757704)
- [KhronosGroup/MoltenVK](https://github.com/KhronosGroup/MoltenVK)
- [llama.cpp issue #15498 — Vulkan/MoltenVK shader compile crash](https://github.com/ggml-org/llama.cpp/issues/15498)
- [laptopmedia.com — Apple M-series bandwidth comparison](https://laptopmedia.com/comparisons/apple-m4-vs-m3-pro-max-vs-m2-pro-max-ultra-vs-m1-pro-max-ultra-the-ultimate-benchmark-comparison/) (secondary aggregation, unverified against Apple's own spec sheets)
- [Smartprix — Snapdragon 8 Elite Gen 5 phones list](https://us.smartprix.com/mobiles/snapdragon-8-elite-gen5-phones-list) (marketing listing, unverified)
