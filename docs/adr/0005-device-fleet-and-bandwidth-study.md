# ADR 0005: Use the available Armv8 device fleet as the Edge AI target, and run a bandwidth-scaling study

- **Status:** Accepted
- **Date:** 2026-08-02
- **Bead:** follow-on to `ob-zh4`, implemented by `ob-8ms.2` / `ob-8ms.3`
- **Refines [ADR 0002](./0002-portable-hedge-target.md)**, which chose a hedge target *conditionally*
  because it did not know what hardware existed. It now does.

## Context

ADR 0002 selected an Android/Termux phone with AWS Graviton as fallback, and explicitly recorded
that it needed to know which devices the maintainer physically owned. The answer: **a Raspberry Pi 5,
an RK3588 board, and a Jetson Nano.** That is materially better than the phone plan, and it changes
the shape of the Edge AI submission.

### Verified device characteristics

ISA features confirmed from clang's predefined feature macros per `-mcpu`; bandwidth from vendor
documentation.

| Device | SoC | CPU | ISA level | dotprod | i8mm | SVE | Spec bandwidth | GPU | NPU |
|---|---|---|---|:-:|:-:|:-:|---:|---|---|
| Jetson Nano | Tegra X1 | 4× A57 | Armv8.0-A | **no** | no | no | **25.6 GB/s** | Maxwell (CUDA) | — |
| Raspberry Pi 5 | BCM2712 | 4× A76 | Armv8.2-A | yes | no | no | **~17 GB/s** | VideoCore VII | — |
| RK3588 board | RK3588 | 4× A76 + 4× A55 | Armv8.2-A | yes | no | no | **~34 GB/s** (quad 16-bit LPDDR4x/5, *estimated*) | **Mali-G610 MP4** | ~6 TOPS |
| *Orion O6 (target, absent)* | CIX P1 | 4× A720 big + 4× A720 med + 4× A520 | Armv9.2-A | yes | **yes** | **yes (SVE2)** | **100 GB/s** | Immortalis-G720 MC10 | 28.8 TOPS |

**None of the three available devices has SVE.** All take the NEON path. That retrospectively
justifies writing the NEON kernels — had they been SVE-only, none of this hardware could have run
them, and the Edge AI track would have had no measurable target at all.

The Pi 5's bandwidth being *lower* than the Jetson Nano's despite far newer cores is not a mistake:
BCM2712 has a 32-bit LPDDR4X interface where Tegra X1 has 64-bit LPDDR4. That inversion is useful —
see below.

## Decision

**1. The RK3588 board is the primary Edge AI target.** It is the closest available analogue to the
Orion O6 on the three axes that matter to this project:

- **Asymmetric CPU clusters** (A76 big + A55 little ↔ A720 big/medium + A520 little), so the
  big.LITTLE affinity work (`ob-dqu`) is developable and measurable now rather than speculatively.
- **An Arm Mali GPU** (Mali-G610 MP4) in the same vendor family as the O6's Immortalis-G720. These
  are different generations — Valhall-era versus Arm's 5th-gen architecture — so performance will
  not transfer, but **Vulkan compute shaders and the OpenCL path largely will**. That makes the GPU
  scan kernel (`ob-q44`) developable without the O6, which was previously blocked outright.
- **Its own NPU** (~6 TOPS, Rockchip RKNN), which allows a second, independent test of the central
  operator-coverage finding: does *another* vendor's NPU toolchain also fail to express a
  runtime-length recurrence? If it does, the finding generalises well beyond CIX and becomes much
  stronger.

**2. The Raspberry Pi 5 is the reproducibility control.** Homogeneous A76, no cluster asymmetry to
confound results, and — the real argument — it is the single most widely available Arm SBC in the
world. A judge can reproduce our numbers on one. That is worth more for the Developer Experience
criterion than a marginally faster board would be.

**3. The Jetson Nano is the floor case.** Armv8.0-A with no dotprod and no fp16 vector arithmetic,
so it exercises the least-capable dispatch path. Its value is demonstrating graceful degradation
across a 3-generation ISA span, and its 4GB of RAM forces the 0.8B checkpoint, which tests the
fallback selection from ADR 0003 for real.

**4. Run a bandwidth-scaling study as a first-class result**, not just three separate benchmarks.

### Why the scaling study is the valuable part

The project's central technical claim is that GDN decode is **memory-bandwidth-bound** — argued from
an arithmetic intensity of ~0.25 FLOP/byte ([`METRICS.md`](../METRICS.md)). So far that is an
argument from first principles plus one upstream measurement on unrelated hardware.

These three devices span **17 → 25.6 → ~34 GB/s**, roughly a 2× range, and the O6 would extend it
to 100 GB/s — a ~6× span in total. If the kernels are genuinely bandwidth-bound, then achieved
throughput should track spec bandwidth approximately linearly, *independent of core generation*.
That yields three things a single-device benchmark cannot:

1. **A falsifiable test of the thesis.** Three points on a bandwidth axis either line up or they do
   not. If Pi 5 (17 GB/s, newest cores) underperforms Jetson Nano (25.6 GB/s, oldest cores) on the
   scan kernel, that is strong evidence for bandwidth-boundedness — the *inverted* core-generation
   ordering makes it a genuinely discriminating experiment rather than a confirmation.
2. **A prediction for the O6 before we own one.** Extrapolating to 100 GB/s gives a number we can
   publish as a prediction and later check. Predicting and then confirming is a much stronger
   result than measuring once.
3. **A finding nobody appears to have published**: GDN kernel scaling across Arm memory systems.

## Alternatives considered

| Option | Why not | What would change our mind |
|---|---|---|
| Android phone via Termux (ADR 0002's original choice) | The fleet is strictly better: SSH-accessible, thermally stable, no Termux toolchain friction, and the RK3588 uniquely brings cluster asymmetry plus a Mali GPU plus an NPU. Phone thermal throttling was ADR 0002's own headline risk. | If a phone with an Armv9 SVE2 core were available, it would become the only way to measure the SVE path on real silicon. |
| AWS Graviton | Still useful as a scripted, judge-reproducible CI-like target, and it is the *only* way to measure SVE on real hardware (Graviton3 has SVE1). But it is server silicon, which weakens an Edge AI framing. | Keep as an optional supplementary datapoint specifically to get one real SVE measurement. |
| Pick one device and go deep | Wastes the scaling study, which is the strongest available result and is only possible with several bandwidths. | If time collapses, the T-2 descope should cut to RK3588 alone. |
| Treat the RK3588 NPU as a primary target | Scope risk: a second vendor NPU toolchain (RKNN) is its own learning curve, and the submission is about Arm, not Rockchip. | Worth it only as a cheap generalisation check on the operator-coverage finding — a day at most. |

## Consequences

**Accepted costs.** No available device can measure SVE/SVE2, i8mm, the Immortalis GPU, or the CIX
NPU. Any claim about those remains either O6-gated or must be dropped. The RK3588 bandwidth figure
is *estimated* from its quad-channel 16-bit interface and needs confirming on the device itself —
`bench_gdn` reports achieved GiB/s precisely so the spec figure can be checked rather than trusted.

**Follow-on work.** `ob-8ms.3` (run the fleet study and publish the scaling result), `ob-dqu` (now
unblocked for real big.LITTLE work on RK3588), `ob-q44` (GPU scan kernel now developable on
Mali-G610), and a cheap RKNN operator-coverage check to test whether the recurrence gap generalises.

**Reversal cost.** Low. The benchmark binary is already built per-core (`bench_gdn_pi5_a76`,
`bench_gdn_rk3588_a76`, `bench_gdn_rk3588_a55`, `bench_gdn_jetson_a57`) and the kernels are
unchanged across all of them — the same source, verified identical on every target. Switching
emphasis between devices is a choice of which CSV to feature, not a rebuild.
