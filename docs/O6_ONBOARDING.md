# Orion O6 Onboarding Checklist

First-hours playbook for when the Orion O6 (CIX P1) arrives.
Maps each step to its bead, script, and expected output.

**Goal:** go from unboxing to first real benchmark numbers in under 2 hours.

---

## Phase A: Board bring-up (~30 min)

- [ ] **Flash Debian 12** (ob-iae)
  - Download image from Radxa Download Station
  - Flash to eMMC or NVMe (balenaEtcher / dd)
  - First boot, SSH access

- [ ] **Verify connectivity**
  ```bash
  ssh <o6-host>
  uname -m                    # expect aarch64
  grep -m1 Features /proc/cpuinfo  # expect sve2, i8mm, bf16, asimddp
  nproc                       # expect 12
  free -m                     # expect ~64 GB
  ```

## Phase B: System baseline (~10 min)

- [ ] **Run automated baseline** (ob-41j, uses ob-41j.1)
  ```bash
  sudo bash scripts/o6_system_baseline.sh
  ```
  - Auto-detects 3 clusters (A720 big/med + A520 little)
  - Sets performance governor
  - Captures thermals, memory, OPPs
  - Prints ready-to-paste taskset commands

- [ ] **Record expected topology**
  ```
  Cluster    CPUs    Core         Max Freq   ISA
  BIG        0-3     A720 big     2.8 GHz    Armv9.2-A SVE2 128-bit, i8mm, bf16, dotprod
  MEDIUM     4-7     A720 medium  2.4 GHz    Same ISA features
  LITTLE     8-11    A520 little  1.8 GHz    Armv8.2-A (NO SVE2 — use armv8a binary)
  ```
  > ⚠ Verify actual CPU→cluster mapping with the baseline script. The numbering
  > may differ from this expectation.

## Phase C: GDN kernel benchmarks (~20 min)

- [ ] **Build bench binaries** (ob-8ms.3)
  ```bash
  # On an x86 host (cross-compile):
  ./scripts/build_device_bench.sh
  # Or native on the O6:
  gcc -O3 -fopenmp -march=armv9-a+sve2+i8mm+bf16 -static \
      src/orionsbelt/engines/cpu/kernels/gdn_sve.c \
      src/orionsbelt/engines/cpu/kernels/bench_gdn.c \
      -o dist/bench_gdn_orion_a720 -lm
  ```

- [ ] **Run benchmarks on each cluster**
  ```bash
  # A720 big cluster (SVE2 path):
  taskset -c <big-range> ./dist/bench_gdn_orion_a720 --repeats 30 --csv > results/raw/orion-o6_big.csv

  # A720 medium cluster:
  taskset -c <med-range> ./dist/bench_gdn_orion_a720 --repeats 30 --csv > results/raw/orion-o6_med.csv

  # A520 little cluster (Armv8-A fallback — NO SVE2):
  taskset -c <little-range> ./dist/bench_gdn_armv8a --repeats 30 --csv > results/raw/orion-o6_little.csv
  ```

- [ ] **Capture provenance** (one per device, not per run)
  ```bash
  python3 bench/manifest.py > results/manifests/orion-o6.json
  ```

## Phase D: NPU smoke test (~20 min) — ob-huw

- [ ] **Check NPU availability**
  ```bash
  ls /dev/npu* /dev/cix* 2>/dev/null    # device nodes
  lsmod | grep -iE 'npu|cix|rknn'       # kernel modules
  ```

- [ ] **Install NOE SDK** (runs on x86 host, not on board)
  ```bash
  # On x86 host with Python 3.10:
  conda create -n noe python=3.10 && conda activate noe
  pip install cix-noe-umd cixbuild
  # Verify:
  cixbuild --help
  ```

- [ ] **Run a CIX Model Hub reference model**
  - Download from CIX AI Model Hub on ModelScope
  - Compile with cixbuild
  - Deploy and run on the O6 NPU
  - Record throughput + whether GDN layers compile at all

## Phase E: NOE op-coverage audit (~30 min) — ob-8xc

- [ ] **On-device verification of compile-time audit**
  - ob-t3b.1 already ran the x86-host compile audit
  - This step verifies compiled artifacts actually run on the NPU
  - Key question: does the sequential scan execute or silently fall back?

## Phase F: First results comparison

- [ ] **Compare achieved bandwidth vs spec**
  - O6 spec: 100 GB/s = 93.1 GiB/s (128-bit LPDDR5 @ 5500 MT/s)
  - If achieved bandwidth is <60% of spec, investigate DMC frequency and governor

- [ ] **Compare GDN kernel throughput across fleet**
  ```bash
  python3 bench/fleet_analysis.py
  ```
  - Key test: does the O6 with SVE2 128-bit beat RK3588 with NEON?
  - If SVE2 shows no gain over NEON at 128-bit, that's an expected finding
    (SVE2 wins predication, not vector width, per ob-8qt.1 notes)

---

## Critical decisions to make on the O6

1. **Phase-dependent mapping** (ob-o4g): Can the NPU handle the delta-rule matmul
   in prefill, while CPU keeps the sequential scan? This is the ADR that defines
   the heterogeneous mapping.

2. **Vendor baseline** (ob-mrd.1): Does cix-llama-cpp handle GDN at all?
   If not, our CPU-only path IS the reference implementation.

3. **Thermal behavior** (ob-dgn): Does the 12-core SoC throttle under sustained
   decode load? The 64 GB LPDDR5 gives thermal headroom the RK3588 lacks.

---

## File locations

| What | Where |
|---|---|
| Bench source | `src/orionsbelt/engines/cpu/kernels/` |
| KleidiAI submission | `kleidiai_submission/` |
| E2E decode loop | `src/orionsbelt/engines/cpu/kernels/gdn_e2e_decode.c` |
| Results CSV | `results/raw/` |
| Manifests | `results/manifests/` |
| Findings | `docs/FINDINGS.md` |
| Beads commands | `bd ready`, `bd show <id>`, `bd update <id> --claim` |

---

## Beads unblocked by O6 arrival

| Bead | Task | Priority |
|---|---|---|
| ob-iae | Flash Debian 12 + first boot | P0 |
| ob-41j | System baseline | P1 |
| ob-huw | NPU runtime smoke test | P1 |
| ob-88p | Validate Immortalis G720 Vulkan/OpenCL | P1 |
| ob-v1f | Install perf/Performix on device | P1 |
| ob-8xc | NOE op-coverage audit (on-device) | P1 |
| ob-mrd.1 | Vendor cix-llama-cpp baseline | P1 |
| ob-o4g | Layer-to-engine assignment ADR | P1 |
| ob-i8v | Partitioned execution runtime | P1 |
| ob-7a9 | Dynamic heterogeneous dispatcher | P2 |
| ob-agf | Power/energy sampling | P2 |
| ob-dgn | Thermal-throttle characterization | P2 |
| ob-41j.1 | ✅ System baseline script (DONE) | P2 |
