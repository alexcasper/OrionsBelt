==========================================================================================
ABLATION (model-level, tidy long, cpu/cpu hybrid, 4K)
  run_id    : ['generic_aarch64_20260806T145501Z_23550d6']
  git_sha   : ['23550d6']  (manifest file MISSING from results/manifests/)
  manifest  : ['results/manifests/generic_aarch64_20260806T145501Z_23550d6.json']
  context   : ['32768', '4096']  quant: ['fp16']  engines: [('cpu', 'cpu')]
  model     : ['Qwen/Qwen3.5-4B@ablation-cpu-only']
  --- p50 over repeats (n) ---
  prefill_tokens_per_sec : 25761926088.2  (n=10)
  decode_tokens_per_sec  : 4652473.1      (n=10)
  ttft_seconds           : 0.002243       s = 2.2429 ms (n=10)
  mem[weights][prefill] : 7.8 GiB      (8411693056 bytes, n=10)
  mem[weights][decode] : 7.8 GiB      (8411693056 bytes, n=10)
  mem[kv_cache][prefill] : 576.0 MiB    (603979776 bytes, n=10)
  mem[kv_cache][decode] : 576.6 MiB    (604602368 bytes, n=10)
  mem[recurrent_state][prefill] : 48.0 MiB     (50331648 bytes, n=10)
  mem[recurrent_state][decode] : 48.0 MiB     (50331648 bytes, n=10)

==========================================================================================
KERNEL FLEET BASELINE (4B/0.8B, seq=64, single-core canonical, fp32) — GiB/s @ p50
Device            run_id_sha            dirty   CumDecay  Scan    DWConv1D  GDN2
  Jetson j1 (canon) 2c9ac9f              dirty   3.63   2.97   3.57   2.93 
  Jetson j2 (canon) 6ea1771              dirty   1.15   0.73   1.04     —  
  Pi5 r5           28729f3              dirty   3.74   1.20   3.23     —  
  Pi5 j1           f127a11              dirty   2.93   1.84   2.37     —  
  RK3588 t4 big    28729f3              dirty   22.47  11.09  23.00  6.84 
  RK3588 t3 big    553a96e              CLEAN   23.17  10.33  21.34  9.04 
  RK3588 t4 little 28729f3              dirty   5.87   3.53   5.13   1.52 
  RK3588 t3 little 553a96e              CLEAN   5.56   2.71   5.35   1.24 
  (t4 preferred over t3 per ob-bf7: t3 scan spread=153% contaminated; t4 spread=17%)

  --- Fleet sweep (ob-bf7): commit 234807d, clean tree, single-thread ---
  RK3588 t4 big (clean) 234807d             CLEAN  7.29   5.27   6.98   3.22 
  RK3588 t3 big (clean) 234807d             CLEAN  6.15   2.91   5.96   2.00 
  RK3588 t4 little (cln) 234807d             CLEAN  1.41   0.81   1.12   0.50 
  RK3588 t3 little (cln) 234807d             CLEAN  1.19   0.55   1.12   0.49 
  Jetson j1 (clean)    234807d             CLEAN  1.59   1.18   1.41   1.13 
  Jetson j2 (clean)    234807d             CLEAN  1.50   1.09   0.93   1.07 
  (All at commit 234807d, dirty=false, OMP_NUM_THREADS=1, governor=performance)

==========================================================================================
OPTIMIZATION LADDER on Jetson (Qwen3.5-4B, seq=64) — GiB/s @ p50 / p50 µs
Config                                        CumDecay       Scan          DWConv1D
  baseline single (j1, manifest, dirty)      3.63 GiB/s (538us) 2.97 GiB/s (997us) 3.57 GiB/s (576us)
  baseline single (j2, manifest, dirty)      1.15 GiB/s (1696us) 0.73 GiB/s (4062us) 1.04 GiB/s (1988us)
  CLEAN tree 4-core OMP (j1, dirty=false)    3.79 GiB/s (516us) 2.92 GiB/s (1014us) 3.60 GiB/s (572us)
  single-core (j2_single, NO manifest)       1.32 GiB/s (1485us) 1.13 GiB/s (2626us) 1.20 GiB/s (1710us)
  4-core OMP (j2_omp, NO manifest)           2.35 GiB/s (829us) 1.97 GiB/s (1501us) 2.26 GiB/s (912us)
  4-core OMP full (j2_omp-full, NO manif)    3.85 GiB/s (508us) 2.96 GiB/s (1001us) 3.66 GiB/s (563us)
  conv-unroll (j2, NO manifest)              3.85 GiB/s (507us) 2.91 GiB/s (1018us) 3.57 GiB/s (578us)
  omp+unroll (j2, NO manifest)               3.71 GiB/s (526us) 2.95 GiB/s (1003us) 2.29 GiB/s (901us)
  FULL-OPTIMIZED (j2, manifest, dirty)       3.82 GiB/s (512us) 2.97 GiB/s (997us) 3.61 GiB/s (571us)

  Within-j2-series speedups (4B seq=64, GiB/s):
    gdn_cumdecay         single=1.32  omp=2.35 (1.78x)  full-opt=3.82 (2.89x)
    gdn_gated_scan       single=1.13  omp=1.97 (1.74x)  full-opt=2.97 (2.63x)
    gdn_causal_dwconv1d  single=1.20  omp=2.26 (1.88x)  full-opt=3.61 (3.01x)
  Manifest-backed (j1 device): single j1.csv scan=2.97 -> 4-core CLEAN j1_clean scan=2.92 (0.98x, SUPERLINEAR=confounded)

==========================================================================================
DECODE (seq=1) KERNEL — Qwen3.5-4B — p50 µs/token + GiB/s (recurrence cost)
   j1 CLEAN 4-core (manifest)
     gdn_cumdecay           3.646 us/token   8.37 GiB/s   spread=11.4%
     gdn_gated_scan         4.740 us/token   16.10 GiB/s   spread=2.2%
     gdn_causal_dwconv1d    11.146 us/token   12.32 GiB/s   spread=3.7%
   j2 full-optimized (manifest)
     gdn_cumdecay           3.802 us/token   8.03 GiB/s   spread=13.7%
     gdn_gated_scan         4.479 us/token   17.03 GiB/s   spread=12.8%
     gdn_causal_dwconv1d    9.219 us/token   14.90 GiB/s   spread=1.1%

==========================================================================================
SUSTAINED LOAD / THERMAL (gdn_gated_scan, Qwen3.5-4B, seq=64)
  j1 sustained 120s (manifest, dirty)    first=0.77 last=0.76 median=0.76 GiB/s  drift=-1.3%  (n=24 samples)
  j2 1-core 30s (manifest, dirty)        first=1.03 last=1.03 median=1.03 GiB/s  drift=+0.0%  (n=6 samples)
  j2 4-core 30s (manifest, dirty)        first=2.36 last=2.40 median=2.37 GiB/s  drift=+1.7%  (n=6 samples)
  j2 optimized 120s (manifest, dirty)    first=2.80 last=2.71 median=2.79 GiB/s  drift=-3.2%  (n=24 samples)

==========================================================================================
POWER / ENERGY (jetson-j1_power.json, INA3221, 10s sustained, Qwen3.5-4B seq=64)
  gdn_gated_scan         0.74 GiB/s  delta_board= 925mW  delta_cpu= 619mW  energy=1250 mJ/GiB(board)  836 mJ/GiB(cpu)  therm 0.0517->0.052C
  gdn_causal_dwconv1d    0.88 GiB/s  delta_board= 903mW  delta_cpu= 675mW  energy=1026 mJ/GiB(board)  767 mJ/GiB(cpu)  therm 0.0517->0.052C
  gdn_cumdecay           1.06 GiB/s  delta_board= 925mW  delta_cpu= 706mW  energy= 874 mJ/GiB(board)  667 mJ/GiB(cpu)  therm 0.052->0.0525C
  provenance: bead ob-agf.1/ob-mrd.7, run_id jetson-j1 (no git sha in this manifest)

DONE.
