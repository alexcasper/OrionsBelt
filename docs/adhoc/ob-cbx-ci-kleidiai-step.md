# ob-cbx: CI Step for KleidiAI GDN Kernel QEMU Verification

**Status:** Patch ready, blocked on GitHub OAuth `workflow` scope.
The script `scripts/verify_kleidiai_kernels.sh` exists and is tested.
Only the CI YAML wiring is missing — this file contains the exact patch.

## Why it's blocked

GitHub requires the `workflow` scope on the OAuth token to push changes to
`.github/workflows/`. Neither t3's nor t4's token has this scope. Anyone
with a token that includes `workflow` scope can apply this in 30 seconds.

## Patch (apply to `.github/workflows/ci.yaml`)

Add these 3 lines after the existing `verify_cpu_kernels.sh` step in the
`kernels` job (after line 92):

```yaml
      - name: Cross-compile and verify KleidiAI GDN submission kernels under QEMU
        run: bash scripts/verify_kleidiai_kernels.sh
```

That's it — the `kernels` job already installs `gcc-aarch64-linux-gnu` and
`qemu-user`, which is all the script needs. No additional packages or setup.

## What it tests

`scripts/verify_kleidiai_kernels.sh` cross-compiles the four KleidiAI GDN
micro-kernels for five ISA levels (SVE2, SVE1@128, SVE1@256, NEON-A57,
NEON-A76) and runs each under QEMU, checking that all 14 tests pass at every
tier. This is the only place the SVE code path gets exercised, since no device
in the fleet has SVE.

## Git patch

```diff
diff --git a/.github/workflows/ci.yaml b/.github/workflows/ci.yaml
--- a/.github/workflows/ci.yaml
+++ b/.github/workflows/ci.yaml
@@ -91,6 +91,9 @@ jobs:
       - name: Cross-compile and verify GDN kernels under QEMU
         run: bash scripts/verify_cpu_kernels.sh
 
+      - name: Cross-compile and verify KleidiAI GDN submission kernels under QEMU
+        run: bash scripts/verify_kleidiai_kernels.sh
+
   # ---------------------------------------------------------------------------
   # onnx-probes
```
