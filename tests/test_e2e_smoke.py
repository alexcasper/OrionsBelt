"""End-to-end smoke test: harness → schema → CSV → manifest → summary (bead ob-1lm).

Exercises the full benchmark pipeline with a synthetic backend and a tiny
config so it finishes in CI time (under 2 seconds).  No model weights,
no GPU, no NPU — pure stdlib.

Run without pytest::

    PYTHONPATH=bench:. python3 tests/test_e2e_smoke.py
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# bench/ is on the path for schema/harness imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench"))

from schema import (
    ResultRow,
    SchemaValidationError,
    validate_row,
    validate_rows,
    write_csv,
    read_csv,
)
from harness import (
    HarnessConfig,
    SyntheticBackend,
    run_single,
    result_to_rows,
    run_sweep,
    summarize,
)
from manifest import capture, write


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_failures = []
_passes = 0


def check(condition, msg):
    global _passes
    if condition:
        _passes += 1
    else:
        _failures.append(msg)


def make_tiny_config(**overrides):
    """A minimal config that exercises the pipeline without long runtimes."""
    defaults = dict(
        model_checkpoint="test-tiny-smoke",
        device="generic_aarch64",
        engine_gdn="cpu",
        engine_full_attention="cpu",
        quantization="fp16",
        context_lengths=[4096],
        warmups=1,
        repeats=5,
        decode_tokens=8,
    )
    defaults.update(overrides)
    return HarnessConfig(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEndToEndSweep:
    """Full pipeline: sweep → rows → validate → CSV → read-back → summary."""

    def test_all_rows_validate(self):
        """Every row produced by the sweep must pass schema validation."""
        config = make_tiny_config()
        backend = SyntheticBackend()
        rows = run_sweep(backend, config, progress=False)
        check(len(rows) > 0, "sweep produced zero rows")
        # 3 metrics × 5 repeats = 15 rows for one context length
        check(len(rows) == 15, f"expected 15 rows (3 metrics × 5 repeats), got {len(rows)}")
        for i, row in enumerate(rows):
            try:
                validate_row(row)
            except SchemaValidationError as exc:
                check(False, f"row {i} failed validation: {exc}")
        check(True, "all rows validated")  # only reached if no assertion fired

    def test_csv_roundtrip_preserves_values(self):
        """Write CSV then read it back; values must match within float tolerance."""
        config = make_tiny_config()
        backend = SyntheticBackend()
        rows = run_sweep(backend, config, progress=False)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            csv_path = f.name
        try:
            write_csv(rows, csv_path)
            read_back = read_csv(csv_path)
            check(len(read_back) == len(rows),
                  f"row count mismatch: wrote {len(rows)}, read {len(read_back)}")
            for orig, reread in zip(rows, read_back):
                check(orig.metric_name == reread.metric_name,
                      f"metric_name mismatch: {orig.metric_name} vs {reread.metric_name}")
                check(orig.context_length == reread.context_length,
                      f"context_length mismatch: {orig.context_length} vs {reread.context_length}")
                check(abs(orig.value - reread.value) < 1e-9,
                      f"value mismatch: {orig.value} vs {reread.value}")
        finally:
            os.unlink(csv_path)

    def test_summary_has_all_metric_groups(self):
        """summarize() output must include all three metric types."""
        config = make_tiny_config()
        backend = SyntheticBackend()
        rows = run_sweep(backend, config, progress=False)
        text = summarize(rows)
        check("prefill_tokens_per_sec" in text, "summary missing prefill_tokens_per_sec")
        check("ttft_seconds" in text, "summary missing ttft_seconds")
        check("decode_tokens_per_sec" in text, "summary missing decode_tokens_per_sec")

    def test_run_id_format(self):
        """Run ID must follow device_timestamp_sha pattern."""
        import re
        config = make_tiny_config()
        # Format: <device>_<YYYYmmDDTHHMMSSZ>_<shortsha>
        # Device may contain underscores (e.g. generic_aarch64)
        m = re.match(
            r"^(.+)_(\d{8}T\d{6}Z)_([0-9a-f]{7,})$", config.run_id
        )
        check(m is not None, f"run_id doesn't match pattern: {config.run_id}")
        if m:
            check(m.group(1) == "generic_aarch64",
                  f"run_id device prefix wrong: {m.group(1)}")
            check(len(m.group(3)) >= 7, f"sha too short: {m.group(3)}")

    def test_manifest_ref_points_to_results(self):
        """manifest_ref must be under results/manifests/."""
        config = make_tiny_config()
        check(config.manifest_ref.startswith("results/manifests/"),
              f"manifest_ref not under results/manifests/: {config.manifest_ref}")
        check(config.manifest_ref.endswith(".json"),
              f"manifest_ref not .json: {config.manifest_ref}")


class TestEndToEndWithManifest:
    """Full pipeline plus provenance manifest capture and write."""

    def test_manifest_capture_and_write(self):
        """Capture manifest, write to disk, read back, verify structure."""
        config = make_tiny_config()
        backend = SyntheticBackend()
        rows = run_sweep(backend, config, progress=False)

        # Capture provenance manifest
        manifest = capture(
            run_id=config.run_id,
            device=config.device,
            engine_gdn=config.engine_gdn,
            engine_full_attention=config.engine_full_attention,
            model_checkpoint=config.model_checkpoint,
            quantization=config.quantization,
            git_sha="test_sha_0000001",
            cli_args={"context_lengths": str(config.context_lengths)},
        )

        # Verify required manifest fields
        check("run_id" in manifest, "manifest missing run_id")
        check("timestamp_utc" in manifest, "manifest missing timestamp_utc")
        check("manifest_version" in manifest, "manifest missing manifest_version")
        check("host" in manifest, "manifest missing host section")
        check("software" in manifest, "manifest missing software section")
        check(manifest["run_id"] == config.run_id,
              f"manifest run_id mismatch: {manifest.get('run_id')}")

        # Write manifest to temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            manifest_path = f.name
        try:
            write(manifest, manifest_path)
            with open(manifest_path) as f:
                loaded = json.load(f)
            check(loaded["run_id"] == config.run_id, "loaded manifest run_id mismatch")
            check(loaded["caller"]["device"] == config.device, "loaded manifest device mismatch")
        finally:
            os.unlink(manifest_path)

    def test_manifest_links_to_csv_rows(self):
        """Every CSV row's manifest_ref must match the manifest path."""
        config = make_tiny_config()
        backend = SyntheticBackend()
        rows = run_sweep(backend, config, progress=False)
        for row in rows:
            check(row.manifest_ref == config.manifest_ref,
                  f"row manifest_ref {row.manifest_ref} != config {config.manifest_ref}")

    def test_full_pipeline_writes_both_csv_and_manifest(self):
        """Simulate the full CLI flow: sweep, validate, write CSV, capture+write manifest."""
        config = make_tiny_config()
        backend = SyntheticBackend()
        rows = run_sweep(backend, config, progress=False)

        # Validate all rows
        validate_rows(rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "run.csv")
            manifest_path = os.path.join(tmpdir, "manifest.json")

            # Write CSV
            write_csv(rows, csv_path)

            # Capture and write manifest
            manifest = capture(
                run_id=config.run_id,
                device=config.device,
                engine_gdn=config.engine_gdn,
                engine_full_attention=config.engine_full_attention,
                model_checkpoint=config.model_checkpoint,
                quantization=config.quantization,
                git_sha="test_sha_0000002",
                cli_args={"repeats": str(config.repeats)},
            )
            write(manifest, manifest_path)

            # Verify both files exist and are non-empty
            check(os.path.exists(csv_path), "CSV file not written")
            check(os.path.getsize(csv_path) > 0, "CSV file is empty")
            check(os.path.exists(manifest_path), "manifest file not written")
            check(os.path.getsize(manifest_path) > 0, "manifest file is empty")

            # Read back CSV and verify
            read_rows = read_csv(csv_path)
            check(len(read_rows) == len(rows), "CSV round-trip row count mismatch")

            # Read back manifest and verify
            with open(manifest_path) as f:
                loaded_manifest = json.load(f)
            check(loaded_manifest["run_id"] == config.run_id,
                  "manifest round-trip run_id mismatch")


class TestCLISmoke:
    """Smoke-test the CLI entrypoint (bench.harness main)."""

    def test_cli_exit_zero(self):
        """`python -m bench.harness --backend synthetic` must exit 0."""
        import subprocess
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            csv_path = f.name
        try:
            result = subprocess.run(
                [
                    sys.executable, "-m", "bench.harness",
                    "--backend", "synthetic",
                    "--model-checkpoint", "test-smoke",
                    "--context-lengths", "4096",
                    "--warmups", "1",
                    "--repeats", "5",
                    "--decode-tokens", "8",
                    "--device", "generic_aarch64",
                    "--engine-gdn", "cpu",
                    "--engine-full-attention", "cpu",
                    "--quantization", "fp16",
                    "--output", csv_path,
                    "--no-progress",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=os.path.join(os.path.dirname(__file__), ".."),
                env={**os.environ, "PYTHONPATH": "bench:."},
            )
            check(result.returncode == 0,
                  f"CLI exited {result.returncode}\nstderr: {result.stderr}")
            check(os.path.exists(csv_path), "CLI did not produce CSV output")
        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)

    def test_cli_csv_is_schema_conformant(self):
        """CSV produced by the CLI must validate on read-back."""
        import subprocess
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            csv_path = f.name
        try:
            subprocess.run(
                [
                    sys.executable, "-m", "bench.harness",
                    "--backend", "synthetic",
                    "--model-checkpoint", "test-smoke",
                    "--context-lengths", "4096",
                    "--warmups", "1",
                    "--repeats", "5",
                    "--device", "generic_aarch64",
                    "--engine-gdn", "cpu",
                    "--engine-full-attention", "cpu",
                    "--quantization", "fp16",
                    "--output", csv_path,
                    "--no-progress",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=os.path.join(os.path.dirname(__file__), ".."),
                env={**os.environ, "PYTHONPATH": "bench:."},
            )
            # read_csv validates by default
            rows = read_csv(csv_path)
            check(len(rows) == 15, f"expected 15 rows, got {len(rows)}")
        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)


class TestMultiContextSweep:
    """Verify the pipeline works with multiple context lengths."""

    def test_two_context_lengths(self):
        """Sweep with two context lengths produces rows for both."""
        config = make_tiny_config(context_lengths=[4096, 8192])
        backend = SyntheticBackend()
        rows = run_sweep(backend, config, progress=False)
        ctx_set = {r.context_length for r in rows}
        check(ctx_set == {4096, 8192}, f"context set mismatch: {ctx_set}")
        # 3 metrics × 5 repeats × 2 context lengths = 30 rows
        check(len(rows) == 30, f"expected 30 rows, got {len(rows)}")

    def test_summary_groups_by_context(self):
        """summarize() must show separate groups per context length."""
        config = make_tiny_config(context_lengths=[4096, 8192])
        backend = SyntheticBackend()
        rows = run_sweep(backend, config, progress=False)
        text = summarize(rows)
        # Both context lengths should appear in the summary
        lines_with_4096 = [l for l in text.split("\n") if "4096" in l]
        lines_with_8192 = [l for l in text.split("\n") if "8192" in l]
        check(len(lines_with_4096) > 0, "summary missing 4096 rows")
        check(len(lines_with_8192) > 0, "summary missing 8192 rows")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _run_all():
    tests = [
        TestEndToEndSweep(),
        TestEndToEndWithManifest(),
        TestCLISmoke(),
        TestMultiContextSweep(),
    ]
    for suite in tests:
        for name in sorted(dir(suite)):
            if name.startswith("test_"):
                getattr(suite, name)()


if __name__ == "__main__":
    _run_all()
    if _failures:
        print(f"\n✗ {len(_failures)} failures out of {_passes + len(_failures)} checks:")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print(f"\n✓ All {_passes} checks passed.")
