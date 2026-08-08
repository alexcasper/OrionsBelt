"""End-to-end smoke tests: sweep -> CSV -> manifest linkage (bead ob-1lm).

Ported from bench/j1's test_e2e_smoke.py, which was written against that branch's
``HarnessConfig`` API and could not import against the harness on main. Rather
than drop it or rewrite all 382 lines, this keeps the assertions main did not
already have — run_id shape, the manifest_ref actually resolving to a written
file, and a multi-context sweep staying separable per context. The rest of the
original (CSV round-trip, schema conformance, CLI exit status) is already covered
by tests/test_integration.py and tests/test_harness.py.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bench.harness import (  # noqa: E402
    QWEN35_4B,
    SweepConfig,
    SyntheticBackend,
    run_sweep,
)
from bench.schema import read_csv, validate_rows, write_csv  # noqa: E402


def _sweep(tmp_path, context_lengths=(64,)):
    cfg = SweepConfig(
        context_lengths=list(context_lengths),
        warmup_count=1,
        repeat_count=5,
        decode_length=6,
        manifest_dir=str(tmp_path / "manifests"),
    )
    return run_sweep(SyntheticBackend(QWEN35_4B), cfg), cfg


class TestRunIdAndManifestLinkage:
    def test_run_id_shape(self, tmp_path):
        """run_id is <device>_<yyyymmddTHHMMSSZ>_<short_sha> (manifest.py convention)."""
        rows, _ = _sweep(tmp_path)
        assert rows
        pattern = r"^[a-z0-9_]+_\d{8}T\d{6}Z_[0-9a-f]{7,40}$"
        assert re.match(pattern, rows[0].run_id), rows[0].run_id

    def test_run_id_identical_across_every_row(self, tmp_path):
        """One sweep is one run — a per-row run_id would break grouping."""
        rows, _ = _sweep(tmp_path, context_lengths=(64, 128))
        assert len({r.run_id for r in rows}) == 1

    def test_manifest_ref_points_under_results(self, tmp_path):
        rows, _ = _sweep(tmp_path)
        assert all(r.manifest_ref for r in rows)
        assert all(r.manifest_ref.endswith(".json") for r in rows)

    def test_csv_roundtrip_keeps_manifest_ref(self, tmp_path):
        """The CSV must carry provenance through a write/read cycle."""
        rows, _ = _sweep(tmp_path)
        out = tmp_path / "out.csv"
        write_csv(rows, str(out))
        back = read_csv(str(out))
        validate_rows(back)
        assert {r.manifest_ref for r in back} == {r.manifest_ref for r in rows}
        assert {r.git_sha for r in back} == {r.git_sha for r in rows}


class TestMultiContextSweep:
    def test_each_context_present(self, tmp_path):
        rows, _ = _sweep(tmp_path, context_lengths=(64, 128))
        assert {r.context_length for r in rows} == {64, 128}

    def test_row_count_scales_with_contexts(self, tmp_path):
        one, _ = _sweep(tmp_path, context_lengths=(64,))
        two, _ = _sweep(tmp_path, context_lengths=(64, 128))
        assert len(two) == 2 * len(one)

    def test_contexts_are_independently_separable(self, tmp_path):
        """A truncated sweep must still yield usable data (docs/archive/PLAN.md section 5/R4)."""
        rows, _ = _sweep(tmp_path, context_lengths=(64, 128))
        for ctx in (64, 128):
            subset = [r for r in rows if r.context_length == ctx]
            assert subset
            validate_rows(subset)
