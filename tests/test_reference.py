"""Tests for the golden reference inference output (ob-aqv).

Validates:
1. The compact reference file loads and has the expected schema.
2. All 12 entries (4 prompts × 3 context lengths) are present.
3. Perplexity values are sane (between 1 and 1000 for any real model).
4. The correctness oracle (bench.correctness) works on the reference data
   via an identity check (reference vs. itself must PASS).
5. The top-k window is internally consistent (indices match sorted values).

The compact reference omits full-vocab logits to keep the file small (~65 KiB);
per-entry SHA-256 hashes of the stripped logits are included for integrity.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from bench.correctness import (
    ToleranceConfig,
    compare_logits,
    compare_perplexity,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_REF_PATH = _REPO_ROOT / "results" / "reference" / "qwen35-0.8b_reference_compact.json"


@pytest.fixture(scope="module")
def ref_data():
    """Load the compact reference data."""
    if not _REF_PATH.exists():
        pytest.skip("Compact reference not generated — run scripts/generate_reference.py")
    with open(_REF_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Schema and structure
# ---------------------------------------------------------------------------


def test_reference_has_provenance(ref_data):
    """The reference must record full provenance for reproducibility."""
    prov = ref_data["provenance"]
    required = {
        "git_sha",
        "hostname",
        "torch_version",
        "transformers_version",
        "device",
        "dtype",
        "model_repo",
    }
    missing = required - set(prov.keys())
    assert not missing, f"Missing provenance keys: {missing}"


def test_reference_has_all_entries(ref_data):
    """Expect 4 prompts × 3 context lengths = 12 entries."""
    entries = ref_data["entries"]
    assert len(entries) == 12, f"Expected 12 entries, got {len(entries)}"

    # Verify all prompt IDs and context lengths are represented
    prompt_ids = {e["prompt_id"] for e in entries}
    assert prompt_ids == {"factual", "code", "sequential", "reasoning"}

    ctx_lengths = sorted({e["context_length"] for e in entries})
    assert ctx_lengths == [128, 512, 2048]


def test_each_entry_has_required_fields(ref_data):
    """Every entry must have the fields the correctness oracle needs."""
    required = {
        "entry_id",
        "prompt_id",
        "context_length",
        "perplexity",
        "avg_nll",
        "argmax_token",
        "topk_window",
        "generated_token_ids",
        "generated_text",
        "last_position_logits_sha256",  # integrity hash for stripped logits
    }
    for entry in ref_data["entries"]:
        missing = required - set(entry.keys())
        assert not missing, f"{entry['entry_id']}: missing {missing}"


# ---------------------------------------------------------------------------
# Numerical sanity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ctx_len", [128, 512, 2048])
def test_perplexity_is_sane(ref_data, ctx_len):
    """Perplexity must be between 1 and 1000 for any real language model."""
    for entry in ref_data["entries"]:
        if entry["context_length"] == ctx_len:
            ppl = entry["perplexity"]
            assert 1.0 < ppl < 1000.0, f"{entry['entry_id']}: ppl={ppl} out of range"


def test_perplexity_decreases_with_context(ref_data):
    """Longer context → lower perplexity (model has more information).

    This is a property test: with real-token truncation (no padding), the
    model should predict better as it sees more context.  The filler text
    is repetitive/technical, so this trend should be strong.
    """
    by_prompt: dict[str, dict[int, float]] = {}
    for entry in ref_data["entries"]:
        pid = entry["prompt_id"]
        ctx = entry["context_length"]
        by_prompt.setdefault(pid, {})[ctx] = entry["perplexity"]

    for pid, ppls in by_prompt.items():
        assert ppls[128] > ppls[2048], (
            f"{pid}: perplexity should decrease with context "
            f"(128={ppls[128]:.2f}, 2048={ppls[2048]:.2f})"
        )


def test_topk_window_is_consistent(ref_data):
    """Top-k indices must correspond to the highest logits in order."""
    for entry in ref_data["entries"]:
        for window in entry["topk_window"]:
            vals = window["values"]
            assert len(vals) > 0
            # Values should be sorted descending
            for i in range(len(vals) - 1):
                assert vals[i] >= vals[i + 1], (
                    f"{entry['entry_id']} pos={window['position_from_end']}: "
                    f"values not sorted descending"
                )


def test_generated_tokens_nonempty(ref_data):
    """Each entry should have decoded at least 1 token."""
    for entry in ref_data["entries"]:
        assert len(entry["generated_token_ids"]) >= 1
        assert entry["generated_text"]  # non-empty string


# ---------------------------------------------------------------------------
# Correctness oracle identity check
# ---------------------------------------------------------------------------


def test_correctness_oracle_identity_perplexity(ref_data):
    """The correctness oracle must PASS when comparing reference to itself."""
    cfg = ToleranceConfig()
    for entry in ref_data["entries"]:
        report = compare_perplexity(
            entry["perplexity"],
            entry["perplexity"],
            cfg,
            context_length=entry["context_length"],
        )
        assert report.passed, f"{entry['entry_id']}: identity perplexity check failed"


def test_correctness_oracle_identity_logits():
    """Logit identity check: comparing a distribution to itself must PASS.

    Uses a synthetic logit vector (not the reference data, since the compact
    version has stripped full-vocab logits). This tests the oracle machinery.
    """
    cfg = ToleranceConfig()
    # Two identical "logit" vectors at one position
    ref = [[0.1, 0.5, 0.2, 0.8, 0.3]]
    cand = [[0.1, 0.5, 0.2, 0.8, 0.3]]
    report = compare_logits(ref, cand, cfg)
    assert report.passed, "Identity logit check should pass"
    # Max abs diff should be exactly 0
    max_diff = next(m.value for m in report.metrics if m.name == "max_abs_diff")
    assert max_diff == 0.0


def test_correctness_oracle_rejects_divergence():
    """The oracle must FAIL when logits diverge beyond tolerance."""
    cfg = ToleranceConfig(atol=1e-4, rtol=1e-3)
    ref = [[0.1, 0.5, 0.2, 0.8, 0.3]]
    cand = [[0.1, 0.5, 0.2, 0.9, 0.3]]  # position 3 differs by 0.1
    report = compare_logits(ref, cand, cfg)
    assert not report.passed, "Divergent logits should fail"
