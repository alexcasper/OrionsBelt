"""Tests for scripts/fetch_weights.py weight-fetch tooling.

Tests cover pure logic (plan computation, display formatting, manifest
serialization) without making any network calls.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure scripts/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from fetch_weights import (  # noqa: E402
    MODELS,
    DownloadRecord,
    FetchManifest,
    _human_size,
    list_models,
    main,
    plan_download,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

REPO_FILES_4B = [
    ".gitattributes",
    "LICENSE",
    "README.md",
    "chat_template.jinja",
    "config.json",
    "merges.txt",
    "model.safetensors-00001-of-00002.safetensors",
    "model.safetensors-00002-of-00002.safetensors",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
]

REPO_FILES_0_8B = [
    ".gitattributes",
    "LICENSE",
    "README.md",
    "chat_template.jinja",
    "config.json",
    "merges.txt",
    "model.safetensors-00001-of-00001.safetensors",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
]


@pytest.fixture
def mock_repo_files():
    """Patch _list_repo_files so dry-run tests don't hit the live HF API.

    Without this, tests that exercise ``main(["--dry-run", ...])`` call the real
    HuggingFace Hub REST endpoint, which intermittently fails with HTTP 429
    under CI's shared IP (bead ob-fty).
    """

    def _mock(repo_id, *args, **kwargs):
        if "4B" in repo_id:
            return sorted(REPO_FILES_4B)
        elif "0.8B" in repo_id:
            return sorted(REPO_FILES_0_8B)
        return []

    with patch("fetch_weights._list_repo_files", side_effect=_mock):
        yield


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


class TestModelRegistry:
    def test_has_two_models(self):
        assert set(MODELS) == {"4B", "0.8B"}

    def test_4b_info(self):
        m = MODELS["4B"]
        assert m.name == "Qwen3.5-4B"
        assert m.repo_id == "Qwen/Qwen3.5-4B"
        assert m.license == "Apache-2.0"

    def test_0_8b_info(self):
        m = MODELS["0.8B"]
        assert m.name == "Qwen3.5-0.8B"
        assert m.repo_id == "Qwen/Qwen3.5-0.8B"
        assert m.license == "Apache-2.0"

    def test_both_apache2(self):
        for m in MODELS.values():
            assert m.license == "Apache-2.0"

    def test_huggingface_urls(self):
        for m in MODELS.values():
            assert m.huggingface_url.startswith("https://huggingface.co/Qwen/")

    def test_model_info_is_frozen(self):
        m = MODELS["4B"]
        with pytest.raises(AttributeError):
            m.name = "other"  # type: ignore[misc]

    def test_skip_files_present(self):
        for m in MODELS.values():
            assert "video_preprocessor_config.json" in m.skip_files


# ---------------------------------------------------------------------------
# plan_download
# ---------------------------------------------------------------------------


class TestPlanDownload:
    def test_full_download_4b(self):
        m = MODELS["4B"]
        files = plan_download(m, REPO_FILES_4B, metadata_only=False)
        assert "config.json" in files
        assert "tokenizer.json" in files
        assert "LICENSE" in files
        assert "model.safetensors-00001-of-00002.safetensors" in files
        assert "model.safetensors-00002-of-00002.safetensors" in files

    def test_metadata_only_4b(self):
        m = MODELS["4B"]
        files = plan_download(m, REPO_FILES_4B, metadata_only=True)
        assert "config.json" in files
        assert "tokenizer.json" in files
        assert "model.safetensors.index.json" in files
        # No weight shards
        assert not any("safetensors-" in f for f in files)
        assert not any(
            f.endswith(".safetensors") and "-" not in f
            for f in files
            if f != "model.safetensors.index.json"
        )

    def test_video_preprocessor_excluded(self):
        m = MODELS["4B"]
        files = plan_download(m, REPO_FILES_4B, metadata_only=False)
        assert "video_preprocessor_config.json" not in files

    def test_gitattributes_excluded(self):
        m = MODELS["4B"]
        files = plan_download(m, REPO_FILES_4B, metadata_only=False)
        assert ".gitattributes" not in files

    def test_readme_excluded(self):
        m = MODELS["4B"]
        files = plan_download(m, REPO_FILES_4B, metadata_only=False)
        assert "README.md" not in files

    def test_single_shard_0_8b(self):
        m = MODELS["0.8B"]
        files = plan_download(m, REPO_FILES_0_8B, metadata_only=False)
        assert "model.safetensors-00001-of-00001.safetensors" in files
        assert len([f for f in files if f.endswith(".safetensors")]) == 1

    def test_license_always_included(self):
        for key, m in MODELS.items():
            repo_files = REPO_FILES_4B if key == "4B" else REPO_FILES_0_8B
            files = plan_download(m, repo_files, metadata_only=True)
            assert "LICENSE" in files

    def test_no_duplicate_files(self):
        m = MODELS["4B"]
        files = plan_download(m, REPO_FILES_4B, metadata_only=False)
        assert len(files) == len(set(files))

    def test_empty_repo_files(self):
        m = MODELS["4B"]
        files = plan_download(m, [], metadata_only=False)
        assert files == []

    def test_partial_repo_files(self):
        """If only some metadata files exist, download those."""
        m = MODELS["4B"]
        partial = ["config.json", "LICENSE", "model.safetensors"]
        files = plan_download(m, partial, metadata_only=False)
        assert "config.json" in files
        assert "LICENSE" in files
        assert "model.safetensors" in files
        # tokenizer not in repo, should not appear
        assert "tokenizer.json" not in files

    def test_weight_files_distinguished_from_index(self):
        """model.safetensors.index.json is metadata, not a weight file."""
        m = MODELS["4B"]
        files_meta = plan_download(m, REPO_FILES_4B, metadata_only=True)
        assert "model.safetensors.index.json" in files_meta
        assert "model.safetensors-00001-of-00002.safetensors" not in files_meta


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


class TestHumanSize:
    def test_bytes(self):
        assert _human_size(0) == "0.0 B"
        assert _human_size(512) == "512.0 B"

    def test_kib(self):
        assert _human_size(1024) == "1.0 KiB"
        assert _human_size(1536) == "1.5 KiB"

    def test_mib(self):
        assert _human_size(1024 * 1024) == "1.0 MiB"
        assert _human_size(50 * 1024 * 1024) == "50.0 MiB"

    def test_gib(self):
        assert _human_size(8 * 1024 * 1024 * 1024) == "8.0 GiB"

    def test_large_gib(self):
        val = 32.8 * 1024 * 1024 * 1024
        result = _human_size(int(val))
        assert "GiB" in result


class TestListModels:
    def test_contains_both_models(self):
        text = list_models()
        assert "Qwen3.5-4B" in text
        assert "Qwen3.5-0.8B" in text

    def test_contains_license_info(self):
        text = list_models()
        assert "Apache-2.0" in text

    def test_contains_urls(self):
        text = list_models()
        assert "huggingface.co" in text


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class TestDownloadRecord:
    def test_defaults(self):
        r = DownloadRecord(filename="config.json", success=True)
        assert r.bytes == 0
        assert r.skipped is False
        assert r.error == ""

    def test_failure(self):
        r = DownloadRecord(filename="weights", success=False, error="timeout")
        assert r.success is False
        assert r.error == "timeout"


class TestFetchManifest:
    def test_defaults(self):
        m = FetchManifest(model_name="Qwen3.5-4B", repo_id="Qwen/Qwen3.5-4B")
        assert m.revision == "main"
        assert m.files == []
        assert m.fetched_at == ""

    def test_serializable(self):
        m = FetchManifest(model_name="Qwen3.5-4B", repo_id="Qwen/Qwen3.5-4B")
        m.files.append({"filename": "config.json", "success": True})
        from dataclasses import asdict

        j = json.dumps(asdict(m))
        parsed = json.loads(j)
        assert parsed["model_name"] == "Qwen3.5-4B"
        assert parsed["files"][0]["filename"] == "config.json"


# ---------------------------------------------------------------------------
# Dry-run (no network)
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_dry_run_4b(self, capsys, tmp_path, mock_repo_files):
        """Dry run should not create any files."""
        rc = main(["--model", "4B", "--dry-run", "--output-dir", str(tmp_path)])
        assert rc == 0
        # No model directory should exist
        assert not (tmp_path / "Qwen3.5-4B").exists()

    def test_dry_run_metadata_only(self, capsys, tmp_path, mock_repo_files):
        rc = main(["--model", "0.8B", "--dry-run", "--output-dir", str(tmp_path)])
        assert rc == 0
        assert not (tmp_path / "Qwen3.5-0.8B").exists()

    def test_dry_run_all(self, capsys, tmp_path, mock_repo_files):
        rc = main(["--model", "all", "--dry-run", "--output-dir", str(tmp_path)])
        assert rc == 0
        assert not (tmp_path / "Qwen3.5-4B").exists()
        assert not (tmp_path / "Qwen3.5-0.8B").exists()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCLI:
    def test_list(self, capsys):
        rc = main(["--list"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "Qwen3.5-4B" in captured.out
        assert "Qwen3.5-0.8B" in captured.out

    def test_invalid_model(self):
        with pytest.raises(SystemExit):
            main(["--model", "999B"])

    def test_default_model_is_4b(self, tmp_path, mock_repo_files):
        """--model defaults to 4B. Verify via dry-run stderr."""
        import io

        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            rc = main(["--dry-run", "--output-dir", str(tmp_path)])
            captured = sys.stderr.getvalue()
            assert rc == 0
            assert "Qwen3.5-4B" in captured
        finally:
            sys.stderr = old_stderr

    def test_output_dir_created_on_dry_run_not_needed(self, tmp_path, mock_repo_files):
        """Dry run doesn't need the output dir to exist."""
        nonexistent = tmp_path / "does_not_exist"
        rc = main(["--model", "4B", "--dry-run", "--output-dir", str(nonexistent)])
        assert rc == 0
