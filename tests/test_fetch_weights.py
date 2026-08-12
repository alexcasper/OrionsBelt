# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/fetch_weights.py weight-fetch tooling.

Tests cover pure logic (plan computation, display formatting, manifest
serialization) without making any network calls.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure scripts/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import fetch_weights  # noqa: E402
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


# ---------------------------------------------------------------------------
# Pure utility functions (_resolve_weight_files, _file_is_present, _sha256)
# ---------------------------------------------------------------------------


class TestResolveWeightFiles:
    def test_filters_safetensors(self):
        from scripts.fetch_weights import _resolve_weight_files

        repo_files = [
            "config.json",
            "model-00001-of-00003.safetensors",
            "tokenizer.json",
            "model-00002-of-00003.safetensors",
            "tokenizer_config.json",
            "model.safetensors.index.json",
        ]
        result = _resolve_weight_files("Qwen/Qwen3.5-4B", repo_files)
        assert len(result) == 2
        assert all(f.endswith(".safetensors") for f in result)

    def test_empty_repo(self):
        from scripts.fetch_weights import _resolve_weight_files

        assert _resolve_weight_files("test/repo", []) == []

    def test_no_weight_files(self):
        from scripts.fetch_weights import _resolve_weight_files

        result = _resolve_weight_files("test/repo", ["config.json", "README.md"])
        assert result == []

    def test_single_shard(self):
        from scripts.fetch_weights import _resolve_weight_files

        result = _resolve_weight_files("test/repo", ["model.safetensors"])
        assert result == ["model.safetensors"]


class TestFileIsPresent:
    def test_existing_nonempty_file(self, tmp_path):
        from scripts.fetch_weights import _file_is_present

        f = tmp_path / "test.bin"
        f.write_bytes(b"hello")
        assert _file_is_present(f) is True

    def test_nonexistent_file(self, tmp_path):
        from scripts.fetch_weights import _file_is_present

        assert _file_is_present(tmp_path / "nope.bin") is False

    def test_empty_file(self, tmp_path):
        from scripts.fetch_weights import _file_is_present

        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        assert _file_is_present(f) is False


class TestSha256:
    def test_known_hash(self, tmp_path):
        from scripts.fetch_weights import _sha256

        f = tmp_path / "test.bin"
        f.write_bytes(b"hello world")
        # Known SHA-256 of "hello world"
        expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        assert _sha256(f) == expected

    def test_empty_file(self, tmp_path):
        from scripts.fetch_weights import _sha256

        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        # SHA-256 of empty string
        expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        assert _sha256(f) == expected

    def test_large_file_chunked(self, tmp_path):
        """File larger than chunk size is read in multiple chunks."""
        from scripts.fetch_weights import _sha256

        f = tmp_path / "large.bin"
        data = b"x" * (1 << 20) * 3  # 3 MiB
        f.write_bytes(data)

        import hashlib

        expected = hashlib.sha256(data).hexdigest()
        # Use small chunk to force multiple reads
        assert _sha256(f, chunk=1024) == expected

    def test_returns_hex_string(self, tmp_path):
        from scripts.fetch_weights import _sha256

        f = tmp_path / "test.bin"
        f.write_bytes(b"data")
        result = _sha256(f)
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)


class TestMainListFlag:
    def test_list_flag_prints_models(self, capsys):
        from scripts.fetch_weights import main

        rc = main(["--list"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "Available models" in captured.out
        assert "4B" in captured.out
        assert "0.8B" in captured.out

    def test_list_flag_with_model_ignored(self, capsys):
        """--list takes priority over --model."""
        from scripts.fetch_weights import main

        rc = main(["--list", "--model", "0.8B"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "Available models" in captured.out


# ---------------------------------------------------------------------------
# _list_repo_files (both huggingface_hub and urllib fallback paths)
# ---------------------------------------------------------------------------


class TestListRepoFiles:
    """Test _list_repo_files with mocked network calls."""

    def test_urllib_fallback_success(self):
        """When huggingface_hub is not available, use urllib to list files."""

        # Simulate ImportError for huggingface_hub
        with (
            patch.dict(sys.modules, {"huggingface_hub": None}),
        ):
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(
                {
                    "siblings": [
                        {"rfilename": "config.json"},
                        {"rfilename": "model.safetensors"},
                    ]
                }
            ).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)

            with patch("urllib.request.urlopen", return_value=mock_resp):
                result = fetch_weights._list_repo_files("Qwen/Qwen3.5-4B")

        assert result == ["config.json", "model.safetensors"]

    def test_urllib_fallback_network_error_raises(self):
        """urllib fallback should re-raise network errors."""

        with (
            patch.dict(sys.modules, {"huggingface_hub": None}),
            patch.object(
                urllib.request,
                "urlopen",
                side_effect=urllib.error.URLError("timeout"),
            ),
            pytest.raises(urllib.error.URLError),
        ):
            fetch_weights._list_repo_files("Qwen/Qwen3.5-4B")

    def test_hub_api_success(self):
        """When huggingface_hub is available, use list_repo_files directly."""
        fake_hub = MagicMock()
        fake_hub.list_repo_files.return_value = ["model.safetensors", "config.json"]
        with patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
            result = fetch_weights._list_repo_files("Qwen/Qwen3.5-4B")
        assert result == ["config.json", "model.safetensors"]


# ---------------------------------------------------------------------------
# _download_file (urllib fallback path)
# ---------------------------------------------------------------------------


class TestDownloadFile:
    """Test _download_file with mocked network calls."""

    def test_urllib_fallback_writes_file(self, tmp_path):
        """When huggingface_hub is unavailable, download via urllib."""

        dest = tmp_path / "weights.safetensors"

        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        file_content = b"\x00" * 1024

        with (
            patch.dict(sys.modules, {"huggingface_hub": None}),
            patch("urllib.request.urlopen", return_value=mock_resp),
            patch("shutil.copyfileobj") as mock_copy,
        ):

            def fake_copy(src, dst):
                dst.write(file_content)

            mock_copy.side_effect = fake_copy
            result = fetch_weights._download_file(
                "Qwen/Qwen3.5-4B",
                "weights.safetensors",
                dest,
                timeout=10,
            )

        assert result == 1024
        assert dest.exists()
        assert dest.stat().st_size == 1024

    def test_urllib_fallback_download_failure(self, tmp_path):
        """urllib download failure should raise."""

        dest = tmp_path / "bad.safetensors"

        with (
            patch.dict(sys.modules, {"huggingface_hub": None}),
            patch.object(
                urllib.request,
                "urlopen",
                side_effect=urllib.error.URLError("connection refused"),
            ),
            pytest.raises(urllib.error.URLError),
        ):
            fetch_weights._download_file(
                "Qwen/Qwen3.5-4B",
                "bad.safetensors",
                dest,
            )

    def test_hub_api_download_success(self, tmp_path):
        """When huggingface_hub is available, use hf_hub_download."""
        dest = tmp_path / "weights.safetensors"
        dest.write_bytes(b"\x00" * 2048)
        fake_hub = MagicMock()
        fake_hub.hf_hub_download.return_value = str(dest)
        with patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
            result = fetch_weights._download_file(
                "Qwen/Qwen3.5-4B",
                "weights.safetensors",
                dest,
            )
        assert result == 2048


# ---------------------------------------------------------------------------
# fetch_model — actual download loop (mock _list_repo_files + _download_file)
# ---------------------------------------------------------------------------


class TestFetchModelDownload:
    """Test fetch_model with mocked file discovery and download."""

    def test_download_success(self, tmp_path):
        """fetch_model downloads all planned files and writes manifest."""

        model = MODELS["4B"]
        files_to_serve = sorted(REPO_FILES_4B)

        def fake_list(repo_id):
            return files_to_serve

        def fake_download(repo_id, filename, dest, **kw):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"\x00" * 2048)
            return 2048

        with (
            patch("fetch_weights._list_repo_files", side_effect=fake_list),
            patch("fetch_weights._download_file", side_effect=fake_download),
        ):
            manifest = fetch_weights.fetch_model(model, tmp_path)

        assert manifest.model_name == "Qwen3.5-4B"
        assert manifest.repo_id == "Qwen/Qwen3.5-4B"
        assert manifest.fetched_at  # timestamp written
        assert len(manifest.files) > 0
        assert all(f["success"] for f in manifest.files)

        # Manifest JSON written to disk
        manifest_path = tmp_path / "Qwen3.5-4B" / "manifest.json"
        assert manifest_path.exists()
        data = json.loads(manifest_path.read_text())
        assert data["model_name"] == "Qwen3.5-4B"

    def test_existing_files_skipped(self, tmp_path):
        """Files already present are skipped, not re-downloaded."""

        model = MODELS["4B"]
        files_to_serve = sorted(REPO_FILES_4B)

        # Pre-create one file so it's "already present"
        model_dir = tmp_path / "Qwen3.5-4B"
        model_dir.mkdir(parents=True)
        existing_file = model_dir / "config.json"
        existing_file.write_bytes(b"\x00" * 512)

        download_count = 0

        def fake_list(repo_id):
            return files_to_serve

        def fake_download(repo_id, filename, dest, **kw):
            nonlocal download_count
            download_count += 1
            dest.write_bytes(b"\x00" * 100)
            return 100

        with (
            patch("fetch_weights._list_repo_files", side_effect=fake_list),
            patch("fetch_weights._download_file", side_effect=fake_download),
        ):
            manifest = fetch_weights.fetch_model(model, tmp_path)

        # The config.json should be marked as skipped
        config_record = next(f for f in manifest.files if f["filename"] == "config.json")
        assert config_record["skipped"] is True
        assert config_record["bytes"] == 512

        # Other files should have been downloaded
        assert download_count > 0

    def test_download_failure_recorded(self, tmp_path):
        """Download failures are recorded in manifest, other files still proceed."""

        model = MODELS["4B"]
        files_to_serve = sorted(REPO_FILES_4B)

        def fake_list(repo_id):
            return files_to_serve

        def fake_download(repo_id, filename, dest, **kw):
            if "safetensors" in filename:
                raise RuntimeError("disk full")
            dest.write_bytes(b"\x00" * 100)
            return 100

        with (
            patch("fetch_weights._list_repo_files", side_effect=fake_list),
            patch("fetch_weights._download_file", side_effect=fake_download),
        ):
            manifest = fetch_weights.fetch_model(model, tmp_path)

        failed = [f for f in manifest.files if not f["success"]]
        succeeded = [f for f in manifest.files if f["success"]]
        assert len(failed) > 0
        assert len(succeeded) > 0
        assert all("error" in f for f in failed)

    def test_repo_list_error(self, tmp_path):
        """If _list_repo_files raises, manifest records the error."""

        model = MODELS["4B"]

        with patch("fetch_weights._list_repo_files", side_effect=RuntimeError("404")):
            manifest = fetch_weights.fetch_model(model, tmp_path)

        assert len(manifest.files) == 1
        assert manifest.files[0]["filename"] == "<repo-list>"
        assert manifest.files[0]["success"] is False
        assert "404" in manifest.files[0]["error"]


# ---------------------------------------------------------------------------
# main() failure path
# ---------------------------------------------------------------------------


class TestMainFailurePath:
    """Test main() when some downloads fail."""

    def test_returns_0_on_all_success(self, tmp_path, mock_repo_files):
        """main() returns 0 when all downloads succeed."""

        def fake_download(repo_id, filename, dest, **kw):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"\x00" * 100)
            return 100

        with patch("fetch_weights._download_file", side_effect=fake_download):
            rc = main(["--model", "4B", "--output-dir", str(tmp_path)])

        assert rc == 0

    def test_returns_1_when_some_fail(self, tmp_path, mock_repo_files):
        """main() returns 1 when some downloads fail."""

        def fake_download(repo_id, filename, dest, **kw):
            if "safetensors" in filename:
                raise RuntimeError("network error")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"\x00" * 100)
            return 100

        with patch("fetch_weights._download_file", side_effect=fake_download):
            rc = main(["--model", "4B", "--output-dir", str(tmp_path)])

        assert rc == 1

    def test_all_models_some_fail(self, tmp_path, mock_repo_files):
        """main() with --model all returns 1 if any model has failures."""

        def fake_download(repo_id, filename, dest, **kw):
            raise RuntimeError("network error")

        with patch("fetch_weights._download_file", side_effect=fake_download):
            rc = main(["--model", "all", "--output-dir", str(tmp_path)])

        assert rc == 1


# ---------------------------------------------------------------------------
# _human_size overflow
# ---------------------------------------------------------------------------


class TestHumanSizeOverflow:
    def test_pib_overflow(self):
        """Very large numbers should show PiB."""
        result = _human_size(1024**5)
        assert "PiB" in result
