"""Tests for scripts/smoke_test_gdn2.py — helper functions.

Tests the portable logic of the GDN-2 smoke test (bead ob-y3f):
  - ``ensure_repo()`` — git clone / cache logic
  - ``check_imports()`` — dependency detection
  - ``main()`` argument parsing

The actual torch/CUDA kernel verification (``run_smoke_test``) is not tested
here as it requires heavy ML dependencies. These tests cover the setup and
guard logic that runs before any kernel work begins.
"""

import builtins
import os
import subprocess
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


# ---------------------------------------------------------------------------
# ensure_repo
# ---------------------------------------------------------------------------


class TestEnsureRepo:
    @patch("os.path.isdir", return_value=True)
    def test_cached_repo_returns_path(self, _mock_isdir):
        import smoke_test_gdn2

        result = smoke_test_gdn2.ensure_repo("/tmp/fake_clone")
        assert result == os.path.join("/tmp/fake_clone", "GatedDeltaNet-2")

    @patch("subprocess.run")
    @patch("os.path.isdir", return_value=False)
    def test_clones_when_not_cached(self, _mock_isdir, mock_run):
        import smoke_test_gdn2

        result = smoke_test_gdn2.ensure_repo("/tmp/fake_clone")
        assert result == os.path.join("/tmp/fake_clone", "GatedDeltaNet-2")
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "git"
        assert call_args[1] == "clone"
        assert "--depth" in call_args

    @patch("subprocess.run")
    @patch("os.path.isdir", return_value=False)
    def test_clone_uses_shallow_depth(self, _mock_isdir, mock_run):
        import smoke_test_gdn2

        smoke_test_gdn2.ensure_repo("/tmp/fake_clone")
        call_args = mock_run.call_args[0][0]
        depth_idx = call_args.index("--depth")
        assert call_args[depth_idx + 1] == "1"

    @patch("subprocess.run")
    @patch("os.path.isdir", return_value=False)
    def test_clone_uses_correct_repo_url(self, _mock_isdir, mock_run):
        import smoke_test_gdn2

        smoke_test_gdn2.ensure_repo("/tmp/fake_clone")
        call_args = mock_run.call_args[0][0]
        assert smoke_test_gdn2.REPO_URL in call_args

    @patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "git"))
    @patch("os.path.isdir", return_value=False)
    def test_clone_failure_propagates(self, _mock_isdir, _mock_run):
        import smoke_test_gdn2

        with pytest.raises(subprocess.CalledProcessError):
            smoke_test_gdn2.ensure_repo("/tmp/fake_clone")

    @patch("subprocess.run")
    @patch("os.path.isdir", return_value=True)
    def test_cached_does_not_call_subprocess(self, _mock_isdir, mock_run):
        import smoke_test_gdn2

        smoke_test_gdn2.ensure_repo("/cached/path")
        mock_run.assert_not_called()


# ---------------------------------------------------------------------------
# check_imports
# ---------------------------------------------------------------------------


class TestCheckImports:
    """check_imports() should sys.exit(1) when deps are missing."""

    def _make_import_blocker(self, blocked_names: set[str]):
        """Create a mock __import__ that blocks specific module names."""
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name in blocked_names:
                raise ImportError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)

        return fake_import

    def test_exits_on_missing_torch(self):
        import smoke_test_gdn2

        blocker = self._make_import_blocker({"torch"})
        with patch("builtins.__import__", side_effect=blocker):
            with pytest.raises(SystemExit) as exc_info:
                smoke_test_gdn2.check_imports()
            assert exc_info.value.code == 1

    def test_exits_on_missing_triton(self):
        import smoke_test_gdn2

        # torch import succeeds, triton fails
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "triton":
                raise ImportError("No module named 'triton'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with pytest.raises(SystemExit) as exc_info:
                smoke_test_gdn2.check_imports()
            assert exc_info.value.code == 1

    def test_exits_on_missing_fla(self):
        import smoke_test_gdn2

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "fla":
                raise ImportError("No module named 'fla'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with pytest.raises(SystemExit) as exc_info:
                smoke_test_gdn2.check_imports()
            assert exc_info.value.code == 1

    def test_missing_deps_all_reported(self, capsys):
        """When multiple deps are missing, all should be listed in output."""
        import smoke_test_gdn2

        blocker = self._make_import_blocker({"torch", "triton", "fla"})
        with patch("builtins.__import__", side_effect=blocker), pytest.raises(SystemExit):
            smoke_test_gdn2.check_imports()
        captured = capsys.readouterr()
        assert "torch" in captured.out
        assert "triton" in captured.out
        assert "flash-linear-attention" in captured.out


# ---------------------------------------------------------------------------
# main / argument parsing
# ---------------------------------------------------------------------------


class TestMainArgparse:
    """Test that main() parses arguments correctly."""

    @patch("smoke_test_gdn2.run_smoke_test", return_value=True)
    @patch("smoke_test_gdn2.ensure_repo", return_value="/fake/repo")
    @patch("smoke_test_gdn2.check_imports")
    @patch("sys.argv", ["smoke_test_gdn2.py", "--device", "cpu"])
    def test_main_device_cpu(self, _mock_check, _mock_ensure, _mock_run):
        import smoke_test_gdn2

        with pytest.raises(SystemExit) as exc_info:
            smoke_test_gdn2.main()
        assert exc_info.value.code == 0

    @patch("smoke_test_gdn2.run_smoke_test", return_value=True)
    @patch("smoke_test_gdn2.check_imports")
    @patch("sys.argv", ["smoke_test_gdn2.py", "--no-clone"])
    def test_no_clone_skips_ensure_repo(self, _mock_check, _mock_run):
        import smoke_test_gdn2

        with patch("smoke_test_gdn2.ensure_repo") as mock_ensure:
            with pytest.raises(SystemExit):
                smoke_test_gdn2.main()
            mock_ensure.assert_not_called()

    @patch("sys.argv", ["smoke_test_gdn2.py", "--device", "invalid"])
    def test_invalid_device_rejected(self):
        import smoke_test_gdn2

        with pytest.raises(SystemExit):
            smoke_test_gdn2.main()

    @patch("smoke_test_gdn2.run_smoke_test", return_value=True)
    @patch("smoke_test_gdn2.ensure_repo", return_value="/custom/path/GatedDeltaNet-2")
    @patch("smoke_test_gdn2.check_imports")
    @patch("sys.argv", ["smoke_test_gdn2.py", "--clone-dir", "/custom/path"])
    def test_clone_dir_passed_to_ensure_repo(self, _mock_check, mock_ensure, _mock_run):
        import smoke_test_gdn2

        with pytest.raises(SystemExit):
            smoke_test_gdn2.main()
        mock_ensure.assert_called_once_with("/custom/path")

    @patch("smoke_test_gdn2.run_smoke_test", return_value=False)
    @patch("smoke_test_gdn2.ensure_repo", return_value="/fake/repo")
    @patch("smoke_test_gdn2.check_imports")
    @patch("sys.argv", ["smoke_test_gdn2.py"])
    def test_main_exits_1_on_failure(self, _mock_check, _mock_ensure, _mock_run):
        import smoke_test_gdn2

        with pytest.raises(SystemExit) as exc_info:
            smoke_test_gdn2.main()
        assert exc_info.value.code == 1

    @patch("smoke_test_gdn2.run_smoke_test", return_value=True)
    @patch("smoke_test_gdn2.ensure_repo", return_value="/fake/repo")
    @patch("smoke_test_gdn2.check_imports")
    @patch("sys.argv", ["smoke_test_gdn2.py", "--no-clone"])
    def test_no_clone_does_not_add_to_sys_path(self, _mock_check, _mock_ensure, _mock_run):
        """--no-clone should skip adding repo to sys.path."""
        import smoke_test_gdn2

        with pytest.raises(SystemExit):
            smoke_test_gdn2.main()
        # sys.path should be unchanged (no ensure_repo, no sys.path.insert after it)


# ---------------------------------------------------------------------------
# Constants and module-level configuration
# ---------------------------------------------------------------------------


class TestModuleConstants:
    def test_repo_url_is_github(self):
        import smoke_test_gdn2

        assert "github.com" in smoke_test_gdn2.REPO_URL
        assert "GatedDeltaNet-2" in smoke_test_gdn2.REPO_URL

    def test_repo_url_is_nvlabs(self):
        import smoke_test_gdn2

        assert "NVlabs" in smoke_test_gdn2.REPO_URL or "nvabs" in smoke_test_gdn2.REPO_URL.lower()
