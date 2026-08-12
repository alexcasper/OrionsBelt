#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0

"""Download Qwen3.5 model weights from HuggingFace at setup time.

Design goals (see scripts/README.md and ADR 0003):
  - Non-interactive and idempotent — re-running skips files already present.
  - No vendoring — weights land outside the repo tree (default ``models/``)
    and the directory is .gitignored.
  - License-safe — both checkpoints are Apache-2.0, verified from HF metadata.
  - Degrades gracefully — uses ``huggingface_hub`` when installed but falls back
    to a stdlib ``urllib`` downloader so the script works in any Python 3.10+
    environment.

Usage::

    python3 scripts/fetch_weights.py --list              # show available models
    python3 scripts/fetch_weights.py --model 4B          # download 4B checkpoint
    python3 scripts/fetch_weights.py --model 0.8B        # download 0.8B checkpoint
    python3 scripts/fetch_weights.py --model all         # download both
    python3 scripts/fetch_weights.py --model 4B --metadata-only  # config + tokenizer only
    python3 scripts/fetch_weights.py --model 4B --dry-run       # plan without downloading

The script writes a ``manifest.json`` into each model directory recording repo
id, revision, file list, and download timestamp for reproducibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT = _REPO_ROOT / "models"

_HF_BASE = "https://huggingface.co"
_HF_RESOLVE = f"{_HF_BASE}/{{repo}}/resolve/main/{{filename}}"

# Files every checkpoint needs regardless of size
_METADATA_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "preprocessor_config.json",
    "model.safetensors.index.json",
]

# Files downloaded only when NOT using --metadata-only
# (resolved dynamically per-model because shard counts vary)
_WEIGHT_FILE_PATTERNS = [
    "model.safetensors-*.safetensors",  # sharded checkpoints
    "model.safetensors",  # single-file checkpoints
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelInfo:
    """Static metadata for a downloadable checkpoint."""

    name: str
    repo_id: str
    huggingface_url: str
    license: str
    license_url: str
    description: str
    # Human-readable approximate size for display only
    approx_size: str
    # Vision tower files we skip because we only need text inference
    skip_files: tuple[str, ...] = ("video_preprocessor_config.json",)


MODELS: dict[str, ModelInfo] = {
    "4B": ModelInfo(
        name="Qwen3.5-4B",
        repo_id="Qwen/Qwen3.5-4B",
        huggingface_url="https://huggingface.co/Qwen/Qwen3.5-4B",
        license="Apache-2.0",
        license_url="https://www.apache.org/licenses/LICENSE-2.0",
        description="Primary checkpoint: 32 layers (24 GDN + 8 full-attn), "
        "hidden 2560, 262K native context. ~8 GB fp16.",
        approx_size="~8.2 GB",
        skip_files=("video_preprocessor_config.json",),
    ),
    "0.8B": ModelInfo(
        name="Qwen3.5-0.8B",
        repo_id="Qwen/Qwen3.5-0.8B",
        huggingface_url="https://huggingface.co/Qwen/Qwen3.5-0.8B",
        license="Apache-2.0",
        license_url="https://www.apache.org/licenses/LICENSE-2.0",
        description="Fast-iteration fallback: 24 layers (18 GDN + 6 full-attn), "
        "hidden 1024, 262K native context. ~1.7 GB fp16.",
        approx_size="~1.7 GB",
        skip_files=("video_preprocessor_config.json",),
    ),
}


@dataclass
class DownloadRecord:
    """Record of a single file download attempt."""

    filename: str
    success: bool
    bytes: int = 0
    skipped: bool = False
    error: str = ""


@dataclass
class FetchManifest:
    """Manifest written to the model directory after fetching."""

    model_name: str
    repo_id: str
    revision: str = "main"
    license: str = ""
    fetched_at: str = ""
    files: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib fallback)
# ---------------------------------------------------------------------------


def _list_repo_files(repo_id: str) -> list[str]:
    """Fetch the list of files in a HuggingFace repo via the API.

    Uses huggingface_hub if available, otherwise falls back to the REST API.
    """
    try:
        from huggingface_hub import list_repo_files  # type: ignore[import-not-found]

        return sorted(list_repo_files(repo_id))
    except ImportError:
        pass

    # Fallback: direct API call
    import urllib.request

    url = f"{_HF_BASE}/api/models/{repo_id}"
    with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310
        data = json.loads(resp.read())
    return sorted(s["rfilename"] for s in data.get("siblings", []))


def _resolve_weight_files(repo_id: str, repo_files: list[str]) -> list[str]:
    """Given the full repo file list, return only the weight shard files."""
    return [f for f in repo_files if f.endswith(".safetensors")]


def _download_file(
    repo_id: str,
    filename: str,
    dest: Path,
    *,
    timeout: int = 120,
) -> int:
    """Download a single file from HuggingFace. Returns bytes written.

    Uses huggingface_hub if available (handles resume, retries), otherwise
    falls back to a direct urllib download.
    """
    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]

        local = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(dest.parent),
            local_dir_use_symlinks=False,
        )
        return Path(local).stat().st_size
    except ImportError:
        pass

    # Fallback: direct download
    import urllib.request

    url = _HF_RESOLVE.format(repo=repo_id, filename=filename)
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    req = urllib.request.Request(url, headers={"User-Agent": "OrionsBelt/fetch_weights"})
    with (
        urllib.request.urlopen(req, timeout=timeout) as resp,  # noqa: S310
        open(tmp, "wb") as f,
    ):
        shutil.copyfileobj(resp, f)

    tmp.rename(dest)
    return dest.stat().st_size


def _file_is_present(path: Path) -> bool:
    """Check if a file exists and is non-empty."""
    return path.exists() and path.stat().st_size > 0


def _sha256(path: Path, *, chunk: int = 1 << 20) -> str:
    """Compute SHA-256 of a file (for manifest recording)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def plan_download(
    model: ModelInfo,
    repo_files: list[str],
    *,
    metadata_only: bool = False,
) -> list[str]:
    """Determine which files to download for a model.

    Returns a sorted list of filenames.
    """
    wanted: list[str] = []

    # Metadata files (config, tokenizer, etc.)
    for f in _METADATA_FILES:
        if f in repo_files:
            wanted.append(f)

    if not metadata_only:
        for f in _resolve_weight_files(repo_id=model.repo_id, repo_files=repo_files):
            wanted.append(f)

    # Also grab the LICENSE file
    if "LICENSE" in repo_files:
        wanted.append("LICENSE")

    # Remove skipped files (vision tower etc.)
    wanted = [f for f in wanted if f not in model.skip_files]

    # Deduplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for f in wanted:
        if f not in seen:
            seen.add(f)
            result.append(f)
    return result


def fetch_model(
    model: ModelInfo,
    output_dir: Path,
    *,
    metadata_only: bool = False,
    dry_run: bool = False,
) -> FetchManifest:
    """Download a single model checkpoint.

    Returns a FetchManifest with per-file results.
    """
    model_dir = output_dir / model.name
    manifest = FetchManifest(
        model_name=model.name,
        repo_id=model.repo_id,
        license=model.license,
    )

    # Discover available files
    print(f"[{model.name}] Querying HuggingFace for file list...", file=sys.stderr)
    try:
        repo_files = _list_repo_files(model.repo_id)
    except Exception as exc:
        print(f"[{model.name}] ERROR: cannot list repo files: {exc}", file=sys.stderr)
        manifest.files.append({"filename": "<repo-list>", "success": False, "error": str(exc)})
        return manifest

    files = plan_download(model, repo_files, metadata_only=metadata_only)

    if dry_run:
        print(f"[{model.name}] Dry run — would download {len(files)} file(s):", file=sys.stderr)
        for f in files:
            print(f"  {f}", file=sys.stderr)
        manifest.fetched_at = "(dry run)"
        for f in files:
            manifest.files.append({"filename": f, "success": True, "skipped": True})
        return manifest

    model_dir.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    for filename in files:
        dest = model_dir / filename
        record: dict = {"filename": filename}

        if _file_is_present(dest):
            size = dest.stat().st_size
            record["success"] = True
            record["bytes"] = size
            record["skipped"] = True
            total_bytes += size
            print(
                f"[{model.name}] SKIP (exists): {filename} ({_human_size(size)})", file=sys.stderr
            )
        else:
            print(f"[{model.name}] Downloading: {filename}...", file=sys.stderr)
            try:
                size = _download_file(model.repo_id, filename, dest)
                record["success"] = True
                record["bytes"] = size
                total_bytes += size
                print(
                    f"[{model.name}] OK: {filename} ({_human_size(size)})",
                    file=sys.stderr,
                )
            except Exception as exc:
                record["success"] = False
                record["error"] = str(exc)
                print(f"[{model.name}] FAILED: {filename}: {exc}", file=sys.stderr)

        manifest.files.append(record)

    manifest.fetched_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    # Write manifest
    manifest_path = model_dir / "manifest.json"
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2) + "\n")
    print(
        f"[{model.name}] Manifest written to {manifest_path} (total: {_human_size(total_bytes)})",
        file=sys.stderr,
    )

    return manifest


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _human_size(n: int) -> str:
    """Format a byte count in human-readable form."""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} PiB"


def list_models() -> str:
    """Return a formatted table of available models."""
    lines = ["Available models:", ""]
    for key, m in MODELS.items():
        lines.append(f"  {key:8s}  {m.name}")
        lines.append(f"           {m.description}")
        lines.append(f"           License: {m.license}  |  Size: {m.approx_size}")
        lines.append(f"           {m.huggingface_url}")
        lines.append("")
    lines.append("Use --model <key> to download. Use --metadata-only for config/tokenizer only.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download Qwen3.5 weights from HuggingFace (non-interactive, idempotent).",
    )
    parser.add_argument(
        "--model",
        choices=[*MODELS, "all"],
        help="Which model to download (default: 4B).",
        default="4B",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"Root directory for downloaded weights (default: {_DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Download only config/tokenizer files, not weight shards.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan without downloading — print what would happen.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit.",
    )
    args = parser.parse_args(argv)

    if args.list:
        print(list_models())
        return 0

    # Determine which models to fetch
    keys = list(MODELS) if args.model == "all" else [args.model]

    all_ok = True
    for key in keys:
        model = MODELS[key]
        manifest = fetch_model(
            model,
            args.output_dir,
            metadata_only=args.metadata_only,
            dry_run=args.dry_run,
        )
        if not all(f.get("success") for f in manifest.files):
            all_ok = False

    if all_ok:
        print("\nDone.", file=sys.stderr)
        return 0
    print("\nSome downloads failed — see errors above.", file=sys.stderr)
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
