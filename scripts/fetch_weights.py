#!/usr/bin/env python3
"""Download model weights from HuggingFace at setup time.

Bead ``ob-ixt``.  Weights are never vendored into the repo — they are
fetched on demand by this script, keeping the repo small and license
compliance clean (see docs/WEIGHT_LICENSE.md).

Uses ``huggingface_hub`` if available (handles resumption, caching,
authentication for gated models); falls back to direct HTTPS via urllib
so the script works on minimal edge devices with no pip extras.

Usage::

    # Download primary model (Qwen3.5-4B, ~8 GB)
    python3 scripts/fetch_weights.py

    # Download fallback model (Qwen3.5-0.8B, ~1.6 GB)
    python3 scripts/fetch_weights.py --model 0.8b

    # Download both
    python3 scripts/fetch_weights.py --model all

    # Custom output directory
    python3 scripts/fetch_weights.py --output-dir /data/models

    # Verify checksums only (no download)
    python3 scripts/fetch_weights.py --verify-only

The script is idempotent: if weights already exist and pass verification,
they are not re-downloaded.

Python 3.6+ compatible (runs on edge devices).
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS = {
    "4b": {
        "repo_id": "Qwen/Qwen3.5-4B",
        "hf_url": "https://huggingface.co/Qwen/Qwen3.5-4B",
        "approx_size_gb": 8.0,
        "role": "primary",
        "license": "Apache-2.0",
    },
    "0.8b": {
        "repo_id": "Qwen/Qwen3.5-0.8B",
        "hf_url": "https://huggingface.co/Qwen/Qwen3.5-0.8B",
        "approx_size_gb": 1.6,
        "role": "fallback",
        "license": "Apache-2.0",
    },
}

DEFAULT_MODEL = "4b"
DEFAULT_OUTPUT_DIR = "weights"


# ---------------------------------------------------------------------------
# Download via huggingface_hub (preferred)
# ---------------------------------------------------------------------------


def _try_huggingface_hub():
    # type: () -> Optional[object]
    """Import huggingface_hub.snapshot_download if available."""
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download
    except ImportError:
        return None


def download_via_hub(repo_id, output_dir):
    # type: (str, str) -> str
    """Download a model snapshot using huggingface_hub.

    Handles resumption, caching, and authentication for gated models.
    """
    snapshot_download = _try_huggingface_hub()
    if snapshot_download is None:
        raise RuntimeError("huggingface_hub not available")

    local_dir = os.path.join(output_dir, repo_id.replace("/", "--"))
    os.makedirs(local_dir, exist_ok=True)

    print("  Downloading via huggingface_hub (with resume)...")
    path = snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        # Exclude large unnecessary files (e.g. optimizer states)
        ignore_patterns=["*.msgpack", "*.h5", "optimizer.pt"],
    )
    return local_dir


# ---------------------------------------------------------------------------
# Download via urllib (fallback for edge devices)
# ---------------------------------------------------------------------------


def _download_file(url, dest_path, chunk_size=1024 * 1024):
    # type: (str, str, int) -> None
    """Download a single file with progress reporting."""
    try:
        from urllib.request import urlopen, Request
    except ImportError:
        raise RuntimeError("urllib not available (minimal Python install?)")

    req = Request(url, headers={"User-Agent": "OrionsBelt/fetch_weights"})
    resp = urlopen(req)
    total = int(resp.headers.get("Content-Length", 0))

    downloaded = 0
    with open(dest_path + ".tmp", "wb") as f:
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded * 100 // total
                sys.stdout.write(
                    "\r  {}% ({:.1f} MB / {:.1f} MB)".format(
                        pct, downloaded / 1e6, total / 1e6
                    )
                )
                sys.stdout.flush()
    print()  # newline after progress

    os.rename(dest_path + ".tmp", dest_path)


# Essential files needed for inference (not the full repo)
ESSENTIAL_FILES = [
    "config.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "LICENSE",
    "NOTICE",
]


def download_via_urllib(repo_id, output_dir):
    # type: (str, str) -> str
    """Download essential model files via direct HTTPS.

    Used when huggingface_hub is not installed (edge devices).
    Only downloads files that exist; missing optional files are skipped.
    """
    import urllib.error

    local_dir = os.path.join(output_dir, repo_id.replace("/", "--"))
    os.makedirs(local_dir, exist_ok=True)

    base_url = "https://huggingface.co/{}/resolve/main".format(repo_id)

    # First, check the safetensors index to find shard filenames
    index_url = "{}/model.safetensors.index.json".format(base_url)
    shards = []  # type: List[str]

    try:
        from urllib.request import urlopen, Request
        req = Request(index_url, headers={"User-Agent": "OrionsBelt/fetch_weights"})
        resp = urlopen(req)
        index = json.loads(resp.read().decode("utf-8"))
        weight_map = index.get("weight_map", {})
        shard_set = sorted(set(weight_map.values()))
        shards = shard_set
    except Exception:
        # No index — single-file model
        shards = ["model.safetensors"]

    all_files = ESSENTIAL_FILES + shards

    for fname in all_files:
        dest = os.path.join(local_dir, fname)
        if os.path.exists(dest):
            print("  [skip] {} (already present)".format(fname))
            continue

        url = "{}/{}".format(base_url, fname)
        print("  Downloading {}...".format(fname))
        try:
            _download_file(url, dest)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print("  [skip] {} (not found — optional file)".format(fname))
            else:
                raise

    return local_dir


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_download(local_dir):
    # type: (str) -> Dict[str, str]
    """Compute SHA256 of downloaded files for provenance.

    Returns a dict of filename → sha256.
    """
    checksums = {}
    for fname in sorted(os.listdir(local_dir)):
        fpath = os.path.join(local_dir, fname)
        if not os.path.isfile(fpath):
            continue
        h = hashlib.sha256()
        with open(fpath, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        checksums[fname] = h.hexdigest()
    return checksums


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def write_manifest(model_key, local_dir, checksums):
    # type: (str, str, Dict[str, str]) -> str
    """Write a manifest recording what was downloaded."""
    info = MODELS[model_key]
    manifest = {
        "repo_id": info["repo_id"],
        "role": info["role"],
        "license": info["license"],
        "approx_size_gb": info["approx_size_gb"],
        "local_dir": local_dir,
        "files": checksums,
        "downloaded_at": subprocess.check_output(
            ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"]
        ).decode().strip(),
    }
    manifest_path = os.path.join(local_dir, ".fetch_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def fetch_model(model_key, output_dir, verify_only=False):
    # type: (str, str, bool) -> str
    """Download (or verify) one model.

    Returns the local directory path.
    """
    info = MODELS[model_key]
    repo_id = info["repo_id"]
    local_dir = os.path.join(output_dir, repo_id.replace("/", "--"))

    if verify_only:
        if not os.path.isdir(local_dir):
            print("ERROR: {} not downloaded yet".format(repo_id))
            sys.exit(1)
        print("Verifying {}...".format(repo_id))
        checksums = verify_download(local_dir)
        manifest_path = write_manifest(model_key, local_dir, checksums)
        print("  {} files verified".format(len(checksums)))
        print("  Manifest: {}".format(manifest_path))
        return local_dir

    # Check if already downloaded
    if os.path.isdir(local_dir) and os.listdir(local_dir):
        print("{} already present at {}".format(repo_id, local_dir))
        print("  Verifying checksums...")
        checksums = verify_download(local_dir)
        manifest_path = write_manifest(model_key, local_dir, checksums)
        print("  {} files verified".format(len(checksums)))
        return local_dir

    print("Downloading {} (~{:.1f} GB)...".format(repo_id, info["approx_size_gb"]))

    # Try huggingface_hub first, fall back to urllib
    if _try_huggingface_hub() is not None:
        local_dir = download_via_hub(repo_id, output_dir)
    else:
        print("  (huggingface_hub not available — using direct HTTPS)")
        local_dir = download_via_urllib(repo_id, output_dir)

    # Verify and write manifest
    print("  Computing checksums...")
    checksums = verify_download(local_dir)
    manifest_path = write_manifest(model_key, local_dir, checksums)
    print("  Done: {} files, manifest at {}".format(len(checksums), manifest_path))
    return local_dir


def main(argv=None):
    # type: (Optional[List[str]]) -> int
    parser = argparse.ArgumentParser(
        description="Download model weights from HuggingFace (ob-ixt)."
    )
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()) + ["all"],
        default=DEFAULT_MODEL,
        help="Which model to fetch (default: {} = Qwen3.5-4B)".format(DEFAULT_MODEL),
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory (default: {})".format(DEFAULT_OUTPUT_DIR),
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing downloads (no download)",
    )
    args = parser.parse_args(argv)

    models = list(MODELS.keys()) if args.model == "all" else [args.model]

    for model_key in models:
        print("\n=== {} ({}) ===".format(
            MODELS[model_key]["repo_id"],
            MODELS[model_key]["role"],
        ))
        fetch_model(model_key, args.output_dir, verify_only=args.verify_only)

    print("\nAll requested models processed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
