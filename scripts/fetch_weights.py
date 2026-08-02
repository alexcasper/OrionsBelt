#!/usr/bin/env python3
"""Fetch model weights from HuggingFace at setup time (ob-ixt).

Downloads Qwen3.5 checkpoints to a local cache rather than vendoring them in
the repo. This keeps the repo small and avoids redistribution concerns — the
checkpoints are downloaded at setup time under their own Apache-2.0 license.

Usage::

    python3 scripts/fetch_weights.py                    # fetch 4B (primary)
    python3 scripts/fetch_weights.py --model 0.8b        # fetch 0.8B (fallback)
    python3 scripts/fetch_weights.py --all               # fetch both
    python3 scripts/fetch_weights.py --check             # check what's cached

The script uses ``huggingface_hub.snapshot_download`` if available, falling back
to printing manual download instructions. Weights are NOT vendored in the repo.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CHECKPOINTS = {
    "4b": {
        "repo_id": "Qwen/Qwen3.5-4B",
        "description": "Primary checkpoint — 4B params, 24 GDN + 8 FA layers",
        "license": "Apache-2.0",
        "license_url": "https://huggingface.co/Qwen/Qwen3.5-4B/raw/main/LICENSE",
        "approx_size_gb": 16,  # bf16 safetensors + vision tower
        "config_ref": "docs/adr/0003-model-checkpoint-selection.md",
    },
    "0.8b": {
        "repo_id": "Qwen/Qwen3.5-0.8B",
        "description": "Fallback checkpoint — 0.8B params, 18 GDN + 6 FA layers",
        "license": "Apache-2.0",
        "license_url": "https://huggingface.co/Qwen/Qwen3.5-0.8B/raw/main/LICENSE",
        "approx_size_gb": 3,
        "config_ref": "docs/adr/0003-model-checkpoint-selection.md",
    },
}

CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"


def _try_hf_download(repo_id: str, **kwargs) -> Path | None:
    """Download via huggingface_hub if installed. Returns local path or None."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return None

    path = snapshot_download(repo_id=repo_id, **kwargs)
    return Path(path)


def fetch(model_key: str, *, dry_run: bool = False) -> Path | None:
    """Fetch a checkpoint. Returns the local path, or None on failure."""
    info = CHECKPOINTS[model_key]
    repo_id = info["repo_id"]

    if dry_run:
        print(f"  [{model_key}] Would download {repo_id} (~{info['approx_size_gb']} GB)")
        print(f"         License: {info['license']} ({info['license_url']})")
        return None

    print(f"  [{model_key}] Downloading {repo_id} (~{info['approx_size_gb']} GB)...")
    path = _try_hf_download(repo_id)
    if path is not None:
        print(f"  [{model_key}] Cached at: {path}")
        # Write a manifest recording what was fetched
        _write_fetch_manifest(model_key, path, info)
        return path

    # Fallback: print manual instructions
    print(f"  [{model_key}] huggingface_hub not installed. Manual download:")
    print("         pip install huggingface_hub")
    print(f"         huggingface-cli download {repo_id}")
    print(f"         Or: git clone https://huggingface.co/{repo_id}")
    print(f"         License: {info['license']}")
    return None


def _write_fetch_manifest(model_key: str, path: Path, info: dict) -> None:
    """Record what was fetched, for provenance (PLAN.md section 9)."""
    manifest_dir = Path("results/manifests")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "type": "weight_fetch",
        "model_key": model_key,
        "repo_id": info["repo_id"],
        "license": info["license"],
        "local_path": str(path),
        "files": sorted(p.name for p in path.iterdir()) if path.exists() else [],
    }
    manifest_path = manifest_dir / f"weights_{model_key}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def check_cached() -> dict[str, bool]:
    """Check which checkpoints are already in the HF cache."""
    cached = {}
    for key, info in CHECKPOINTS.items():
        repo_dir = CACHE_DIR / f"models--{info['repo_id'].replace('/', '--')}"
        cached[key] = repo_dir.exists()
    return cached


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch Qwen3.5 model weights (ob-ixt)")
    parser.add_argument("--model", default="4b", choices=list(CHECKPOINTS), help="Which checkpoint")
    parser.add_argument("--all", action="store_true", help="Fetch all checkpoints")
    parser.add_argument(
        "--check", action="store_true", help="Check cache status without downloading"
    )
    args = parser.parse_args(argv)

    if args.check:
        cached = check_cached()
        for key, is_cached in cached.items():
            status = "✅ cached" if is_cached else "❌ not cached"
            print(f"  {key:>5}: {CHECKPOINTS[key]['repo_id']} — {status}")
        return 0

    if args.all:
        for key in CHECKPOINTS:
            fetch(key)
    else:
        fetch(args.model)

    return 0


if __name__ == "__main__":
    sys.exit(main())
