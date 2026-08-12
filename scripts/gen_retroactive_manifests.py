#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0

"""Generate retroactive manifests for t4 CSVs that are missing them.

Uses an existing t4 retroactive manifest as the device-info template,
varying only the CSV name and git SHA per file.
"""

import json
from datetime import datetime, timezone

TEMPLATE = "results/manifests/rk3588-t4_big_singlethread.json"
OUTPUT_DIR = "results/manifests"

# (csv_name, first_commit_sha) pairs
CSV_TO_SHA = {
    "rk3588-t4_a55_fp32": "80f61bf168f86318295aa2f16a4899a6308fb612",
    "rk3588-t4_a55_int4": "80f61bf168f86318295aa2f16a4899a6308fb612",
    "rk3588-t4_a55_int8": "80f61bf168f86318295aa2f16a4899a6308fb612",
    "rk3588-t4_a76_fp32": "80f61bf168f86318295aa2f16a4899a6308fb612",
    "rk3588-t4_a76_int4": "80f61bf168f86318295aa2f16a4899a6308fb612",
    "rk3588-t4_a76_int8": "80f61bf168f86318295aa2f16a4899a6308fb612",
    "rk3588-t4_prefill_big_4b_optimized": "98629691f15b7c1c6ee928272d4aee62f1778349",
    "rk3588-t4_prefill_big_int8_naive_m8": "98629691f15b7c1c6ee928272d4aee62f1778349",
    "rk3588-t4_prefill_big_int8_optimized": "98629691f15b7c1c6ee928272d4aee62f1778349",
    "rk3588-t4_prefill_big_naive_m8": "98629691f15b7c1c6ee928272d4aee62f1778349",
    "rk3588-t4_prefill_big_optimized": "98629691f15b7c1c6ee928272d4aee62f1778349",
    "rk3588-t4_prefill_little_naive_m8": "98629691f15b7c1c6ee928272d4aee62f1778349",
    "rk3588-t4_prefill_little_optimized": "98629691f15b7c1c6ee928272d4aee62f1778349",
}


def main():
    with open(TEMPLATE) as f:
        template = json.load(f)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for csv_name, sha in sorted(CSV_TO_SHA.items()):
        manifest = json.loads(json.dumps(template))  # deep copy
        manifest["caller"]["original_csv"] = csv_name + ".csv"
        manifest["caller"]["retroactive"] = True
        manifest["git"]["sha"] = sha
        manifest["git"]["dirty"] = True
        manifest["git"]["retroactive_note"] = (
            "Manifest generated after the fact. git.sha is the commit where "
            "the CSV was first committed to the repo, not necessarily the "
            "exact tree state at benchmark time. dirty=true is conservative "
            "(actual state unknown)."
        )
        manifest["run_id"] = "t4_retroactive_" + sha[:7]
        manifest["timestamp_utc"] = now

        out_path = OUTPUT_DIR + "/" + csv_name + ".json"
        with open(out_path, "w") as f:
            json.dump(manifest, f, indent=2)
            f.write("\n")
        print(f"  wrote {out_path}")

    print(f"\nGenerated {len(CSV_TO_SHA)} retroactive manifests")


if __name__ == "__main__":  # pragma: no cover
    main()
