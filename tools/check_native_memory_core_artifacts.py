"""Validate the frozen E0.14 fixed-core artifacts required by E0.16."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wamrx.canonical import sha256_json  # noqa: E402
from wamrx.recurrent_selection import file_sha256, validate_selection_files  # noqa: E402


def check() -> dict:
    contract = json.loads((ROOT / "contracts" / "wamrx_native_memory_v1.json").read_text())
    manifest_path = ROOT / contract["frozen_core_weights"]["path"]
    manifest = json.loads(manifest_path.read_text())
    selection = validate_selection_files(ROOT)
    result = json.loads((ROOT / "results" / "e0_14_recurrent_model_comparison.json").read_text())
    reports = {
        int(row["seed"]): row["checkpoint"]
        for row in result["training_reports"]
        if row["arm_id"] == "fixed-depth-v1"
    }
    identity_errors = []
    if sha256_json(manifest) != contract["frozen_core_weights"]["content_hash"]:
        identity_errors.append("core checkpoint manifest hash differs from contract")
    if manifest["source_result_sha256"] != selection["source_result_sha256"]:
        identity_errors.append("core checkpoint manifest names a different E0.14 result")
    if set(reports) != {row["seed"] for row in manifest["checkpoints"]}:
        identity_errors.append("core checkpoint seed set differs from E0.14")

    artifacts = []
    for row in manifest["checkpoints"]:
        report = reports.get(row["seed"], {})
        identity_matches = all(
            report.get(key) == row[key]
            for key in ("bytes", "file_sha256", "metadata_hash", "state_hash")
        )
        if not identity_matches:
            identity_errors.append(f"seed {row['seed']} identity differs from E0.14")
        relocation = ROOT / row["relocation_path"]
        source = pathlib.Path(row["source_path"])
        located = relocation if relocation.exists() else source if source.exists() else None
        size_matches = located is not None and located.stat().st_size == row["bytes"]
        hash_matches = size_matches and file_sha256(located) == row["file_sha256"]
        artifacts.append(
            {
                "seed": row["seed"],
                "identity_matches_e0_14": identity_matches,
                "available": located is not None,
                "located_path": str(located) if located is not None else None,
                "size_matches": size_matches,
                "file_sha256_matches": hash_matches,
                "relocation_path": str(relocation),
            }
        )
    ready = not identity_errors and all(row["file_sha256_matches"] for row in artifacts)
    e0_16 = json.loads(
        (ROOT / "results" / "e0_16_native_memory_comparison.json").read_text()
    )
    return {
        "status": "READY" if ready else "NOT_AVAILABLE" if not identity_errors else "INVALID",
        "e0_16_status": e0_16["status"],
        "manifest_id": manifest["manifest_id"],
        "manifest_hash": sha256_json(manifest),
        "identity_errors": identity_errors,
        "artifacts": artifacts,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-files",
        action="store_true",
        help="exit non-zero unless all five exact checkpoint files are available",
    )
    args = parser.parse_args()
    report = check()
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] == "INVALID":
        return 1
    if args.require_files and report["status"] != "READY":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
