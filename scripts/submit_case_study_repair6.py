"""Submit the corrected CPU-only case-study finalization repair."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import submit_case_study_repair5 as base


ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = "case_study_20260920_repair6"
base.RUN_NAME = RUN_NAME
base.REMOTE_SNAPSHOT = f"{base.REMOTE_ROOT}/snapshots/{RUN_NAME}"
base.REMOTE_RUN = f"{base.REMOTE_ROOT}/run/{RUN_NAME}"
base.REMOTE_SCRIPT = f"{base.REMOTE_SNAPSHOT}/run/finalize_case_study.slurm"
base.JOBS_PATH = ROOT / "run" / "case_study_20260920" / "submitted_jobs_case_study_repair6_20260921.json"


def main() -> None:
    base.main()
    payload = json.loads(base.JOBS_PATH.read_text())
    payload.update({
        "job_name": "athena-case-finalize-r6",
        "repair_of": "22201150",
        "root_cause": (
            "Repair5 incorrectly required the fixed reader at the realised argmax to be correct, "
            "which is stronger than the requested end-to-end router-correct case-study criterion."
        ),
        "repair": (
            "Finalize exactly three rows where Base and RAG are wrong and the router answer is correct; "
            "retain the realised argmax and fixed-path output as diagnostics without overstating standalone-reader correctness."
        ),
    })
    base.JOBS_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"recorded corrected case-study repair6 manifest at {base.JOBS_PATH}")


if __name__ == "__main__":
    main()
