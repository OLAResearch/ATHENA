"""Submit a self-contained case-study finalization repair."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import submit_case_study_repair5 as base


ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = "case_study_20260920_repair7"
base.RUN_NAME = RUN_NAME
base.REMOTE_SNAPSHOT = f"{base.REMOTE_ROOT}/snapshots/{RUN_NAME}"
base.REMOTE_RUN = f"{base.REMOTE_ROOT}/run/{RUN_NAME}"
base.REMOTE_SCRIPT = f"{base.REMOTE_SNAPSHOT}/run/finalize_case_study.slurm"
base.JOBS_PATH = ROOT / "run" / "case_study_20260920" / "submitted_jobs_case_study_repair7_20260921.json"


def main() -> None:
    base.main()
    payload = json.loads(base.JOBS_PATH.read_text())
    payload.update({
        "job_name": "athena-case-finalize-r7",
        "repair_of": "22201170",
        "root_cause": "Repair6 imported the full inference module during post-processing and did not reproduce the valid source selection in the container.",
        "repair": "Use a self-contained post-processor over the completed repair3 results to select exactly three Base/RAG-wrong, router-correct examples.",
    })
    base.JOBS_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"recorded corrected case-study repair7 manifest at {base.JOBS_PATH}")


if __name__ == "__main__":
    main()
