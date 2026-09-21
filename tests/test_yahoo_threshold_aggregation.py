import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("integer_tag", ["1", "1p0"])
def test_integer_threshold_directory(tmp_path, integer_tag):
    for key, tag in [("0.9", "0p9"), ("1", integer_tag)]:
        folder = tmp_path / f"threshold-{tag}"
        folder.mkdir()
        payload = {
            "checkpoint": {"path": "same-checkpoint"},
            "summary": {key: 0.57},
            "evaluation": {key: {"accuracy": 0.57}},
        }
        (folder / "results.json").write_text(json.dumps(payload))
    output = tmp_path / "aggregated"
    script = Path(__file__).resolve().parents[1] / "scripts/aggregate_yahoo_router_threshold_sweep.py"
    subprocess.run(
        [sys.executable, str(script), "--input-dir", str(tmp_path),
         "--output-dir", str(output), "--thresholds", "0.9,1.0"],
        check=True,
    )
    result = json.loads((output / "results.json").read_text())
    assert result["summary"] == {"0.9": 0.57, "1": 0.57}
