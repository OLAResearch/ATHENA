import json
import math

from scripts.eval_code_functional import (
    estimate_pass_at_k,
    evaluate_kodcode_record,
    prediction_samples,
    update_result_file,
)


def test_pass_at_k_matches_unbiased_estimator_and_undefined_k():
    assert math.isclose(estimate_pass_at_k(10, 1, 1), 0.1)
    assert math.isclose(estimate_pass_at_k(10, 1, 5), 0.5)
    assert estimate_pass_at_k(10, 0, 10) == 0.0
    assert estimate_pass_at_k(1, 1, 5) is None


def test_prediction_samples_preserves_multi_sample_records():
    assert prediction_samples({"predictions": ["a", "b"]}) == ["a", "b"]
    assert prediction_samples({"prediction": "a"}) == ["a"]


def test_kodcode_functional_record_passes():
    result = evaluate_kodcode_record(
        {
            "task_id": "demo",
            "prediction": "def add(a, b):\n    return a + b\n",
            "tests": "from solution import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        },
        timeout=5,
    )
    assert result["passed"] is True
    assert result["status"] == "pass"


def test_kodcode_renames_first_function_like_memgen_evaluator():
    result = evaluate_kodcode_record(
        {
            "task_id": "rename-demo",
            "prediction": "def generated_name(a, b):\n    return a + b\n",
            "tests": "from solution import expected_name\n\ndef test_add():\n    assert expected_name(2, 3) == 5\n",
            "test_info": [{"function_name": "expected_name"}],
        },
        timeout=5,
    )
    assert result["passed"] is True


def test_update_result_file_preserves_lexical_metrics(tmp_path):
    path = tmp_path / "results.json"
    path.write_text(json.dumps({
        "tasks": {
            "kodcode": {
                "metrics": {"acc": None, "em": 0.1, "f1": 0.2}
            }
        }
    }))
    update_result_file(path, "kodcode", 0.75, {
        "passed": 3,
        "total": 4,
        "test_pass_rate": 0.75,
        "pass_at_k": {"pass@1": 0.75, "pass@5": None, "pass@10": None},
    })
    metrics = json.loads(path.read_text())["tasks"]["kodcode"]["metrics"]
    assert metrics["acc"] == 0.75
    assert metrics["em"] == 0.1
    assert metrics["f1"] == 0.2
    assert metrics["test_pass_rate"] == 0.75
    assert metrics["pass_at_k"]["pass@1"] == 0.75
