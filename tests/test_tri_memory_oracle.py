import json

import pytest

from scripts.tri_memory_oracle import build_all_oracles, build_oracle_metrics, capture_ratio
from scripts.recompute_tri_oracles import READER_MODES, _recompute
from scripts.eval_fair_joint_openqa_paired import TRI_MODE_MAP
from scripts.eval_tri_memory_openqa_paired import RUNTIME_MODES


def _openqa_metrics(rows):
    correct_values = [float(correct) for _, correct, _ in rows]
    f1_values = [float(f1) for _, _, f1 in rows]
    return {
        "em": sum(correct_values) / len(correct_values),
        "f1": sum(f1_values) / len(f1_values),
        "sample_predictions": [
            {
                "question": f"q{index}",
                "prediction": source,
                "answers": ["a"],
                "correct": correct,
                "f1": f1,
            }
            for index, (source, correct, f1) in enumerate(rows)
        ]
    }


def test_openqa_oracle_selects_one_source_per_example_by_f1():
    oracle = build_oracle_metrics(
        "nq",
        {
            "tri_E": _openqa_metrics([("e0", True, 0.5), ("e1", False, 0.1)]),
            "tri_GE": _openqa_metrics([("ge0", False, 0.8), ("ge1", True, 0.4)]),
            "tri_GH": _openqa_metrics([("gh0", True, 0.7), ("gh1", False, 0.9)]),
        },
    )
    assert oracle["f1"] == pytest.approx(0.85)
    assert oracle["em"] == pytest.approx(0.0)
    assert oracle["metricwise_upper"]["em"] == pytest.approx(1.0)
    assert oracle["selection_counts"] == {"tri_E": 0, "tri_GE": 1, "tri_GH": 1}


def test_all_pairwise_and_three_way_oracles_are_present():
    evaluation = {
        source: {"nq": {"metrics": _openqa_metrics([(source, False, score)])}}
        for source, score in (
            ("tri_E", 0.1),
            ("tri_GE", 0.2),
            ("tri_GH", 0.3),
            ("tri_E_GE", 0.1),
            ("tri_E_GH", 0.1),
            ("tri_GE_GH", 0.1),
            ("tri_reader_soft", 0.1),
            ("tri_subset_reader_hard", 0.1),
            ("tri_subset_reader_soft", 0.1),
        )
    }
    result = build_all_oracles("nq", evaluation)
    assert set(result) == {
        "oracle_source_E_GE",
        "oracle_source_E_GH",
        "oracle_source_GE_GH",
        "oracle_source_E_GE_GH",
        "oracle_E_GE",
        "oracle_E_GH",
        "oracle_GE_GH",
        "oracle_E_GE_GH",
        "oracle_all_nonempty_subsets",
    }
    assert result["oracle_E_GE_GH"]["f1"] == pytest.approx(0.3)
    assert result["oracle_E_GE"]["f1"] == pytest.approx(0.2)


def test_runtime_oracle_credits_pair_and_triple_endpoints():
    evaluation = {
        source: {"nq": {"metrics": _openqa_metrics([(source, False, score)])}}
        for source, score in (
            ("tri_E", 0.1),
            ("tri_GE", 0.2),
            ("tri_GH", 0.3),
            ("tri_E_GE", 0.4),
            ("tri_E_GH", 0.5),
            ("tri_GE_GH", 0.6),
            # The unified Reader is evaluated separately and is not an oracle
            # endpoint; it must not inflate the theoretical ceiling.
            ("tri_reader_hard", 0.99),
            ("tri_reader_soft", 0.98),
            ("tri_subset_reader_hard", 0.7),
            ("tri_subset_reader_soft", 0.8),
        )
    }
    result = build_all_oracles("nq", evaluation)
    assert result["oracle_all_nonempty_subsets"]["f1"] == pytest.approx(0.98)


def test_capture_ratio_handles_unavailable_and_overshot_upper_bound():
    assert capture_ratio(12.0, 10.0, 14.0) == pytest.approx(0.5)
    assert capture_ratio(15.0, 10.0, 14.0) == pytest.approx(1.25)
    assert capture_ratio(10.0, 10.0, 10.0) is None


def test_corrected_oracle_excludes_learned_subset_reader():
    assert READER_MODES == (
        "tri_reader_hard",
        "tri_reader_soft",
        "tri_reader_advantage",
        "tri_subset_reader_hard",
        "tri_subset_reader_soft",
    )


def test_advantage_reader_is_registered_in_paired_evaluation_modes():
    assert RUNTIME_MODES["tri_reader_advantage"] == "tri_advantage_routed"
    assert TRI_MODE_MAP["tri_reader_advantage"] == "tri_advantage_routed"


def test_recompute_writes_corrected_artifact_without_overwriting_input(tmp_path):
    modes = (
        "engram_baseline",
        "tri_E",
        "tri_GE",
        "tri_GH",
        "tri_E_GE",
        "tri_E_GH",
        "tri_GE_GH",
        "tri_reader_hard",
        "tri_reader_soft",
        "tri_reader_advantage",
        "tri_subset_reader_hard",
        "tri_subset_reader_soft",
    )
    scores = {
        "engram_baseline": 0.1,
        "tri_E": 0.2,
        "tri_GE": 0.3,
        "tri_GH": 0.4,
        "tri_E_GE": 0.5,
        "tri_E_GH": 0.6,
        "tri_GE_GH": 0.7,
        "tri_reader_hard": 0.8,
        "tri_reader_soft": 0.85,
        "tri_reader_advantage": 0.9,
        "tri_subset_reader_hard": 0.99,
        "tri_subset_reader_soft": 0.98,
    }
    evaluation = {
        mode: {"nq": {"metrics": _openqa_metrics([(mode, False, score)])}}
        for mode, score in scores.items()
    }
    source = tmp_path / "results.json"
    source.write_text(json.dumps({
        "completed": True,
        "tasks": {"nq": {}},
        "modes": list(modes),
        "evaluation": evaluation,
    }))

    output = _recompute(source, bootstrap_samples=8, seed=42)
    assert output == tmp_path / "results_with_corrected_oracle.json"
    assert source.exists()
    corrected = json.loads(output.read_text())
    assert corrected["oracle_definition"] == (
        "strict_fixed_seven_endpoints_excluding_learned_subset_reader"
    )
    assert corrected["oracles"]["nq"]["oracle_all_nonempty_subsets"]["f1"] == pytest.approx(0.85)
    assert "oracle_all_nonempty_subsets" in corrected["paired_comparisons"]["nq"]
