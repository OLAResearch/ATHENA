import numpy as np
from types import SimpleNamespace

from scripts.bootstrap_dual_reader_openqa import build_comparisons
from scripts.eval_dual_reader_openqa_paired import (
    compare_openqa_results,
    compare_truthfulqa_results,
    paired_bootstrap,
    set_reader_mode,
)


def test_legacy_dual_reader_does_not_dispatch_to_compat_tri_setter():
    class LegacyAdaptor:
        fusion_type = "dual_reader"
        router = object()

        def __init__(self):
            self.calls = []

        def set_tri_reader_mode(self, mode):
            self.calls.append(("tri", mode))

        def set_dual_reader_mode(self, mode):
            self.calls.append(("dual", mode))

    adaptor = LegacyAdaptor()
    set_reader_mode(SimpleNamespace(adaptor=adaptor), "generated_only")
    assert adaptor.calls == [("dual", "generated_only")]


def test_tri_advantage_mode_dispatches_to_tri_setter():
    class TriAdaptor:
        fusion_type = "tri_reader"
        router = object()

        def __init__(self):
            self.calls = []

        def set_tri_reader_mode(self, mode):
            self.calls.append(mode)

    adaptor = TriAdaptor()
    set_reader_mode(SimpleNamespace(adaptor=adaptor), "tri_advantage_routed")
    assert adaptor.calls == ["tri_advantage_routed"]


def test_paired_bootstrap_reports_direction_and_counts():
    result = paired_bootstrap(
        np.asarray([0.0, 1.0, 0.0]),
        np.asarray([1.0, 1.0, 0.0]),
        seed=7,
        samples=100,
    )
    assert result["delta"] == 1 / 3
    assert result["improved"] == 1
    assert result["regressed"] == 0
    assert result["tied"] == 2
    assert len(result["delta_95pct_paired_bootstrap"]) == 2


def test_compare_openqa_results_pairs_em_and_f1_by_question():
    baseline = {
        "sample_predictions": [
            {"question": "q1", "correct": False, "f1": 0.0},
            {"question": "q2", "correct": True, "f1": 0.5},
        ]
    }
    candidate = {
        "sample_predictions": [
            {"question": "q1", "correct": True, "f1": 1.0},
            {"question": "q2", "correct": True, "f1": 0.25},
        ]
    }
    result = compare_openqa_results(
        baseline,
        candidate,
        candidate_name="routed",
        seed=42,
        bootstrap_samples=100,
    )
    assert result["n_examples"] == 2
    assert result["metrics"]["em"]["delta"] == 0.5
    assert result["metrics"]["f1"]["delta"] == 0.375


def test_compare_truthfulqa_results_pairs_all_mc_metrics():
    def row(question, mc1, mc2, mc3):
        return {
            "question": question,
            "metrics": {"MC1": mc1, "MC2": mc2, "MC3": mc3},
        }

    baseline = {"sample_examples": [row("q1", 0.0, 0.2, 0.0), row("q2", 1.0, 0.6, 0.5)]}
    candidate = {"sample_examples": [row("q1", 1.0, 0.4, 0.5), row("q2", 1.0, 0.5, 1.0)]}
    result = compare_truthfulqa_results(
        baseline,
        candidate,
        candidate_name="routed",
        seed=42,
        bootstrap_samples=100,
    )
    assert set(result["metrics"]) == {"mc1", "mc2", "mc3", "mc_avg"}
    assert result["metrics"]["mc1"]["delta"] == 0.5
    assert np.isclose(result["metrics"]["mc2"]["delta"], 0.05)
    assert np.isclose(result["metrics"]["mc3"]["delta"], 0.5)
    assert np.isclose(result["metrics"]["mc_avg"]["delta"], 0.35)


def test_cpu_postprocessor_builds_comparisons_for_each_mode():
    predictions = [
        {"question": "q1", "correct": False, "f1": 0.0},
        {"question": "q2", "correct": True, "f1": 1.0},
    ]
    improved = [
        {"question": "q1", "correct": True, "f1": 1.0},
        {"question": "q2", "correct": True, "f1": 1.0},
    ]
    results = {
        "completed": True,
        "modes": ["engram_only", "both", "routed"],
        "tasks": {"webqa": {"n_examples": 2}},
        "evaluation": {
            "engram_only": {"webqa": {"metrics": {"sample_predictions": predictions}}},
            "both": {"webqa": {"metrics": {"sample_predictions": predictions}}},
            "routed": {"webqa": {"metrics": {"sample_predictions": improved}}},
        },
    }
    comparisons = build_comparisons(results, samples=100, seed=42)
    assert comparisons["webqa"]["both"]["metrics"]["em"]["delta"] == 0.0
    assert comparisons["webqa"]["routed"]["metrics"]["em"]["delta"] == 0.5
