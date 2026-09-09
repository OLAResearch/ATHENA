import ast
import copy
from pathlib import Path
import unittest

from scripts.validate_tri_advantage_eval import EXPECTED_MODES, TASK_COUNTS, validate


def fixture(task):
    count = TASK_COUNTS[task]
    keys = ("mc1", "mc2", "mc3", "mc_avg") if task == "truthfulqa" else ("em", "f1")
    sample_key = "sample_examples" if task == "truthfulqa" else "sample_predictions"
    return {
        "completed": True, "bootstrap_completed": False,
        "tasks": {task: {"n_examples": count}}, "modes": sorted(EXPECTED_MODES),
        "evaluation": {mode: {task: {"metrics": {
            **dict.fromkeys(keys, 0.5), "total": count, "elapsed_s": 1.0,
            sample_key: [None] * count,
        }}} for mode in EXPECTED_MODES},
    }


class ValidateTriEvaluationTest(unittest.TestCase):
    def test_full_results_for_all_tasks(self):
        for task in TASK_COUNTS:
            with self.subTest(task=task):
                self.assertEqual(len(validate(fixture(task), task)[task]), 13)

    def test_incomplete_or_invalid_results_are_rejected(self):
        original = fixture("truthfulqa")
        for corruption in ("missing_mode", "partial_samples", "nan", "incomplete"):
            result = copy.deepcopy(original)
            metrics = result["evaluation"]["engram_baseline"]["truthfulqa"]["metrics"]
            if corruption == "missing_mode":
                del result["evaluation"]["tri_reader_advantage"]
            elif corruption == "partial_samples":
                metrics["sample_examples"].pop()
            elif corruption == "nan":
                metrics["mc1"] = float("nan")
            else:
                result["completed"] = False
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                validate(result, "truthfulqa")

    def test_modes_match_actual_evaluator(self):
        source = Path(__file__).resolve().parents[1] / "scripts/eval_fair_joint_openqa_paired.py"
        tree = ast.parse(source.read_text())
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "TRI_MODE_MAP" for t in node.targets))
        self.assertEqual(EXPECTED_MODES, {"engram_baseline", *ast.literal_eval(assignment.value)})


if __name__ == "__main__":
    unittest.main()
