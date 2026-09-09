import json
import sys
from types import SimpleNamespace

from scripts import eval_tri_memory_openqa_paired as paired
from scripts.eval_fair_joint_openqa_paired import _candidate_modes


class _FakeAdaptor:
    fusion_type = "tri_reader"

    def __init__(self, has_advantage):
        self.advantage_router = object() if has_advantage else None
        self.mode = "tri_soft_fused"

    def set_tri_reader_mode(self, mode):
        self.mode = mode


class _FakeWrapper:
    def __init__(self, has_advantage=False):
        self.adaptor = _FakeAdaptor(has_advantage)
        self.tokenizer = SimpleNamespace()

    def cleanup(self):
        pass


def _metrics(mode):
    score = {
        "engram_only": 0.10,
        "generated_from_engram_only": 0.20,
        "generated_from_context_only": 0.30,
        "e_ge": 0.25,
        "e_gh": 0.35,
        "ge_gh": 0.40,
        "tri_routed": 0.45,
        "tri_soft_fused": 0.50,
        "tri_advantage_routed": 0.60,
        "tri_subset_routed": 0.55,
        "tri_subset_soft_fused": 0.65,
    }[mode]
    rows = [
        {"question": "q0", "prediction": "a", "answers": ["a"], "correct": True, "f1": score},
        {"question": "q1", "prediction": "b", "answers": ["a"], "correct": False, "f1": score},
    ]
    return {
        "em": 0.5,
        "f1": score,
        "correct": 1,
        "total": 2,
        "sample_predictions": rows,
    }


def _run_paired(monkeypatch, tmp_path, *, has_advantage):
    def fake_setup(args, condition, device, dtype):
        if condition == "baseline":
            return _FakeWrapper(), None
        return _FakeWrapper(has_advantage), None

    def fake_evaluate(wrapper, _canon, task_data, args, _device):
        mode = wrapper.adaptor.mode
        return {task: {"metrics": _metrics(mode)} for task in args.tasks}

    monkeypatch.setattr(paired, "setup_condition", fake_setup)
    monkeypatch.setattr(paired, "_evaluate_tasks", fake_evaluate)
    monkeypatch.setattr(
        paired,
        "load_task",
        lambda task, _config: ([{"question": "q0"}, {"question": "q1"}], {"name": task}),
    )
    output_dir = tmp_path / ("new" if has_advantage else "old")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_tri_memory_openqa_paired.py",
            "--baseline-adaptor-dir",
            "baseline",
            "--tri-adaptor-dir",
            "tri",
            "--source-memory",
            "memory.pt",
            "--memory-config",
            "memory.json",
            "--output-dir",
            str(output_dir),
            "--tasks",
            "nq",
        ],
    )
    paired.main()
    return json.loads((output_dir / "results.json").read_text())


def test_paired_tri_evaluation_skips_advantage_for_legacy_checkpoint(monkeypatch, tmp_path):
    results = _run_paired(monkeypatch, tmp_path, has_advantage=False)
    assert "tri_reader_advantage" not in results["modes"]
    assert "tri_reader_advantage" not in results["evaluation"]


def test_paired_tri_evaluation_records_advantage_for_new_checkpoint(monkeypatch, tmp_path):
    results = _run_paired(monkeypatch, tmp_path, has_advantage=True)
    assert "tri_reader_advantage" in results["modes"]
    assert "tri_reader_advantage" in results["evaluation"]
    assert results["scalar_summary"]["tri_reader_advantage"]["nq"] == 60.0


def test_fair_joint_detects_legacy_tri_config_keys(tmp_path):
    adaptor_dir = tmp_path / "adaptor"
    adaptor_dir.mkdir()
    (adaptor_dir / "config.json").write_text(
        '{"fusion_type": "tri_reader", "advantage_reader": {"enabled": false}}'
    )
    modes, is_tri = _candidate_modes(str(adaptor_dir))
    assert is_tri is True
    assert "tri_reader_advantage" not in modes
