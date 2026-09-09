import json
import sys
from types import SimpleNamespace

from scripts.eval_openqa import (
    _aggregate_parallel_results,
    _parallel_child_argv,
    resolve_reader_mode_contract,
)


def test_parallel_child_argv_isolates_task_and_runtime_controls(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_openqa.py",
            "--target-model",
            "model",
            "--tasks",
            "nq",
            "webqa",
            "--conditions",
            "baseline",
            "transferred",
            "--output-dir",
            "results",
            "--parallel-tasks",
            "5",
            "--task-devices",
            "0,1",
            "--max-examples",
            "2",
        ],
    )

    child = _parallel_child_argv("webqa", tmp_path / "webqa")

    assert child[child.index("--tasks") + 1] == "webqa"
    assert "nq" not in child
    assert child[child.index("--output-dir") + 1] == str(tmp_path / "webqa")
    assert child[child.index("--parallel-tasks") + 1] == "1"
    assert "--task-devices" not in child
    assert child[child.index("--max-examples") + 1] == "2"


def test_parallel_aggregation_preserves_task_order_and_scalar_summary(tmp_path):
    args = SimpleNamespace(
        tasks=["nq", "truthfulqa"],
        conditions=["baseline", "transferred"],
        target_model="model",
        seed=42,
        canon_mode="word_boundary",
        triviaqa_config="rc.nocontext",
        reasoning_mode="vanilla",
        adaptor_checkpoint=None,
        dual_reader_mode="both",
    )
    task_dirs = {}
    payloads = {
        "nq": {
            "baseline": {"f1": 0.2},
            "transferred": {"f1": 0.4},
        },
        "truthfulqa": {
            "baseline": {"mc_avg": 0.3},
            "transferred": {"mc_avg": 0.5},
        },
    }
    for task, metrics in payloads.items():
        task_dir = tmp_path / task
        task_dir.mkdir()
        task_dirs[task] = task_dir
        (task_dir / "openqa_results.json").write_text(
            json.dumps({"tasks": {task: metrics}})
        )

    result = _aggregate_parallel_results(args, task_dirs)

    assert list(result["tasks"]) == ["nq", "truthfulqa"]
    assert result["summary"]["baseline"]["per_task_scores"] == [20.0, 30.0]
    assert result["summary"]["transferred"]["per_task_scores"] == [40.0, 50.0]
    assert result["summary"]["delta"]["absolute"] == 20.0


def test_auto_reader_prefers_advantage_checkpoint_contract():
    result = resolve_reader_mode_contract(
        "auto",
        {
            "advantage_reader": {"enabled": True, "candidates": "sources"},
            "deployment_reader_mode": "tri_routed",
            "joint_tri_route_only": True,
        },
    )
    assert result["effective_reader_mode"] == "tri_advantage_routed"
    assert result["reader_mode_provenance"] == "checkpoint.advantage_reader"
