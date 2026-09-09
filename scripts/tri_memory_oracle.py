"""Gold-aware diagnostic oracles for E, GE, and GH predictions.

These functions are evaluation-only.  They never expose benchmark labels to
training or to the learned Reader.
"""

from __future__ import annotations

import numpy as np


SOURCE_MODES = ("tri_E", "tri_GE", "tri_GH")

# Auxiliary ceiling for a Reader that can only choose one source.  This is
# useful for quantifying the value of a hard three-way source classifier, but
# it is not the full upper bound of the user's one/two/three-expert design.
ORACLE_SOURCE_SETS = {
    "oracle_source_E_GE": ("tri_E", "tri_GE"),
    "oracle_source_E_GH": ("tri_E", "tri_GH"),
    "oracle_source_GE_GH": ("tri_GE", "tri_GH"),
    "oracle_source_E_GE_GH": SOURCE_MODES,
}

# The theoretical ceiling for the user's seven-subset design.  Each pair
# oracle compares the two singleton endpoints and its pair endpoint.  The
# three-way oracle compares exactly the seven fixed subset endpoints:
#
#   E, GE, GH, E+GE, E+GH, GE+GH, and E+GE+GH.
#
# ``tri_subset_reader_hard`` and ``tri_subset_reader_soft`` are learned
# runtime models, not oracle endpoints.  Including either one here would make
# the supposed upper bound depend on the model whose capture is being
# measured.  ``tri_reader_soft`` is the fixed all-three endpoint (the hard
# source Reader selects one source and is therefore not an E+GE+GH endpoint).
ORACLE_SUBSET_SETS = {
    "oracle_E_GE": ("tri_E", "tri_GE", "tri_E_GE"),
    "oracle_E_GH": ("tri_E", "tri_GH", "tri_E_GH"),
    "oracle_GE_GH": ("tri_GE", "tri_GH", "tri_GE_GH"),
    "oracle_E_GE_GH": (
        "tri_E",
        "tri_GE",
        "tri_GH",
        "tri_E_GE",
        "tri_E_GH",
        "tri_GE_GH",
        "tri_reader_soft",
    ),
    # Explicit name for downstream analysis and paper tables.
    "oracle_all_nonempty_subsets": (
        "tri_E",
        "tri_GE",
        "tri_GH",
        "tri_E_GE",
        "tri_E_GH",
        "tri_GE_GH",
        "tri_reader_soft",
    ),
}


def _metric_spec(task: str) -> tuple[str, tuple[tuple[str, str], ...]]:
    if task == "truthfulqa":
        return "mc_avg", (("mc1", "MC1"), ("mc2", "MC2"), ("mc3", "MC3"))
    return "f1", (("em", "correct"), ("f1", "f1"))


def _rows(task: str, metrics: dict) -> list[dict]:
    return metrics["sample_examples" if task == "truthfulqa" else "sample_predictions"]


def _check_sources(task: str, source_metrics: dict[str, dict]) -> list[str]:
    names = list(source_metrics)
    if not names:
        raise ValueError("At least one oracle source is required")
    rows = [_rows(task, source_metrics[name]) for name in names]
    if not rows[0]:
        raise ValueError("Cannot construct an oracle from empty predictions")
    expected_questions = [row["question"] for row in rows[0]]
    for name, candidate_rows in zip(names[1:], rows[1:]):
        questions = [row["question"] for row in candidate_rows]
        if questions != expected_questions:
            raise ValueError(f"Oracle source {name} is not paired in question order")
    return names


def build_oracle_metrics(task: str, source_metrics: dict[str, dict]) -> dict:
    """Select one complete source per example using the task's primary metric."""
    names = _check_sources(task, source_metrics)
    primary_metric, metric_spec = _metric_spec(task)
    source_rows = [_rows(task, source_metrics[name]) for name in names]

    arrays = {}
    for metric_name, row_key in metric_spec:
        if task == "truthfulqa":
            arrays[metric_name] = np.asarray(
                [
                    [float(row["metrics"][row_key]) for row in rows]
                    for rows in source_rows
                ],
                dtype=np.float64,
            )
        else:
            arrays[metric_name] = np.asarray(
                [[float(row[row_key]) for row in rows] for rows in source_rows],
                dtype=np.float64,
            )
    if task == "truthfulqa":
        arrays["mc_avg"] = np.stack(
            [arrays["mc1"], arrays["mc2"], arrays["mc3"]], axis=-1
        ).mean(axis=-1)

    selected = arrays[primary_metric].argmax(axis=0)
    columns = np.arange(selected.size)
    selected_arrays = {
        metric: values[selected, columns] for metric, values in arrays.items()
    }
    metricwise_upper = {
        metric: float(values.max(axis=0).mean()) for metric, values in arrays.items()
    }
    selection_counts = {
        name: int((selected == source_index).sum())
        for source_index, name in enumerate(names)
    }

    selected_rows = [
        source_rows[int(source_index)][example_index]
        for example_index, source_index in enumerate(selected)
    ]
    if task == "truthfulqa":
        sample_examples = []
        for row, mc1, mc2, mc3 in zip(
            selected_rows,
            selected_arrays["mc1"],
            selected_arrays["mc2"],
            selected_arrays["mc3"],
        ):
            copied = dict(row)
            copied["metrics"] = dict(row["metrics"])
            copied["metrics"].update(
                {"MC1": float(mc1), "MC2": float(mc2), "MC3": float(mc3)}
            )
            sample_examples.append(copied)
        return {
            "acc": float(selected_arrays["mc1"].mean()),
            "mc1": float(selected_arrays["mc1"].mean()),
            "mc2": float(selected_arrays["mc2"].mean()),
            "mc3": float(selected_arrays["mc3"].mean()),
            "mc_avg": float(selected_arrays["mc_avg"].mean()),
            "total": int(selected.size),
            "sample_examples": sample_examples,
            "selection_metric": primary_metric,
            "selection_counts": selection_counts,
            "metricwise_upper": metricwise_upper,
        }

    sample_predictions = []
    for row, em, f1 in zip(
        selected_rows, selected_arrays["em"], selected_arrays["f1"]
    ):
        copied = dict(row)
        copied["correct"] = bool(em)
        copied["f1"] = float(f1)
        sample_predictions.append(copied)
    return {
        "acc": float(selected_arrays["em"].mean()),
        "em": float(selected_arrays["em"].mean()),
        "f1": float(selected_arrays["f1"].mean()),
        "correct": int(selected_arrays["em"].sum()),
        "total": int(selected.size),
        "sample_predictions": sample_predictions,
        "selection_metric": primary_metric,
        "selection_counts": selection_counts,
        "metricwise_upper": metricwise_upper,
    }


def build_all_oracles(task: str, evaluation: dict[str, dict]) -> dict[str, dict]:
    oracles = {
        oracle_name: build_oracle_metrics(
            task,
            {
                source: evaluation[source][task]["metrics"]
                for source in source_names
            },
        )
        for oracle_name, source_names in ORACLE_SOURCE_SETS.items()
    }
    for oracle_name, source_names in ORACLE_SUBSET_SETS.items():
        missing = [source for source in source_names if source not in evaluation]
        if missing:
            raise ValueError(
                f"Runtime oracle requires evaluated paths {missing}; "
                "run the complete tri-reader evaluation matrix"
            )
        oracles[oracle_name] = build_oracle_metrics(
            task,
            {
                source: evaluation[source][task]["metrics"]
                for source in source_names
            },
        )
    return oracles


def capture_ratio(reader: float, baseline: float, oracle: float) -> float | None:
    available = oracle - baseline
    if available <= 0:
        return None
    return float((reader - baseline) / available)
