from scripts.eval_mlpmemory_external_qa import (
    EOSTokenCriteria,
    generate_memory_prompt,
    score_generate_memory_predictions,
)

import torch


def test_generate_memory_prompt_matches_athena_openqa_prompt():
    assert generate_memory_prompt("Who wrote Hamlet") == (
        "Answer these questions:\nQuestion: who wrote Hamlet?\nAnswer:"
    )


def test_generate_memory_metrics_use_normalized_em_and_best_answer_f1():
    examples = [
        {"question": "q1", "answers": ["The Eiffel Tower", "Eiffel"]},
        {"question": "q2", "answers": ["Barack Obama"]},
    ]
    metrics = score_generate_memory_predictions(examples, ["eiffel tower!", "Obama"])
    assert metrics["em"] == 0.5
    assert round(metrics["f1"], 4) == round((1.0 + 2 / 3) / 2, 4)
    assert metrics["correct"] == 1
    assert metrics["total"] == 2


def test_eos_criteria_tracks_each_sequence():
    criteria = EOSTokenCriteria(eos_token_id=2)
    stopped = criteria(torch.tensor([[10, 2], [11, 3]]), scores=None)
    assert stopped.tolist() == [True, False]
