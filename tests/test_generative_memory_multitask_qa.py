from scripts.train_generative_memory_multitask_qa import (
    AnswerOnlyCollator,
    EVAL_TASKS,
    TRAIN_TASKS,
    extract_training_example,
)


class TinyTokenizer:
    eos_token = "<eos>"
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=True):
        prefix = [1] if add_special_tokens else []
        return {"input_ids": prefix + [ord(ch) + 2 for ch in text]}


def test_joint_task_sets_train_four_and_evaluate_five():
    assert TRAIN_TASKS == ("nq", "webqa", "triviaqa", "hotpotqa")
    assert EVAL_TASKS == ("nq", "webqa", "triviaqa", "truthfulqa", "hotpotqa")


def test_extract_training_examples_for_all_four_schemas():
    assert extract_training_example("nq", {"question": "Q", "answer": ["A"]})["answer"] == "A"
    assert extract_training_example("webqa", {"question": "Q", "answers": ["A", "B"]})["answers"] == ["A", "B"]
    trivia = {"question": "Q", "answer": {"value": "Best", "aliases": ["Alias"]}}
    assert extract_training_example("triviaqa", trivia)["answer"] == "Best"
    assert extract_training_example("hotpotqa", {"question": "Q", "answer": "yes"})["answer"] == "yes"
    assert extract_training_example("nq", {"question": "bad", "answer": [")"]}) is None


def test_answer_only_collator_masks_prompt_and_pads():
    batch = AnswerOnlyCollator(TinyTokenizer(), max_length=256)([
        {"question": "Short?", "answer": "A"},
        {"question": "Longer question?", "answer": "Answer"},
    ])
    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["input_ids"].shape == batch["attention_mask"].shape
    assert (batch["labels"] != -100).any(dim=1).all()
    assert (batch["labels"] == -100).any(dim=1).all()
