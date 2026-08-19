from scripts.eval_alfworld_memory import (
    build_action_prompt,
    choose_admissible_action,
    normalize_action,
)


def test_normalize_action_strips_wrapper_and_prefix():
    assert normalize_action("<action>Action: Open fridge 1</action>\nextra") == "open fridge 1"


def test_choose_admissible_action_exact_match():
    action, exact = choose_admissible_action(
        "Action: take apple 1 from table 1",
        ["look", "take apple 1 from table 1"],
    )
    assert action == "take apple 1 from table 1"
    assert exact is True


def test_choose_admissible_action_fuzzy_fallback_is_valid():
    candidates = ["look", "go to fridge 1", "open fridge 1"]
    action, exact = choose_admissible_action("open the fridge", candidates)
    assert action in candidates
    assert exact is False


def test_action_prompt_includes_only_recent_history():
    prompt = build_action_prompt(
        "Your task is to cool an apple.",
        [("look", "obs0"), ("go to fridge", "obs1")],
        "obs2",
        ["look", "open fridge"],
        history_turns=1,
    )
    assert "obs0" not in prompt
    assert "go to fridge" in prompt
    assert "open fridge" in prompt
