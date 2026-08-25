from scripts.memgen_vanilla_triviaqa import (
    answer_is_correct,
    format_retrieval_result,
    preprocess_action,
    process_action,
    select_device,
)


def test_preprocess_stops_at_first_closed_search_like_official_env():
    text = "<think>x</think><search>query</search>ignored<answer>wrong</answer>"
    assert preprocess_action(text) == "<think>x</think><search>query</search>"


def test_process_action_extracts_first_line():
    assert process_action("<search> query\nignored </search>") == ("search", "query")
    assert process_action("<answer> Beijing\nignored </answer>") == ("answer", "Beijing")
    assert process_action("still thinking") == ("think", "still thinking")


def test_alias_containment_matches_official_triviaqa_reward():
    aliases = ["mount kilimanjaro", "kilimanjaro"]
    assert answer_is_correct("The answer is Mount Kilimanjaro.", aliases)
    assert not answer_is_correct("Mount Kenya", aliases)


def test_search_r1_results_use_official_memgen_format():
    result = [
        {"document": {"contents": "Mount Kilimanjaro\nA dormant volcano in Tanzania."}},
        {"document": {"contents": "Kibo\nThe highest cone."}},
    ]
    assert format_retrieval_result(result) == (
        "Doc 1(Title: Mount Kilimanjaro) A dormant volcano in Tanzania.\n"
        "Doc 2(Title: Kibo) The highest cone.\n"
    )


def test_explicit_cpu_device_selection():
    assert select_device("cpu").type == "cpu"
