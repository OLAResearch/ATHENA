from pathlib import Path

from scripts.eval_general_paper_aligned import (
    _load_agn,
    _load_cb,
    _load_hyp,
    _load_rt,
    _load_rte,
    _load_sentiment_csv,
    _load_sst2,
)
from scripts.eval_vanilla_general_paper_protocol import load_tasks


ROOT = Path(__file__).resolve().parents[1]


def test_public_prompts_keep_exact_leading_verbalizer_spacing():
    cb = _load_cb(ROOT / "run/vanilla_cb_agn_protocol_20260918/task_data/cb/dev.jsonl")
    rte = _load_rte(ROOT / "run/vanilla_rte_yahoo_protocol_20260918/task_data/rte/val.jsonl")
    hyp = _load_hyp(ROOT / "run/vanilla_general_protocol_20260918r2/task_data/hyp/test.csv")
    assert cb[0]["context"].endswith("answer: The hypothesis is")
    assert cb[0]["domain_context"] == "answer: The hypothesis is"
    assert all(choice.startswith(" ") for choice in cb[0]["choices"])
    assert rte[0]["domain_context"] == "true or false?\nanswer:"
    assert hyp[0]["domain_context"].startswith("\n ")


def test_paper_prompts_and_synonym_protocol_are_exact():
    data = ROOT / "run/vanilla_general_protocol_20260918r2/task_data"
    sst2 = _load_sst2(data / "sst2/dev.tsv")
    rt = _load_rt(data / "rotten_tomatoes/test.jsonl")
    agn = _load_agn(ROOT / "run/vanilla_cb_agn_protocol_20260918/task_data/agn/test.csv")
    assert sst2[0]["context"].endswith(" The sentence has a tone that is")
    assert rt[0]["context"].endswith(" It is")
    assert agn[0]["context"].startswith("title: ")
    assert agn[0]["context"].endswith("\ntopic:")
    assert "label_synonyms" in sst2[0]


def test_aligned_task_sizes_match_published_suite_inputs():
    data = ROOT / "run/vanilla_general_protocol_20260918r2/task_data"
    tasks = {
        "sst2": _load_sst2(data / "sst2/dev.tsv"),
        "mr": _load_sentiment_csv(data / "mr/test.csv"),
        "cr": _load_sentiment_csv(data / "cr/test.csv"),
        "rt": _load_rt(data / "rotten_tomatoes/test.jsonl"),
        "hyp": _load_hyp(data / "hyp/test.csv"),
        "cb": _load_cb(ROOT / "run/vanilla_cb_agn_protocol_20260918/task_data/cb/dev.jsonl"),
        "rte": _load_rte(ROOT / "run/vanilla_rte_yahoo_protocol_20260918/task_data/rte/val.jsonl"),
        "agn": _load_agn(ROOT / "run/vanilla_cb_agn_protocol_20260918/task_data/agn/test.csv"),
    }
    assert {name: len(rows) for name, rows in tasks.items()} == {
        "sst2": 872,
        "mr": 2000,
        "cr": 2000,
        "rt": 1066,
        "hyp": 65,
        "cb": 56,
        "rte": 277,
        "agn": 7600,
    }


def test_vanilla_cr_uses_the_canonical_aligned_verbalizer():
    data = ROOT / "run/vanilla_general_protocol_20260918/task_data"
    tasks = load_tasks(data)
    assert tasks["cr"][0]["choices"] == ["terrible", "great"]
