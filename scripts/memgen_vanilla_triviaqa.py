#!/usr/bin/env python3
"""Paper-aligned MemGen vanilla evaluation on TriviaQA.

This reproduces the dynamic TriviaQA environment in KANABOON1/MemGen while
loading only the unmodified Hugging Face reasoner.  In particular, ``<search>``
actions are executed against the Search-R1 retriever and returned to the model
as the next user turn.  No weaver, trigger, adaptor, LoRA, or latent memory is
loaded.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = """Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>."""

INVALID_ACTION_OBSERVATION = """\nMy previous action is invalid. If I want to search, I should put the query between <search> and </search>. If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--retriever-url", default="http://127.0.0.1:8001/retrieve")
    parser.add_argument("--retriever-topk", type=int, default=3)
    parser.add_argument("--retriever-timeout", type=float, default=60.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--max-response-length", type=int, default=1024)
    parser.add_argument("--max-observation-length", type=int, default=512)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    return parser.parse_args()


def preprocess_action(action: str) -> str:
    """Match ``TriviaQAEnv.preprocess_action`` from the official repository."""
    if "</search>" in action:
        return action.split("</search>", 1)[0] + "</search>"
    if "</answer>" in action:
        return action.split("</answer>", 1)[0] + "</answer>"
    return action


def process_action(action: str) -> tuple[str, str]:
    """Return the official environment action type and first-line payload."""
    action = action.strip()
    if "<search>" in action and "</search>" in action:
        content = action.split("<search>", 1)[1].split("</search>", 1)[0]
        return "search", content.strip().split("\n", 1)[0].strip()
    if "<answer>" in action and "</answer>" in action:
        content = action.split("<answer>", 1)[1].split("</answer>", 1)[0]
        return "answer", content.strip().split("\n", 1)[0].strip()
    return "think", action


def answer_is_correct(answer: str, aliases: list[str]) -> bool:
    lowered = answer.lower()
    return any(alias.lower() in lowered for alias in aliases)


def format_retrieval_result(result: list[dict[str, Any]]) -> str:
    """Match Search-R1 passage formatting used by MemGen's ``Retriever``."""
    formatted = ""
    for index, item in enumerate(result):
        contents = item["document"]["contents"]
        title, _, body = contents.partition("\n")
        formatted += f"Doc {index + 1}(Title: {title}) {body}\n"
    return formatted


def batch_search(
    url: str,
    queries: list[str],
    *,
    topk: int,
    timeout: float,
) -> list[str]:
    headers = {"Content-Type": "application/json"}
    service_token = os.environ.get("ATHENA_SERVICE_TOKEN")
    if service_token:
        headers["X-ATHENA-Service-Token"] = service_token
    access_client_id = os.environ.get("CF_ACCESS_CLIENT_ID")
    access_client_secret = os.environ.get("CF_ACCESS_CLIENT_SECRET")
    if access_client_id and access_client_secret:
        headers["CF-Access-Client-Id"] = access_client_id
        headers["CF-Access-Client-Secret"] = access_client_secret
    request = urllib.request.Request(
        url,
        data=json.dumps({"queries": queries, "topk": topk, "return_scores": True}).encode(),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode())
    except Exception as exc:
        raise RuntimeError(f"Retriever request failed for {url}: {exc}") from exc
    results = payload["result"]
    if len(results) != len(queries):
        raise RuntimeError(
            f"Retriever returned {len(results)} result sets for {len(queries)} queries"
        )
    return [format_retrieval_result(result) for result in results]


def truncate_observation(tokenizer, observation: str, max_tokens: int) -> str:
    token_ids = tokenizer.encode(observation, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return observation
    ellipsis_ids = tokenizer.encode("...", add_special_tokens=False)
    kept = token_ids[: max_tokens - len(ellipsis_ids)] + ellipsis_ids
    return tokenizer.decode(kept, skip_special_tokens=True)


def retriever_preflight(args) -> None:
    observations = batch_search(
        args.retriever_url,
        ["Mount Kilimanjaro"],
        topk=args.retriever_topk,
        timeout=args.retriever_timeout,
    )
    if not observations or not observations[0].startswith("Doc 1(Title:"):
        raise RuntimeError("Search-R1 retriever returned an unexpected response")
    print("RETRIEVER_READY", args.retriever_url, flush=True)


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    args = parse_args()
    retriever_preflight(args)
    device = select_device(args.device)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "answer.jsonl"
    summary_path = output_dir / "summary.json"
    if result_path.exists() or summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing results in {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = torch.bfloat16 if device.type in {"cuda", "mps"} else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, low_cpu_mem_usage=True
    ).to(device).eval()
    print("DEVICE", device.type, flush=True)
    if device.type == "cuda":
        print("GPU", torch.cuda.get_device_name(0), flush=True)
    print("MODEL_LOADED", args.model, flush=True)

    dataset = load_dataset(
        "mandarjoshi/trivia_qa", "rc.wikipedia.nocontext", split="validation"
    )
    if args.max_examples:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))
    print("DATASET_SIZE", len(dataset), flush=True)

    correct = total = 0
    with result_path.open("x") as writer, torch.inference_mode():
        for start in range(0, len(dataset), args.batch_size):
            batch = dataset[start : start + args.batch_size]
            histories = [
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ]
                for question in batch["question"]
            ]
            aliases = [answer["normalized_aliases"] for answer in batch["answer"]]
            final_answers = [""] * len(histories)
            done = [False] * len(histories)

            for _turn in range(args.max_turns):
                active_indices = [index for index, finished in enumerate(done) if not finished]
                if not active_indices:
                    break
                active_histories = [histories[index] for index in active_indices]
                inputs = tokenizer.apply_chat_template(
                    active_histories,
                    tokenize=True,
                    add_generation_prompt=True,
                    padding=True,
                    return_tensors="pt",
                    return_dict=True,
                ).to(device)
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_response_length,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                responses = tokenizer.batch_decode(
                    generated[:, inputs["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                )
                actions = [preprocess_action(response) for response in responses]
                parsed = [process_action(action) for action in actions]

                search_positions = [i for i, (kind, _content) in enumerate(parsed) if kind == "search"]
                search_observations: dict[int, str] = {}
                if search_positions:
                    queries = [parsed[i][1] for i in search_positions]
                    retrieved = batch_search(
                        args.retriever_url,
                        queries,
                        topk=args.retriever_topk,
                        timeout=args.retriever_timeout,
                    )
                    search_observations = dict(zip(search_positions, retrieved))

                for active_position, dataset_index in enumerate(active_indices):
                    action = actions[active_position]
                    action_type, content = parsed[active_position]
                    histories[dataset_index].append({"role": "assistant", "content": action})
                    if action_type == "answer":
                        final_answers[dataset_index] = content
                        done[dataset_index] = True
                        observation = ""
                    elif action_type == "search":
                        observation = search_observations[active_position]
                    else:
                        observation = INVALID_ACTION_OBSERVATION
                    observation = truncate_observation(
                        tokenizer, observation, args.max_observation_length
                    )
                    histories[dataset_index].append({"role": "user", "content": observation})

            for question, answer_aliases, answer, history in zip(
                batch["question"], aliases, final_answers, histories
            ):
                hit = bool(answer) and answer_is_correct(answer, answer_aliases)
                total += 1
                correct += int(hit)
                writer.write(
                    json.dumps(
                        {
                            "question": question,
                            "aliases": answer_aliases,
                            "answer": answer,
                            "correct": hit,
                            "history": history,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            writer.flush()
            print("PROGRESS", total, "ACCURACY", 100.0 * correct / total, flush=True)

    summary = {
        "model": args.model,
        "condition": "vanilla_backbone",
        "dataset": "mandarjoshi/trivia_qa:rc.wikipedia.nocontext:validation",
        "protocol": "KANABOON1/MemGen dynamic TriviaQA environment",
        "retriever_url": args.retriever_url,
        "retriever_topk": args.retriever_topk,
        "max_turns": args.max_turns,
        "max_response_length": args.max_response_length,
        "max_observation_length": args.max_observation_length,
        "total": total,
        "correct": correct,
        "accuracy": 100.0 * correct / total,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print("VANILLA_TRIVIAQA_COMPLETE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
