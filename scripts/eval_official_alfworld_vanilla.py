#!/usr/bin/env python3
"""Run the public-protocol vanilla baseline on ALFWorld.

The MemGen repository does not publish its ALFWorld wrapper.  This evaluator
therefore keeps the public model-generation convention (one chat turn, greedy
decoding) and uses the official ALFWorld ``AlfredTWEnv`` on ``valid_unseen``.
The reported task metric is the environment's episode success rate.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--alfworld-data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", choices=["valid_seen", "valid_unseen"], default="valid_unseen")
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--history-turns", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--official-chat-template", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_alfworld_config(data_root: Path, split: str, max_episodes: int, max_steps: int) -> dict:
    from alfworld.info import ALFRED_PDDL_PATH, ALFRED_TWL2_PATH

    split_path = data_root / "json_2.1.1" / split
    if not split_path.is_dir():
        raise FileNotFoundError(f"Missing ALFWorld split directory: {split_path}")
    return {
        "dataset": {
            "data_path": str(data_root / "json_2.1.1" / "train"),
            "eval_id_data_path": str(data_root / "json_2.1.1" / "valid_seen"),
            "eval_ood_data_path": str(split_path),
            "num_train_games": -1,
            "num_eval_games": -1 if max_episodes <= 0 else max_episodes,
        },
        "logic": {"domain": ALFRED_PDDL_PATH, "grammar": ALFRED_TWL2_PATH},
        "env": {
            "type": "AlfredTWEnv",
            "domain_randomization": False,
            "task_types": [1, 2, 3, 4, 5, 6],
            "expert_timeout_steps": 150,
            "expert_type": "handcoded",
            "goal_desc_human_anns_prob": 0.0,
        },
        "general": {"training_method": "dagger"},
        "dagger": {"training": {"max_nb_steps_per_episode": max_steps}},
    }


def normalize_action(text: str) -> str:
    """Normalize only presentation wrappers; do not fuzzy-correct actions."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.S)
    tagged = re.findall(r"<action>(.*?)</action>", text, flags=re.I | re.S)
    if tagged:
        text = tagged[-1]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text = lines[0] if lines else ""
    text = re.sub(r"^(?:action|command)\s*:\s*", "", text, flags=re.I)
    return re.sub(r"\s+", " ", text.strip().strip("`\"'"))


def select_official_action(generated: str, candidates: list[str]) -> tuple[str, bool]:
    """Return an exact admissible command when possible.

    ALFWorld itself remains the source of truth.  Invalid generations are sent
    to the environment unchanged instead of being repaired with fuzzy matching
    or a hidden oracle policy.
    """
    normalized = normalize_action(generated)
    by_lower = {candidate.strip().lower(): candidate for candidate in candidates}
    action = by_lower.get(normalized.lower())
    if action is not None:
        return action, True
    return normalized, False


def build_action_prompt(
    initial_observation: str,
    history: list[tuple[str, str]],
    observation: str,
    candidates: list[str],
    history_turns: int,
) -> str:
    trajectory = history[-history_turns:]
    transcript = "\n".join(
        f"Action: {action}\nObservation: {next_observation}"
        for action, next_observation in trajectory
    )
    commands = "\n".join(f"- {command}" for command in candidates)
    return (
        "You are an agent interacting with ALFWorld. Complete the household "
        "task. Output exactly one admissible command and nothing else.\n\n"
        f"Task and initial observation:\n{initial_observation}\n\n"
        + (f"Recent trajectory:\n{transcript}\n\n" if transcript else "")
        + f"Current observation:\n{observation}\n\n"
        + f"Admissible commands:\n{commands}\n\nAction:"
    )


def generate_action(model, tokenizer, prompt: str, generation_config: GenerationConfig) -> str:
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {key: value.cuda() for key, value in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(**inputs, generation_config=generation_config)
    prompt_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=True)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.alfworld_data = args.alfworld_data.resolve()
    os.environ["ALFWORLD_DATA"] = str(args.alfworld_data)

    config = build_alfworld_config(
        args.alfworld_data, args.split, args.max_episodes, args.max_steps
    )
    from alfworld.agents.environment import get_environment

    manager = get_environment("AlfredTWEnv")(
        config,
        train_eval=("eval_in_distribution" if args.split == "valid_seen" else "eval_out_of_distribution"),
    )
    env = manager.init_env(batch_size=1)
    full_count = manager.num_games
    count = min(full_count, args.max_episodes) if args.max_episodes > 0 else full_count
    print(f"OFFICIAL_ALFWORLD_DATASET split={args.split} full={full_count} evaluated={count}", flush=True)
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "model": args.model,
        "dataset": "ALFWorld",
        "environment": "AlfredTWEnv",
        "split": args.split,
        "evaluation_count": count,
        "max_steps": args.max_steps,
        "history_turns": args.history_turns,
        "max_new_tokens": args.max_new_tokens,
        "official_chat_template": args.official_chat_template,
        "protocol": "official ALFWorld valid split + greedy one-command generation",
        "metric": "episode success rate from infos['won']",
        "action_postprocessing": "exact admissible-command match; no fuzzy correction",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))

    if args.dry_run:
        env.close()
        print("OFFICIAL_ALFWORLD_DRY_RUN_COMPLETE", flush=True)
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.official_chat_template:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "official" / "MemGen"))
        from memgen.utils import CONVERSATION_TEMPLATE

        tokenizer.chat_template = CONVERSATION_TEMPLATE
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).cuda().eval()
    generation_config = GenerationConfig(
        do_sample=False,
        use_cache=False,
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    episodes_path = args.output / "episodes.jsonl"
    total_success = 0.0
    total_steps = 0
    exact_actions = 0
    action_count = 0
    started = time.time()
    try:
        env.seed(42)
        with episodes_path.open("x") as writer:
            for episode_index in range(count):
                observations, infos = env.reset()
                observation = str(observations[0])
                initial_observation = observation
                history: list[tuple[str, str]] = []
                actions = []
                success = 0.0
                gamefile = str(infos.get("extra.gamefile", [""])[0])
                for step_index in range(args.max_steps):
                    candidates = [str(item) for item in infos["admissible_commands"][0]]
                    prompt = build_action_prompt(
                        initial_observation,
                        history,
                        observation,
                        candidates,
                        args.history_turns,
                    )
                    raw = generate_action(model, tokenizer, prompt, generation_config)
                    action, exact = select_official_action(raw, candidates)
                    next_observations, _, dones, infos = env.step([action])
                    next_observation = str(next_observations[0])
                    success = float(infos["won"][0])
                    actions.append({
                        "step": step_index + 1,
                        "raw_generation": raw,
                        "action": action,
                        "exact_admissible_match": exact,
                        "observation": next_observation,
                    })
                    history.append((action, next_observation))
                    observation = next_observation
                    total_steps += 1
                    action_count += 1
                    exact_actions += int(exact)
                    if bool(dones[0]):
                        break
                total_success += success
                writer.write(json.dumps({
                    "episode": episode_index,
                    "gamefile": gamefile,
                    "success": success,
                    "steps": len(actions),
                    "actions": actions,
                }, ensure_ascii=False) + "\n")
                writer.flush()
                print(
                    f"OFFICIAL_ALFWORLD_PROGRESS {episode_index + 1}/{count} "
                    f"success={success:.0f} steps={len(actions)}",
                    flush=True,
                )
    finally:
        env.close()
        del model
        torch.cuda.empty_cache()

    summary = {
        **metadata,
        "completed": True,
        "accuracy": total_success / count if count else 0.0,
        "successful_episodes": int(total_success),
        "total_steps": total_steps,
        "exact_admissible_action_rate": exact_actions / action_count if action_count else 0.0,
        "elapsed_s": time.time() - started,
        "episodes_file": str(episodes_path),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print("OFFICIAL_ALFWORLD_COMPLETE " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
