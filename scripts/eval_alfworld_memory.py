"""Evaluate vanilla or ATHENA memory-augmented models in ALFWorld.

The task-facing metrics are ALFWorld episode success rate (SR), taken from
``infos["won"]``, and goal-condition success rate (GC-SR) when the selected
environment exposes ``infos["goal_condition_success_rate"]``.  The text-only
``AlfredTWEnv`` shipped by ALFWorld normally exposes SR but not GC-SR; in that
case GC-SR is recorded as ``null`` with an explicit availability note rather
than silently substituting SR.  ALFWorld has no reference answer string, so
``em`` and ``f1`` remain null.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.eval_openqa import get_model_max_context, greedy_generate, setup_condition


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", required=True)
    parser.add_argument(
        "--condition",
        choices=["baseline", "transferred"],
        default="transferred",
        help="baseline loads only the target backbone; transferred loads ATHENA memory readers",
    )
    parser.add_argument("--adaptor-dir", default=None)
    parser.add_argument("--adaptor-checkpoint", default=None)
    parser.add_argument("--source-memory", default=None)
    parser.add_argument("--memory-config", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--alfworld-data", required=True)
    parser.add_argument(
        "--split",
        choices=["valid_seen", "valid_unseen"],
        default="valid_unseen",
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--max-context-length", type=int, default=None)
    parser.add_argument("--history-turns", type=int, default=6)
    parser.add_argument("--reasoning-mode", choices=["vanilla", "cot"], default="vanilla")
    parser.add_argument(
        "--dual-reader-mode",
        choices=["both", "engram_only", "generated_only"],
        default="both",
    )
    parser.add_argument("--canon-mode", choices=["vocab", "word_boundary"], default="word_boundary")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_alfworld_config(data_root: str, split: str, max_episodes: int | None, max_steps: int):
    from alfworld.info import ALFRED_PDDL_PATH, ALFRED_TWL2_PATH

    data_root = str(Path(data_root).resolve())
    split_path = str(Path(data_root) / "json_2.1.1" / split)
    if not Path(split_path).is_dir():
        raise FileNotFoundError(f"Missing ALFWorld split directory: {split_path}")
    return {
        "dataset": {
            "data_path": str(Path(data_root) / "json_2.1.1" / "train"),
            "eval_id_data_path": str(Path(data_root) / "json_2.1.1" / "valid_seen"),
            "eval_ood_data_path": split_path,
            "num_train_games": -1,
            "num_eval_games": -1 if max_episodes is None else int(max_episodes),
        },
        "logic": {
            "domain": ALFRED_PDDL_PATH,
            "grammar": ALFRED_TWL2_PATH,
        },
        "env": {
            "type": "AlfredTWEnv",
            "domain_randomization": False,
            "task_types": [1, 2, 3, 4, 5, 6],
            "expert_timeout_steps": 150,
            "expert_type": "handcoded",
            "goal_desc_human_anns_prob": 0.0,
        },
        "general": {"training_method": "dagger"},
        "dagger": {"training": {"max_nb_steps_per_episode": int(max_steps)}},
    }


def normalize_action(text: str) -> str:
    text = text.strip()
    tagged = re.findall(r"<action>(.*?)</action>", text, flags=re.I | re.S)
    if tagged:
        text = tagged[-1]
    text = text.splitlines()[0] if text else ""
    text = re.sub(r"^(?:action|command)\s*:\s*", "", text, flags=re.I)
    text = text.strip().strip("`\"'")
    return re.sub(r"\s+", " ", text).lower()


def choose_admissible_action(generated: str, candidates: list[str]) -> tuple[str, bool]:
    """Map free generation to an official admissible command deterministically."""
    if not candidates:
        return "look", False
    normalized = normalize_action(generated)
    candidate_map = {normalize_action(candidate): candidate for candidate in candidates}
    if normalized in candidate_map:
        return candidate_map[normalized], True

    contained = [
        (key, value)
        for key, value in candidate_map.items()
        if key and (key in normalized or normalized in key)
    ]
    if contained:
        key, value = max(contained, key=lambda item: len(item[0]))
        return value, False

    best_key = max(
        candidate_map,
        key=lambda key: difflib.SequenceMatcher(None, normalized, key).ratio(),
    )
    return candidate_map[best_key], False


def build_action_prompt(
    initial_observation: str,
    history: list[tuple[str, str]],
    current_observation: str,
    candidates: list[str],
    history_turns: int,
    reasoning_mode: str = "vanilla",
) -> str:
    recent = history[-history_turns:]
    transcript = []
    for action, observation in recent:
        transcript.append(f"Action: {action}\nObservation: {observation}")
    command_list = "\n".join(f"- {command}" for command in candidates)
    reasoning_hint = (
        "Think through the next action step by step internally, then output only the command. "
        if reasoning_mode == "cot" else ""
    )
    return (
        "You are controlling an agent in ALFWorld. Complete the household task. "
        + reasoning_hint
        + "Choose exactly one command from the admissible command list and output only that command.\n\n"
        f"Initial task and observation:\n{initial_observation}\n\n"
        + ("Recent trajectory:\n" + "\n".join(transcript) + "\n\n" if transcript else "")
        + f"Current observation:\n{current_observation}\n\n"
        + f"Admissible commands:\n{command_list}\n\nAction:"
    )


def evaluate_environment(env, wrapper, set_canon_fn, device, args, num_games: int):
    tokenizer = wrapper.tokenizer
    max_context = get_model_max_context(wrapper, args.max_context_length)
    episodes = []
    total_success = 0.0
    total_steps = 0
    exact_action_count = 0
    action_count = 0

    env.seed(args.seed)
    for episode_index in range(num_games):
        observations, infos = env.reset()
        observation = str(observations[0])
        initial_observation = observation
        gamefile = str(infos.get("extra.gamefile", [""])[0])
        history: list[tuple[str, str]] = []
        success = 0.0
        goal_condition_values: list[float] = []
        episode_actions = []
        started = time.time()

        for step_index in range(args.max_steps):
            candidates = [str(command) for command in infos["admissible_commands"][0]]
            prompt = build_action_prompt(
                initial_observation,
                history,
                observation,
                candidates,
                args.history_turns,
                args.reasoning_mode,
            )
            generated = greedy_generate(
                wrapper,
                tokenizer,
                prompt,
                device,
                set_canon_fn,
                max_new_tokens=args.max_new_tokens,
                max_context_length=max_context,
                official_tokenization=True,
                stop_at_newline=True,
            )
            action, exact_action = choose_admissible_action(generated, candidates)
            next_observations, _, dones, infos = env.step([action])
            next_observation = str(next_observations[0])
            success = float(infos["won"][0])
            raw_gc = infos.get("goal_condition_success_rate")
            if raw_gc is not None:
                try:
                    goal_condition_values.append(float(raw_gc[0]))
                except (TypeError, IndexError, ValueError):
                    pass
            history.append((action, next_observation))
            episode_actions.append({
                "step": step_index + 1,
                "generated": generated,
                "action": action,
                "exact_admissible_match": exact_action,
                "observation": next_observation,
            })
            exact_action_count += int(exact_action)
            action_count += 1
            total_steps += 1
            observation = next_observation
            if bool(dones[0]):
                break

        total_success += success
        episode_gc = max(goal_condition_values) if goal_condition_values else None
        record = {
            "episode": episode_index,
            "gamefile": gamefile,
            "success": success,
            "goal_condition_success_rate": episode_gc,
            "steps": len(episode_actions),
            "elapsed_s": time.time() - started,
            "actions": episode_actions,
        }
        episodes.append(record)
        print(
            f"ALFWORLD_EPISODE_COMPLETE {episode_index + 1}/{num_games} "
            f"success={success:.0f} steps={len(episode_actions)}"
        )

    return {
        "acc": total_success / max(num_games, 1),
        "success_rate": total_success / max(num_games, 1),
        "goal_condition_success_rate": (
            float(np.mean([
                episode["goal_condition_success_rate"]
                for episode in episodes
                if episode["goal_condition_success_rate"] is not None
            ]))
            if any(episode["goal_condition_success_rate"] is not None for episode in episodes)
            else None
        ),
        "goal_condition_metric_note": (
            "mean over episodes of the maximum environment-reported "
            "goal_condition_success_rate"
            if any(episode["goal_condition_success_rate"] is not None for episode in episodes)
            else "unavailable: text-only AlfredTWEnv did not expose goal_condition_success_rate"
        ),
        "em": None,
        "f1": None,
        "n_examples": num_games,
        "successful_episodes": int(total_success),
        "total_steps": total_steps,
        "exact_admissible_action_rate": exact_action_count / max(action_count, 1),
        "episodes": episodes,
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ["ALFWORLD_DATA"] = str(Path(args.alfworld_data).resolve())

    output_dir = Path(args.output_dir)
    result_path = output_dir / "results.json"
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing existing output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    wrapper, set_canon_fn = setup_condition(args, args.condition, device, dtype)
    wrapper.eval()

    from alfworld.agents.environment import get_environment

    config = build_alfworld_config(
        args.alfworld_data, args.split, args.max_episodes, args.max_steps
    )
    manager = get_environment("AlfredTWEnv")(
        config,
        train_eval=("eval_in_distribution" if args.split == "valid_seen" else "eval_out_of_distribution"),
    )
    env = manager.init_env(batch_size=1)
    num_games = manager.num_games
    if args.max_episodes is not None:
        num_games = min(num_games, args.max_episodes)

    try:
        metrics = evaluate_environment(env, wrapper, set_canon_fn, device, args, num_games)
    finally:
        env.close()
        wrapper.cleanup()

    result = {
        "completed": True,
        "task": "alfworld",
        "target_model": args.target_model,
        "condition": args.condition,
        "architecture": (
            "vanilla_backbone"
            if args.condition == "baseline"
            else "generated_memory+engram+dual_reader"
        ),
        "dual_reader_mode": (
            None if args.condition == "baseline" else args.dual_reader_mode
        ),
        "reasoning_mode": args.reasoning_mode,
        "dataset": {
            "name": "ALFWorld",
            "environment": "AlfredTWEnv",
            "split": args.split,
            "data_root": str(Path(args.alfworld_data).resolve()),
        },
        "protocol": {
            "action_selection": "generate_then_map_to_admissible_command",
            "max_steps": args.max_steps,
            "history_turns": args.history_turns,
            "em_f1_applicable": False,
        },
        "metrics": metrics,
    }
    result_path.write_text(json.dumps(result, indent=2))
    print("ALFWORLD_MEMORY_EVAL_COMPLETE " + json.dumps({
        "acc": metrics["acc"],
        "success_rate": metrics["success_rate"],
        "goal_condition_success_rate": metrics["goal_condition_success_rate"],
        "em": None,
        "f1": None,
        "n_examples": metrics["n_examples"],
    }))


if __name__ == "__main__":
    main()
