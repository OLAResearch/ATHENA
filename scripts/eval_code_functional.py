"""Compute functional accuracy for ATHENA code-generation outputs.

KodCode tests are executed in subprocesses with resource limits. BigCodeBench
is evaluated through its official local evaluator. The Slurm wrapper runs this
script inside a dedicated Apptainer process with networking disabled; this
module is not intended to make arbitrary generated code safe on its own.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import resource
import subprocess
import sys
import tempfile
import ast
import re
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["kodcode", "bigcodebench"], required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--results-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: str) -> list[dict]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prediction_samples(record: dict) -> list[str]:
    """Return all generated solutions while preserving old one-sample files."""
    values = record.get("predictions")
    if isinstance(values, list) and values:
        return [str(value) for value in values]
    if "prediction" in record:
        return [str(record["prediction"])]
    if "solution" in record:
        return [str(record["solution"])]
    raise KeyError(f"Code record has no prediction/solution: {record.keys()}")


def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float | None:
    """Unbiased pass@k estimator used by HumanEval/BigCodeBench.

    A value is undefined when fewer than k samples were generated for a
    problem, so callers receive ``None`` instead of a misleading zero.
    """
    if num_samples < k:
        return None
    if num_samples - num_correct < k:
        return 1.0
    return float(1.0 - math.prod(
        1.0 - k / value
        for value in range(num_samples - num_correct + 1, num_samples + 1)
    ))


def summarize_pass_at_k(counts: list[tuple[int, int]]) -> dict[str, float | None]:
    return {
        f"pass@{k}": (
            float(sum(value for value in values) / len(values))
            if (values := [estimate_pass_at_k(n, c, k) for n, c in counts if n >= k])
            else None
        )
        for k in (1, 5, 10)
    }


def _limit_child(cpu_seconds: int):
    memory_bytes = 4 * 1024**3
    file_bytes = 32 * 1024**2
    limits = [
        (resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1)),
        (resource.RLIMIT_AS, (memory_bytes, memory_bytes)),
        (resource.RLIMIT_FSIZE, (file_bytes, file_bytes)),
        (resource.RLIMIT_NOFILE, (64, 64)),
    ]
    if hasattr(resource, "RLIMIT_NPROC"):
        limits.append((resource.RLIMIT_NPROC, (32, 32)))
    for kind, value in limits:
        try:
            soft, hard = resource.getrlimit(kind)
            requested_soft, requested_hard = value
            if hard != resource.RLIM_INFINITY:
                requested_hard = min(requested_hard, hard)
                requested_soft = min(requested_soft, requested_hard)
            resource.setrlimit(kind, (requested_soft, requested_hard))
        except (OSError, ValueError):
            # Platform-specific limits can be unsupported (notably RLIMIT_AS
            # on macOS). The LUMI execution path additionally relies on a
            # networkless Apptainer boundary.
            pass
    try:
        os.setsid()
    except OSError:
        pass


def _kodcode_expected_function_name(record: dict) -> str:
    info = record.get("test_info")
    if isinstance(info, str):
        try:
            info = json.loads(info)
        except json.JSONDecodeError:
            try:
                info = ast.literal_eval(info)
            except (SyntaxError, ValueError):
                info = None
    if isinstance(info, list) and info and isinstance(info[0], dict):
        name = info[0].get("function_name")
        if name:
            return str(name)
    imported = re.search(r"from\s+solution\s+import\s+([A-Za-z_]\w*)", record["tests"])
    return imported.group(1) if imported else ""


def _prepare_kodcode_prediction(record: dict) -> str:
    """Match MemGen's code extraction and first-function renaming protocol."""
    prediction = record["prediction"]
    blocks = re.findall(r"```python(.*?)```", prediction, flags=re.DOTALL | re.IGNORECASE)
    if not blocks:
        blocks = [prediction]
    extracted = []
    for block in blocks:
        imports = re.findall(
            r"^(?:from\s+\S+\s+import\s+\S+|import\s+\S+.*)$",
            block,
            flags=re.MULTILINE,
        )
        functions = re.findall(
            r"(def\s+\w+\(.*?:[\s\S]*?)(?=^def\s|\Z)",
            block.strip(),
            flags=re.MULTILINE,
        )
        if imports:
            functions = ["\n".join(imports)] + functions
        extracted.extend(functions)
    prepared = "\n".join(extracted) if extracted else prediction
    expected_name = _kodcode_expected_function_name(record)
    if expected_name:
        prepared = re.sub(
            r"def\s+(\w+)\s*\(",
            f"def {expected_name}(",
            prepared,
            count=1,
        )
    return prepared


def evaluate_kodcode_record(record: dict, timeout: int) -> dict:
    with tempfile.TemporaryDirectory(prefix="athena-kodcode-") as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "solution.py").write_text(_prepare_kodcode_prediction(record))
        (tmp_path / "test_solution.py").write_text(record["tests"])
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "HOME": tmp,
            "TMPDIR": tmp,
        }
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "test_solution.py"],
                cwd=tmp,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout + 2,
                preexec_fn=lambda: _limit_child(timeout),
            )
            passed = completed.returncode == 0
            output = completed.stdout[-4000:]
            status = "pass" if passed else "fail"
        except subprocess.TimeoutExpired as exc:
            passed = False
            output = ((exc.stdout or "") + (exc.stderr or ""))[-4000:]
            status = "timeout"
        except BaseException as exc:
            passed = False
            output = f"{type(exc).__name__}: {exc}"
            status = "error"
    return {
        "task_id": record["task_id"],
        "passed": passed,
        "status": status,
        "output": output,
    }


def evaluate_kodcode_candidate(record: dict, prediction: str, sample_index: int, timeout: int) -> dict:
    candidate = dict(record)
    candidate["prediction"] = prediction
    result = evaluate_kodcode_record(candidate, timeout)
    result["sample_index"] = sample_index
    return result


def evaluate_kodcode(records: list[dict], output_dir: Path, parallel: int, timeout: int):
    results = []
    task_counts = []
    candidate_jobs = []
    for record in records:
        candidates = prediction_samples(record)
        task_counts.append((record["task_id"], len(candidates)))
        candidate_jobs.extend(
            (record, prediction, index)
            for index, prediction in enumerate(candidates)
        )
    # A process pool avoids combining fork/preexec resource setup with a
    # multi-threaded parent process.
    with concurrent.futures.ProcessPoolExecutor(max_workers=parallel) as executor:
        futures = [
            executor.submit(evaluate_kodcode_candidate, record, prediction, index, timeout)
            for record, prediction, index in candidate_jobs
        ]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            results.append(result)
            print(
                f"KODCODE_FUNCTIONAL_COMPLETE {index}/{len(candidate_jobs)} "
                f"{result['task_id']} sample={result['sample_index']} {result['status']}"
            )
    results.sort(key=lambda item: item["task_id"])
    (output_dir / "kodcode_execution_results.json").write_text(
        json.dumps(results, indent=2)
    )
    passed = sum(int(result["passed"]) for result in results)
    by_task = {}
    for result in results:
        by_task.setdefault(result["task_id"], []).append(int(result["passed"]))
    counts = [(len(values), sum(values)) for values in by_task.values()]
    pass_at_k = summarize_pass_at_k(counts)
    pass1 = pass_at_k["pass@1"]
    return pass1 if pass1 is not None else 0.0, {
        "passed": passed,
        "total": len(results),
        "problem_count": len(by_task),
        "test_pass_rate": passed / len(results) if results else 0.0,
        "pass_at_k": pass_at_k,
        "samples_per_problem": sorted({len(values) for values in by_task.values()}),
        "evaluator": "pytest_subprocess_inside_networkless_apptainer",
    }


def evaluate_bigcodebench(records: list[dict], output_dir: Path, parallel: int):
    official_samples = output_dir / "bigcodebench_official_samples.jsonl"
    with open(official_samples, "w") as handle:
        for record in records:
            for solution in prediction_samples(record):
                handle.write(json.dumps({
                    "task_id": record["task_id"],
                    "solution": solution,
                }) + "\n")

    from bigcodebench.evaluate import evaluate

    sample_counts = [len(prediction_samples(record)) for record in records]
    requested_k = [1]
    if sample_counts and min(sample_counts) >= 5:
        requested_k.append(5)
    if sample_counts and min(sample_counts) >= 10:
        requested_k.append(10)
    evaluate(
        split="instruct",
        subset="full",
        samples=str(official_samples),
        execution="local",
        pass_k=",".join(str(k) for k in requested_k),
        save_pass_rate=True,
        calibrated=False,
        parallel=parallel,
        min_time_limit=1,
    )
    pass_path = Path(str(official_samples).replace(".jsonl", "_pass_at_k.json"))
    eval_path = Path(str(official_samples).replace(".jsonl", "_eval_results.json"))
    if not pass_path.is_file() or not eval_path.is_file():
        raise FileNotFoundError("Official BigCodeBench evaluator did not emit expected outputs")
    pass_data = json.loads(pass_path.read_text())
    pass_at_k = {f"pass@{k}": pass_data.get(f"pass@{k}") for k in (1, 5, 10)}
    return float(pass_data["pass@1"]), {
        "total": len(records),
        "total_solutions": sum(sample_counts),
        "samples_per_problem": sorted(set(sample_counts)),
        "pass_at_k": pass_at_k,
        "requested_pass_k": requested_k,
        "evaluator": "bigcodebench.evaluate local",
        "split": "instruct",
        "subset": "full",
        "calibrated": False,
        "official_eval_results": str(eval_path),
        "official_pass_at_k": str(pass_path),
        "groundtruth_pass_rate": pass_data.get("gt_pass_rate"),
        "failed_groundtruth_tasks": pass_data.get("failed_tasks", []),
    }


def update_result_file(path: Path, task: str, acc: float, detail: dict):
    payload = json.loads(path.read_text())
    task_result = payload["tasks"][task]
    task_result["metrics"]["acc"] = acc
    if "pass_at_k" in detail:
        task_result["metrics"]["pass_at_k"] = detail["pass_at_k"]
    if "test_pass_rate" in detail:
        task_result["metrics"]["test_pass_rate"] = detail["test_pass_rate"]
    task_result["metrics"]["functional_evaluation"] = detail
    note = task_result["metrics"].get("metric_note", "")
    task_result["metrics"]["metric_note"] = (
        "acc is functional Pass@1; code test_pass_rate and Pass@1/5/10 are "
        "reported when enough solutions were generated; EM/F1 remain lexical diagnostics"
        + (f"; previous note: {note}" if note else "")
    )
    path.write_text(json.dumps(payload, indent=2))


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    marker = output_dir / f"{args.task}_functional_results.json"
    if marker.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {marker}")
    output_dir.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(args.samples)

    if args.task == "kodcode":
        acc, detail = evaluate_kodcode(records, output_dir, args.parallel, args.timeout)
    else:
        acc, detail = evaluate_bigcodebench(records, output_dir, args.parallel)

    summary = {
        "completed": True,
        "task": args.task,
        "acc": acc,
        "em": None,
        "f1": None,
        "pass_at_k": detail.get("pass_at_k"),
        "test_pass_rate": detail.get("test_pass_rate"),
        "detail": detail,
    }
    marker.write_text(json.dumps(summary, indent=2))
    update_result_file(Path(args.results_json), args.task, acc, detail)
    print("CODE_FUNCTIONAL_EVAL_COMPLETE " + json.dumps(summary))


if __name__ == "__main__":
    main()
