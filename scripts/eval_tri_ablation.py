"""Full five-task evaluation of one final ablation, retaining paired predictions."""
import argparse
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.eval_fair_joint_openqa_paired import _evaluate_tasks
from scripts.eval_dual_reader_openqa_paired import load_task
from scripts.eval_openqa import setup_condition
from scripts.validate_tri_advantage_eval import TASK_COUNTS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['adaptor-dir', 'source-memory', 'memory-config', 'output-dir', 'condition']:
        p.add_argument('--' + name, required=True)
    p.add_argument('--target-model', default='mistralai/Mistral-7B-v0.3')
    args = p.parse_args()
    out = Path(args.output_dir)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    args.target_model = args.target_model
    args.seed = 42
    args.canon_mode = 'word_boundary'
    args.adaptor_checkpoint = None
    args.dual_reader_mode = 'both' if args.condition == 'ffn_only' else 'tri_advantage_routed'
    args.max_context_length = None
    args.max_new_tokens = 15
    args.reasoning_mode = 'vanilla'
    args.triviaqa_config = 'rc.nocontext'
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wrapper, canon = setup_condition(args, args.condition, device,
                                     torch.bfloat16 if device.type == 'cuda' else torch.float32)
    evaluation = {}
    for task, count in TASK_COUNTS.items():
        examples, meta = load_task(task, args.triviaqa_config)
        if len(examples) != count:
            raise RuntimeError(f'{task}: expected {count}, found {len(examples)}')
        args.tasks = [task]
        value = _evaluate_tasks(wrapper, canon, {task: examples}, args, device)
        metrics = value[task]['metrics']
        key = 'sample_examples' if task == 'truthfulqa' else 'sample_predictions'
        if metrics['total'] != count or len(metrics[key]) != count:
            raise RuntimeError(f'{task}: incomplete predictions')
        evaluation.update(value)
        (out / f'{task}.json').write_text(json.dumps({'dataset': meta, **value}))
    (out / 'results.json').write_text(json.dumps({'completed': True, 'condition': args.condition,
        'mode': args.dual_reader_mode, 'evaluation': evaluation, 'seed': args.seed,
        'bootstrap_completed': False}))
    print('ATHENA_TRI_ABLATION_DOWNSTREAM_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
