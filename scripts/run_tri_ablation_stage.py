"""Run one immutable, full-data ablation stage; never overwrite artifacts."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

CONDITIONS = ('random_memory', 'permuted_keys', 'no_gate', 'affine_stitch', 'train_from_scratch', 'ffn_only')


def stage_config(template, condition, output, source_memory, memory_config, *, ffn_init=None):
    config = dict(template)
    config.update(condition=condition, output_dir=str(output), source_memory=str(source_memory),
                  memory_config=str(memory_config), max_tokens=20_000_000, validation_max_tokens=2_000_000,
                  seq_len=2048, batch_size=1, grad_accum_steps=1, seed=42, generator_loop_rounds=1,
                  skip_final_test_eval=True, architecture='generative', generator_fusion_type='tri_reader',
                  generator_cue_source='hybrid', generator_adaptive_router=True, init_adaptor=None,
                  deployment_reader_mode=None, advantage_reader=None)
    if condition == 'ffn_only':
        config.update(joint_tri_subset_reader=False, joint_tri_reader=False, joint_tri_route_only=False,
                      deterministic_engram_init_seed=None, deployment_reader_mode=None,
                      init_adaptor=str(ffn_init) if ffn_init else None)
    else:
        config.update(joint_tri_subset_reader=True, joint_tri_reader=False, joint_tri_route_only=False)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['experts', 'router', 'eval'], required=True)
    parser.add_argument('--condition', choices=CONDITIONS, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--source-memory', type=Path, required=True)
    parser.add_argument('--memory-config', type=Path, required=True)
    parser.add_argument('--target-model', default='mistralai/Mistral-7B-v0.3')
    parser.add_argument('--source-tokenizer', default='mistralai/Mistral-7B-v0.3')
    parser.add_argument('--require-tokenizer-match', action='store_true')
    args = parser.parse_args()
    root = args.run_root / args.condition
    experts, reader = root / 'experts', root / 'reader'
    template = json.loads(args.template.read_text())
    template['target_model'] = args.target_model
    template['wikipedia2021_source_tokenizer'] = args.source_tokenizer
    template['wikipedia2021_require_tokenizer_match'] = args.require_tokenizer_match
    output = {'experts': experts, 'router': reader, 'eval': root / 'downstream'}[args.stage]
    if output.exists():
        existing = {p.name for p in output.iterdir()}
        # A job can fail after writing only the resolved runtime config.  It
        # is safe to retry that incomplete output, but never reuse a
        # directory containing checkpoints, results, or training logs.
        if existing - {f'{args.stage}_config.json', 'config.json'}:
            raise FileExistsError(f'Refusing to overwrite non-empty output {output}')
    root.mkdir(parents=True, exist_ok=True)
    if args.stage == 'experts' or (args.stage == 'router' and args.condition == 'ffn_only'):
        config = stage_config(template, args.condition, output, args.source_memory, args.memory_config,
                              ffn_init=experts / 'adaptor_best.pt' if args.stage == 'router' else None)
        config_path = root / f'{args.stage}_config.json'
        config_path.write_text(json.dumps({'adaptor_training': config}, indent=2))
        cmd = [sys.executable, 'scripts/train_adaptor.py', '--condition', args.condition,
               '--config', str(config_path), '--output-dir', str(output)]
    elif args.stage == 'router':
        cmd = [sys.executable, 'scripts/train_counterfactual_router.py', '--adaptor-dir', str(experts),
               '--source-memory', str(args.source_memory), '--memory-config', str(args.memory_config),
               '--output-dir', str(reader), '--candidate-space', 'sources', '--max-tokens', '20000000',
               '--validation-max-tokens', '2000000', '--seq-len', '2048', '--batch-size', '1',
               '--grad-accum-steps', '1', '--warmup-steps', '200', '--eval-every', '2000', '--log-every', '50',
               '--wikipedia2021-source-tokenizer', 'mistralai/Mistral-7B-v0.3',
               '--seed', '42']
        if args.require_tokenizer_match:
            cmd.insert(cmd.index('--seed'), '--wikipedia2021-require-tokenizer-match')
        cmd[cmd.index('--wikipedia2021-source-tokenizer') + 1] = args.source_tokenizer
        cmd[2:2] = ['--target-model', args.target_model]
    else:
        cmd = [sys.executable, 'scripts/eval_tri_ablation.py', '--adaptor-dir', str(reader),
               '--condition', args.condition, '--source-memory', str(args.source_memory),
               '--memory-config', str(args.memory_config), '--output-dir', str(output),
               '--target-model', args.target_model]
    print(json.dumps({'stage': args.stage, 'condition': args.condition, 'command': cmd}), flush=True)
    subprocess.run(cmd, check=True)
    result = json.loads((output / 'results.json').read_text())
    if result.get('completed') is not True:
        raise RuntimeError('Missing completed result')
    if args.stage != 'eval':
        if result['actual_steps'] != 9765 or result['max_tokens'] != 20_000_000:
            raise RuntimeError('Unexpected training budget')
        if args.condition in ('random_memory', 'permuted_keys', 'train_from_scratch'):
            if not (output / 'memory.pt').is_file():
                raise RuntimeError('Missing exact memory artifact')
    print(f'ATHENA_TRI_ABLATION_{args.stage.upper()}_COMPLETE {args.condition}', flush=True)


if __name__ == '__main__':
    main()
