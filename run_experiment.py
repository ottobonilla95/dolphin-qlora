"""Fresh lower-rate MLX experiments with generated-answer monitoring.

Run --mode tiny first (training-set fit diagnostic), then --mode full.
Each invocation requires a new --name and preserves all earlier experiments.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.lora import CONFIG_DEFAULTS, train_model
from mlx_lm.tuner.callbacks import TrainingCallback
from mlx_lm.tuner.datasets import ChatDataset
import numpy as np
import yaml

from baseline import LocalMLX

ROOT = Path(__file__).resolve().parent
GENERAL = [
    {'id': 'dev-python', 'prompt': 'Explain a Python list in two sentences and give one short example.'},
    {'id': 'dev-arithmetic', 'prompt': 'Four pens cost 3 euro each. I pay 20 euro. How much change do I get?'},
    {'id': 'dev-rewrite', 'prompt': 'Rewrite politely in at most 25 words: We cannot deliver today. Your order will arrive on Friday.'},
]


def sha(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def intent(row):
    return json.loads(row['messages'][-1]['content'])['intent']


def one_per_intent(data):
    selected = {}
    for row in data:
        selected.setdefault(intent(row), row)
    return list(selected.values())


def greedy_chat(runner, payload):
    # LocalMLX.chat reseeds the global PRNG. Pure argmax generation needs no
    # reseeding and leaves the trainer's compiled random state untouched.
    prompt = runner.tokenizer.apply_chat_template(payload['messages'], tokenize=True,
                add_generation_prompt=True, return_dict=False)
    chunks = []
    last = None
    for part in runner.stream_generate(runner.model, runner.tokenizer, prompt=prompt,
                max_tokens=payload['max_tokens'], sampler=runner.make_sampler(temp=0)):
        chunks.append(part.text)
        last = part
    if last is None:
        raise RuntimeError('No generation result')
    return {'choices': [{'message': {'content': ''.join(chunks)}, 'finish_reason': last.finish_reason}],
            'usage': {'prompt_tokens': last.prompt_tokens, 'completion_tokens': last.generation_tokens}}


def score(runner, examples, allowed, general=False):
    """Greedy independent conversations, preserving the trainer's random state."""
    numpy_state = np.random.get_state()
    mlx_state = list(mx.random.state)
    mx.eval(mlx_state)
    was_training = runner.model.training
    runner.model.eval()
    result = {'classification': [], 'general': []}
    tick = time.monotonic()
    try:
        for index, row in enumerate(examples, 1):
            raw = greedy_chat(runner, {'messages': row['messages'][:-1], 'model': 'local-mlx',
                               'seed': 42, 'temperature': 0, 'max_tokens': 128})
            answer = raw['choices'][0]['message']['content']
            try:
                parsed = json.loads(answer)
                valid_json = True
            except json.JSONDecodeError:
                parsed, valid_json = None, False
            valid_schema = (isinstance(parsed, dict) and set(parsed) == {'intent'}
                            and isinstance(parsed.get('intent'), str) and parsed['intent'] in allowed)
            result['classification'].append({
                'prompt': row['messages'][-2]['content'], 'expected': intent(row),
                'response': answer, 'valid_json': valid_json, 'valid_schema': valid_schema,
                'correct': bool(valid_schema and parsed['intent'] == intent(row)),
                'finish_reason': raw['choices'][0]['finish_reason'], 'usage': raw['usage'],
            })
            if len(examples) > 50 and (index % 50 == 0 or index == len(examples)):
                print(f'VALIDATION generation: {index}/{len(examples)}', flush=True)
        if general:
            for case in GENERAL:
                raw = greedy_chat(runner, {'messages': [
                    {'role': 'system', 'content': 'You are Dolphin, a helpful AI assistant.'},
                    {'role': 'user', 'content': case['prompt']}],
                    'model': 'local-mlx', 'seed': 42, 'temperature': 0, 'max_tokens': 128})
                result['general'].append({**case, 'response': raw['choices'][0]['message']['content'],
                                          'finish_reason': raw['choices'][0]['finish_reason']})
    finally:
        np.random.set_state(numpy_state)
        runner.model.train(was_training)
        mx.clear_cache()
        if not all(bool(mx.array_equal(a, b).item()) for a, b in zip(mlx_state, mx.random.state)):
            raise RuntimeError('Greedy monitoring unexpectedly changed the trainer PRNG')
    total = len(examples)
    result.update(total=total, correct=sum(c['correct'] for c in result['classification']),
                  invalid_labels=sum(not c['valid_schema'] for c in result['classification']),
                  elapsed_seconds=round(time.monotonic() - tick, 2))
    result['accuracy_percent'] = round(100 * result['correct'] / total, 2)
    return result


class FitComplete(Exception):
    pass


class Monitor(TrainingCallback):
    def __init__(self, runner, examples, allowed, output, tiny):
        self.runner, self.examples, self.allowed = runner, examples, allowed
        self.output, self.tiny = output, tiny
        self.checks = []
        self.last_step = 0

    def record(self, kind, info):
        for key in ('train_loss', 'val_loss'):
            if key in info and not math.isfinite(info[key]):
                raise RuntimeError(f'Non-finite {key}')
        with (self.output / 'metrics.jsonl').open('a') as target:
            target.write(json.dumps({'kind': kind, **info, 'utc': datetime.now(timezone.utc).isoformat()}) + '\n')

    def check(self, step):
        weights = dict(tree_flatten(self.runner.model.trainable_parameters()))
        path = self.output / 'adapter' / f'{step:07d}_adapters.safetensors'
        mx.save_safetensors(str(path), weights)
        mx.save_safetensors(str(self.output / 'adapter/adapters.safetensors'), weights)
        record = {'completed_optimizer_steps': step,
                  **score(self.runner, self.examples, self.allowed, general=True),
                  'checkpoint': str(path), 'checkpoint_sha256': sha(path)}
        self.checks.append(record)
        write(self.output / 'monitor.json', self.checks)
        print(f'GENERATED CHECK step {step}: {record["correct"]}/{record["total"]} correct; '
              f'{record["invalid_labels"]} invalid labels.', flush=True)
        return record

    def on_train_loss_report(self, info):
        self.last_step = info['iteration']
        self.record('training', info)
        check = self.check(self.last_step)
        if self.tiny and check['correct'] == check['total']:
            raise FitComplete('All reviewed training examples generated correctly.')

    def on_val_loss_report(self, info):
        self.record('validation', {**info, 'completed_optimizer_steps': info['iteration']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['tiny', 'full'], required=True)
    parser.add_argument('--name', required=True)
    opts = parser.parse_args()
    if not opts.name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in opts.name):
        parser.error('--name must contain lowercase letters, numbers, underscores or hyphens')
    output = ROOT / 'experiments' / opts.name
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = json.loads((ROOT / 'data/split-manifest.json').read_text())
    preserved = {str(ROOT / 'data' / name): details['sha256']
                 for name, details in manifest['splits'].items()}
    preserved[str(ROOT / 'baseline-large-prompts.json')] = manifest['evaluation_suite_sha256']
    model_manifest = json.loads((ROOT / 'training-model/model-manifest.json').read_text())
    model_dir = ROOT / 'training-model/dolphin-4bit'
    preserved.update({str(model_dir / name): detail['sha256']
                      for name, detail in model_manifest['files'].items()})
    for path in (ROOT / 'adapters/intent-qlora-001').glob('*.safetensors'):
        preserved[str(path)] = sha(path)
    for path, expected in preserved.items():
        if sha(path) != expected:
            raise ValueError(f'Original evidence changed: {path}')
    train_rows = rows(ROOT / 'data/train.jsonl')
    valid_rows = rows(ROOT / 'data/valid.jsonl')
    allowed = set(intent(r) for r in train_rows)
    tiny = opts.mode == 'tiny'
    if tiny:
        # Reviewed first sample per intent, with three clearer alternatives.
        selected = {intent(r): r for r in one_per_intent(train_rows)}
        for index in (8, 92, 40):
            selected[intent(train_rows[index])] = train_rows[index]
        train_rows = list(selected.values())
        write(output / 'reviewed-sample.json', [{
            'training_row_zero_based': rows(ROOT / 'data/train.jsonl').index(r),
            'prompt': r['messages'][-2]['content'], 'expected': intent(r),
        } for r in train_rows])
    monitored = train_rows if tiny else one_per_intent(valid_rows)
    config = deepcopy(CONFIG_DEFAULTS)
    original_config = yaml.safe_load((ROOT / 'train-config.yaml').read_text())
    config.update(original_config)
    config.update(model=str(model_dir), adapter_path=str(output / 'adapter'),
                  learning_rate=1e-5, steps_per_report=50, save_every=50)
    if tiny:
        config.update(batch_size=1, iters=270, val_batches=27, steps_per_eval=90)
    (output / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    summary = {'mode': opts.mode, 'name': opts.name, 'started_utc': datetime.now(timezone.utc).isoformat(),
               'status': 'running', 'config': config, 'training_examples': len(train_rows),
               'monitored_examples': len(monitored),
               'monitor_role': 'Training-set memorization diagnostic' if tiny else 'First validation example per intent; checkpoint selection only',
               'general_probes_role': 'New development prompts; qualitative review, not a general capability score',
               'preserved_files': preserved,
               'original_config_differences': {k: {'original': original_config.get(k), 'new': v}
                                               for k, v in config.items() if k in original_config and v != original_config[k]},
               'note': 'Frozen final classification suite is not used for training or checkpoint selection.'}
    write(output / 'summary.json', summary)
    runner = LocalMLX(str(model_dir))
    summary['base_monitored'] = score(runner, monitored, allowed, general=True)
    print(f'BASE monitored: {summary["base_monitored"]["correct"]}/{len(monitored)}', flush=True)
    write(output / 'summary.json', summary)
    if not tiny:
        summary['base_full_validation'] = score(runner, valid_rows, allowed)
        print(f'BASE full validation: {summary["base_full_validation"]["correct"]}/270', flush=True)
    write(output / 'summary.json', summary)
    dataset = ChatDataset(train_rows, runner.tokenizer, mask_prompt=True)
    validation = ChatDataset(train_rows if tiny else valid_rows, runner.tokenizer, mask_prompt=True)
    # Check that masking trains exactly the assistant continuation used for generation.
    for row in train_rows:
        tokens, offset = dataset.process(row)
        prefix = runner.tokenizer.apply_chat_template(row['messages'][:-1],
                    add_generation_prompt=True, return_dict=False)
        assert tokens[:offset] == prefix and offset == len(prefix)
        assert len(tokens) <= config['max_seq_length']
    monitor = Monitor(runner, monitored, allowed, output, tiny)
    np.random.seed(config['seed'])
    try:
        train_model(SimpleNamespace(**config), runner.model, dataset, validation, training_callback=monitor)
    except FitComplete:
        summary['early_stop_reason'] = 'Perfect generation on the 27 reviewed training examples'
    summary['completed_optimizer_steps'] = monitor.last_step
    best = max(monitor.checks, key=lambda c: (c['correct'], -c['invalid_labels'], -c['completed_optimizer_steps']))
    summary['selected_checkpoint'] = {k: v for k, v in best.items() if k not in ('classification', 'general')}
    runner.model.load_weights(list(mx.load(best['checkpoint']).items()), strict=False)
    if not tiny:
        summary['selected_full_validation'] = score(runner, valid_rows, allowed, general=True)
        summary['validation_delta_percentage_points'] = round(
            summary['selected_full_validation']['accuracy_percent'] - summary['base_full_validation']['accuracy_percent'], 2)
    else:
        summary['fit_passed'] = best['correct'] == best['total']
    selected_dir = output / 'selected-adapter'
    selected_dir.mkdir()
    shutil.copyfile(best['checkpoint'], selected_dir / 'adapters.safetensors')
    shutil.copyfile(output / 'adapter/adapter_config.json', selected_dir / 'adapter_config.json')
    write(selected_dir / 'selection.json', summary['selected_checkpoint'])
    summary['selected_adapter_path'] = str(selected_dir)
    summary['preserved_files_unchanged'] = all(sha(p) == expected for p, expected in preserved.items())
    if not summary['preserved_files_unchanged']:
        raise RuntimeError('Original evidence changed during training')
    summary.update(status='completed', elapsed_seconds=round(time.monotonic() - started, 2),
                   finished_utc=datetime.now(timezone.utc).isoformat())
    write(output / 'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('status', 'completed_optimizer_steps', 'selected_checkpoint',
          'elapsed_seconds')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
