"""Run one reproducible MLX QLoRA experiment with local loss records."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import yaml
from mlx_lm import load
from mlx_lm.lora import CONFIG_DEFAULTS, train_model
from mlx_lm.tuner.callbacks import TrainingCallback
from mlx_lm.tuner.datasets import CacheDataset, load_dataset
from mlx_lm.tuner.trainer import evaluate

ROOT = Path(__file__).resolve().parent


class LocalMetrics(TrainingCallback):
    def __init__(self, path):
        self.path = path
        self.events = []

    def record(self, kind, info):
        event = {'kind': kind, **info, 'utc': datetime.now(timezone.utc).isoformat()}
        for key in ('train_loss', 'val_loss'):
            if key in event and not math.isfinite(event[key]):
                raise RuntimeError(f'Non-finite {key}; stopping the run.')
        with self.path.open('a') as target:
            target.write(json.dumps(event) + '\n')
        self.events.append(event)

    def on_train_loss_report(self, info):
        self.record('training', info)

    def on_val_loss_report(self, info):
        # MLX validates before the current iteration's update.
        self.record('validation', {**info, 'completed_optimizer_steps': info['iteration']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'train-config.yaml')
    options = parser.parse_args()
    config = deepcopy(CONFIG_DEFAULTS)
    config.update(yaml.safe_load(options.config.read_text()))
    args = SimpleNamespace(**config)
    adapter_dir = Path(args.adapter_path)
    metrics_path = ROOT / 'results/training-001-metrics.jsonl'
    summary_path = ROOT / 'results/training-001-summary.json'
    if adapter_dir.exists() or metrics_path.exists() or summary_path.exists():
        raise FileExistsError('Preserving existing training output; use a new run name for another experiment.')
    manifest_path = ROOT / 'data/split-manifest.json'
    split_manifest = json.loads(manifest_path.read_text())
    for filename, details in split_manifest['splits'].items():
        actual = hashlib.sha256((Path(args.data) / filename).read_bytes()).hexdigest()
        if actual != details['sha256']:
            raise ValueError(f'Dataset changed: {filename}')
    suite = Path(split_manifest['evaluation_suite'])
    if hashlib.sha256(suite.read_bytes()).hexdigest() != split_manifest['evaluation_suite_sha256']:
        raise ValueError('The reserved evaluation suite changed.')
    status_path = ROOT / 'experiment-status.json'
    status = json.loads(status_path.read_text())
    started = datetime.now(timezone.utc)
    tick = time.monotonic()
    status.update(stage='training', training_started=True,
                  training_config=str(options.config.resolve()), adapter_path=str(adapter_dir),
                  training_started_utc=started.isoformat())
    status_path.write_text(json.dumps(status, indent=2) + '\n')
    np.random.seed(args.seed)
    print('Loading the frozen 4-bit model and the prepared dataset.', flush=True)
    model, tokenizer = load(args.model)
    train_set, valid_set, _ = load_dataset(args, tokenizer)
    if len(train_set) != 1080 or len(valid_set) != 270:
        raise ValueError('Unexpected dataset sizes.')
    if args.iters * args.batch_size != len(train_set) or args.grad_accumulation_steps != 1:
        raise ValueError('This run is configured for exactly one pass and one update per batch.')
    print(f'One pass: {args.iters} updates x {args.batch_size} examples = {len(train_set)} examples.', flush=True)
    metrics = LocalMetrics(metrics_path)
    train_model(args, model, train_set, valid_set, training_callback=metrics)
    # Record final validation after all updates, including the last update.
    final_val_loss = float(evaluate(model, CacheDataset(valid_set), batch_size=args.batch_size,
                                    num_batches=-1, max_seq_length=args.max_seq_length))
    metrics.record('final_validation', {'val_loss': final_val_loss,
                                      'completed_optimizer_steps': args.iters})
    initial_val_loss = next(event['val_loss'] for event in metrics.events if event['kind'] == 'validation')
    weights = adapter_dir / 'adapters.safetensors'
    summary = {
        'started_utc': started.isoformat(), 'finished_utc': datetime.now(timezone.utc).isoformat(),
        'elapsed_seconds': round(time.monotonic() - tick, 2), 'config': config,
        'training_examples': len(train_set), 'validation_examples': len(valid_set),
        'optimizer_updates': args.iters, 'training_passes': 1,
        'initial_validation_loss': initial_val_loss, 'final_validation_loss': final_val_loss,
        'adapter_weights': str(weights), 'adapter_bytes': weights.stat().st_size,
        'adapter_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(),
        'dataset_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'evaluation_suite_sha256': split_manifest['evaluation_suite_sha256'],
        'metrics': str(metrics_path),
        'note': 'Validation loss is next-token loss on the final assistant message; it is not classification accuracy.',
    }
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')
    status.update(stage='trained_pending_evaluation', training_summary=str(summary_path),
                  training_finished_utc=summary['finished_utc'])
    status_path.write_text(json.dumps(status, indent=2) + '\n')
    print(json.dumps({key: summary[key] for key in ('optimizer_updates', 'initial_validation_loss',
          'final_validation_loss', 'adapter_bytes', 'elapsed_seconds')}, indent=2), flush=True)
    print(f'Training summary: {summary_path}', flush=True)


if __name__ == '__main__':
    main()
