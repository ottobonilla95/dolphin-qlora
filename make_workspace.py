"""Copy inputs and scripts into a fresh workspace without touching saved results."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    destination = args.destination.expanduser().resolve()
    if destination == ROOT or ROOT in destination.parents:
        parser.error('Use a separate folder outside the recorded repository.')
    if destination.exists():
        parser.error('Destination already exists; refusing to overwrite it.')
    destination.mkdir(parents=True)
    for source in ROOT.glob('*.py'):
        shutil.copy2(source, destination / source.name)
    for name in ('requirements-lock.txt', 'baseline-prompts.json',
                 'baseline-large-prompts.json', '.gitignore'):
        shutil.copy2(ROOT / name, destination / name)
    for name in ('data', 'results', 'training-model', 'evaluation-source'):
        (destination / name).mkdir()
    for source in (ROOT / 'data').iterdir():
        if source.is_file():
            shutil.copy2(source, destination / 'data' / source.name)
    manifest_path = destination / 'data/split-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['evaluation_suite'] = str(destination / 'baseline-large-prompts.json')
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    config = json.loads((ROOT / 'results/training-001-summary.json').read_text())['config']
    config['model'] = str(destination / 'training-model/dolphin-4bit')
    config['data'] = str(destination / 'data')
    config['adapter_path'] = str(destination / 'adapters/intent-qlora-001')
    # JSON is valid YAML, so the original YAML-based trainer can read this file.
    (destination / 'train-config.yaml').write_text(json.dumps(config, indent=2) + '\n')
    suite = destination / 'baseline-large-prompts.json'
    status = {
        'stage': 'ready_for_model_preparation', 'training_started': False,
        'model_path': config['model'], 'adapter_path': config['adapter_path'],
        'python': str(destination / '.venv/bin/python'),
        'training_data': str(destination / 'data/train.jsonl'),
        'validation_data': str(destination / 'data/valid.jsonl'),
        'evaluation_suite': str(suite),
        'suite_sha256': hashlib.sha256(suite.read_bytes()).hexdigest(),
        'baselines': {},
    }
    (destination / 'experiment-status.json').write_text(json.dumps(status, indent=2) + '\n')
    (destination / 'workspace-origin.json').write_text(json.dumps({
        'source_repository_folder': str(ROOT),
        'source_suite_sha256': status['suite_sha256'],
        'note': 'New workspace. Model conversion, baselines and training have not run.'
    }, indent=2) + '\n')
    print(f'Prepared {destination}; no downloads or training started.')


if __name__ == '__main__':
    main()

