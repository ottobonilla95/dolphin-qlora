"""Check the committed experiment evidence without MLX, model files or a GPU."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    suite_bytes = (ROOT / 'baseline-large-prompts.json').read_bytes()
    suite_hash = hashlib.sha256(suite_bytes).hexdigest()
    suite = json.loads(suite_bytes)
    allowed = set(suite['allowed_intents'])
    require(len(suite['cases']) == 300, 'Expected 300 evaluation cases.')
    manifest = read('data/split-manifest.json')
    require(suite_hash == manifest['evaluation_suite_sha256'], 'Evaluation suite hash changed.')
    for filename, details in manifest['splits'].items():
        payload = (ROOT / 'data' / filename).read_bytes()
        require(hashlib.sha256(payload).hexdigest() == details['sha256'], f'Data changed: {filename}')
        require(len(payload.splitlines()) == details['count'], f'Count mismatch: {filename}')
    before = read('results/20261028T002422930075Z.json')
    after = read('results/20261028T004036824182Z.json')
    require(before['suite_sha256'] == after['suite_sha256'] == suite_hash, 'Run suite mismatch.')
    require(before['model'] == after['model'] and before['backend'] == after['backend'] == 'mlx', 'Runtime mismatch.')
    require(len(before['results']) == len(after['results']) == 300, 'Incomplete recorded run.')
    gains = losses = 0
    recomputed = []
    for run in (before, after):
        correct = valid_json = valid_schema = intent_cases = 0
        for item in run['results']:
            require('error' not in item, 'Recorded inference error.')
            if item['kind'] != 'intent':
                continue
            intent_cases += 1
            try:
                parsed = json.loads(item['response'])
                valid_json += 1
            except json.JSONDecodeError:
                parsed = None
            schema = (isinstance(parsed, dict) and set(parsed) == {'intent'}
                      and isinstance(parsed['intent'], str) and parsed['intent'] in allowed)
            valid_schema += bool(schema)
            passed = bool(schema and parsed['intent'] == item['expected_intent'])
            require(passed == item['correct'], f'Incorrect recorded judgment: {item["id"]}')
            correct += passed
        summary = run['summary']
        require(intent_cases == 297, 'Expected 297 scored classifications.')
        require(correct == summary['correct_intent_and_schema'], 'Correct total mismatch.')
        require(valid_json == summary['valid_json'] and valid_schema == summary['valid_schema'], 'Format count mismatch.')
        require(round(100 * correct / intent_cases, 2) == summary['intent_accuracy_percent'], 'Accuracy mismatch.')
        recomputed.append(correct)
    for left, right in zip(before['results'], after['results']):
        require(left['id'] == right['id'] and left['request'] == right['request'], 'Before and after requests differ.')
        if left['kind'] == 'intent':
            gains += bool(not left['correct'] and right['correct'])
            losses += bool(left['correct'] and not right['correct'])
    comparison = read('results/training-001-comparison.json')['classification']
    require(recomputed == [210, 161], 'Reported outcome changed.')
    require(gains == comparison['gained_cases'] == 29 and losses == comparison['lost_cases'] == 78, 'Case change mismatch.')
    require(round(100 * (recomputed[1] - recomputed[0]) / 297, 2) == comparison['delta_percentage_points'] == -16.5, 'Delta mismatch.')
    training = read('results/training-001-summary.json')
    require(training['evaluation_suite_sha256'] == suite_hash, 'Training suite mismatch.')
    require(training['dataset_manifest_sha256'] == hashlib.sha256((ROOT / 'data/split-manifest.json').read_bytes()).hexdigest(), 'Training manifest mismatch.')
    require(training['optimizer_updates'] * training['config']['batch_size'] == training['training_examples'] == 1080, 'Training pass mismatch.')
    reload_check = read('results/training-001-reload-check.json')
    require(abs(reload_check['reloaded_validation_loss'] - training['final_validation_loss']) < 1e-6, 'Reload loss mismatch.')
    require(reload_check['training_generation_prefix_mismatches'] == 0, 'Prefix mismatch recorded.')
    diagnostics = read('results/training-001-diagnostics.json')
    require([(r['step'], sum(i['correct'] for i in r['classification'])) for r in diagnostics['checkpoints']]
            == [(0, 17), (180, 0), (360, 10), (540, 13)], 'Diagnostic checkpoint count mismatch.')
    print('Verified dataset and suite hashes, identical evaluation requests, strict scores, case changes, training provenance, reload loss, and diagnostic counts.')
    print('MLX: 210/297 (70.71%) before; 161/297 (54.21%) after; -16.50 percentage points.')
    print('This verifies recorded evidence; it does not rerun inference or rehash excluded model binaries.')


if __name__ == '__main__':
    main()
