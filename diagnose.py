"""Inspect saved checkpoints on validation examples without training."""
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.tuner.utils import load_adapters

from baseline import LocalMLX

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/training-001-diagnostics.json'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    status = json.loads((ROOT / 'experiment-status.json').read_text())
    suite = json.loads((ROOT / 'baseline-large-prompts.json').read_text())
    selected = {}
    for line in (ROOT / 'data/valid.jsonl').read_text().splitlines():
        row = json.loads(line)
        label = json.loads(row['messages'][-1]['content'])['intent']
        selected.setdefault(label, row)
    runner = LocalMLX(status['model_path'])
    adapter_dir = Path(status['adapter_path'])
    result = {'purpose': 'Diagnostic validation subset, not the frozen test suite.',
              'selection': 'First validation example per intent, 27 total.', 'checkpoints': []}
    for step in (0, 180, 360, 540):
        if step == 180:
            runner.model = load_adapters(runner.model, str(adapter_dir))
        if step:
            weights = adapter_dir / f'{step:07d}_adapters.safetensors'
            runner.model.load_weights(list(mx.load(str(weights)).items()), strict=False)
        runner.model.eval()
        record = {'step': step, 'classification': [], 'general': []}
        for label, row in selected.items():
            payload = {'model': status['model_path'], 'messages': row['messages'][:-1],
                       'temperature': 0, 'seed': 42, 'max_tokens': 64}
            answer = runner.chat(payload)['choices'][0]['message']['content']
            try:
                parsed = json.loads(answer)
            except json.JSONDecodeError:
                parsed = None
            correct = isinstance(parsed, dict) and set(parsed) == {'intent'} and parsed['intent'] == label
            record['classification'].append({'expected': label, 'response': answer, 'correct': correct})
        record['correct'] = sum(r['correct'] for r in record['classification'])
        for case in [c for c in suite['cases'] if c['kind'] == 'general']:
            payload = {'model': status['model_path'], 'messages': [
                {'role': 'system', 'content': suite['general_system']},
                {'role': 'user', 'content': case['prompt']}],
                'temperature': 0, 'seed': 42, 'max_tokens': 64}
            record['general'].append({'id': case['id'], 'response': runner.chat(payload)['choices'][0]['message']['content']})
        result['checkpoints'].append(record)
        OUT.write_text(json.dumps(result, indent=2) + '\n')
        print(f'Step {step}: {record["correct"]}/27 validation questions correct.', flush=True)

    # Compare full-sequence teacher forcing with separate-prefix next-token predictions.
    audits = []
    for label in ('cancel_order', 'track_order', 'contact_human_agent'):
        row = selected[label]
        tokens = runner.tokenizer.apply_chat_template(row['messages'], return_dict=False)
        prefix = runner.tokenizer.apply_chat_template(row['messages'][:-1], add_generation_prompt=True, return_dict=False)
        assert tokens[:len(prefix)] == prefix
        inputs = mx.array([tokens[:-1]])
        targets = mx.array(tokens[1:])
        full = runner.model(inputs)[0].astype(mx.float32)
        ce = nn.losses.cross_entropy(full, targets)
        details = []
        for pos in range(len(prefix), len(tokens)):
            logits = full[pos - 1]
            expected = tokens[pos]
            predicted = int(mx.argmax(logits).item())
            details.append({'target_token': runner.tokenizer.decode([expected]),
                            'predicted_token': runner.tokenizer.decode([predicted]),
                            'target_probability': float(mx.softmax(logits)[expected].item()),
                            'loss': float(ce[pos - 1].item())})
        # Greedy decoding without any KV cache, to check generation implementation.
        growing = list(prefix)
        generated = []
        for _ in range(24):
            logits = runner.model(mx.array([growing]))[0, -1]
            nxt = int(mx.argmax(logits).item())
            if nxt in runner.tokenizer.eos_token_ids:
                break
            generated.append(nxt)
            growing.append(nxt)
        audits.append({'expected': label, 'teacher_forced_answer_mean_loss': float(mx.mean(ce[len(prefix)-1:]).item()),
                       'teacher_forced_tokens': details,
                       'uncached_greedy_response': runner.tokenizer.decode(generated)})
    result['token_audits'] = audits
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Saved {OUT}', flush=True)


if __name__ == '__main__':
    main()
