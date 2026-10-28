"""Evaluate the MLX training model before and after fine tuning."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parent


class LocalMLX:
    """Run each conversation directly against one local MLX model."""
    def __init__(self, model_path, adapter_path=None):
        import mlx.core as mx
        from mlx_lm import load, stream_generate
        from mlx_lm.sample_utils import make_sampler
        self.mx = mx
        self.stream_generate = stream_generate
        self.make_sampler = make_sampler
        print(f'Loading local MLX model: {model_path}', flush=True)
        self.model, self.tokenizer = load(model_path, adapter_path=adapter_path)
        if not self.tokenizer.chat_template:
            raise ValueError('The model has no chat template; preserve its original tokenizer.')

    def chat(self, payload):
        self.mx.random.seed(payload['seed'])
        prompt = self.tokenizer.apply_chat_template(
            payload['messages'], tokenize=True, add_generation_prompt=True, return_dict=False)
        chunks = []
        last = None
        for part in self.stream_generate(
            self.model, self.tokenizer, prompt=prompt, max_tokens=payload['max_tokens'],
            sampler=self.make_sampler(temp=payload['temperature']),
        ):
            chunks.append(part.text)
            last = part
        if last is None:
            raise RuntimeError('MLX returned no generation result.')
        return {
            'model': payload['model'], 'backend': 'mlx',
            'choices': [{'message': {'role': 'assistant', 'content': ''.join(chunks)},
                         'finish_reason': last.finish_reason}],
            'usage': {'prompt_tokens': last.prompt_tokens,
                      'completion_tokens': last.generation_tokens,
                      'total_tokens': last.prompt_tokens + last.generation_tokens},
            'performance': {'prompt_tps': last.prompt_tps,
                            'generation_tps': last.generation_tps,
                            'peak_memory_gb': last.peak_memory},
        }


def chat(endpoint, payload):
    request = urllib.request.Request(
        endpoint.rstrip('/') + '/chat/completions',
        data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', help='MLX model path, or an explicit API model identifier.')
    parser.add_argument('--endpoint', default='http://127.0.0.1:1234/v1')
    parser.add_argument('--prompt', help='Try one custom prompt instead of the fixed suite.')
    parser.add_argument('--label', default='original', help='Name this run, e.g. original or tuned.')
    parser.add_argument('--suite', type=Path, default=ROOT / 'baseline-large-prompts.json')
    parser.add_argument('--backend', choices=['api', 'mlx'], default='mlx')
    parser.add_argument('--adapter-path', type=Path, help='Load an MLX adapter for the after-training evaluation.')
    args = parser.parse_args()
    if args.model is None:
        if args.backend == 'api':
            parser.error('--backend api requires an explicit --model identifier')
        args.model = str(ROOT / 'training-model/dolphin-4bit')
    suite_path = args.suite.resolve()
    suite_bytes = suite_path.read_bytes()
    suite = json.loads(suite_bytes)
    cases = suite['cases']
    if args.prompt:
        cases = [{'id': 'custom', 'kind': 'general', 'prompt': args.prompt}]
    if args.backend == 'mlx':
        runner = LocalMLX(args.model, str(args.adapter_path.resolve()) if args.adapter_path else None)
        infer = runner.chat
        model_manifest = ROOT / 'training-model/model-manifest.json'
    else:
        if args.adapter_path:
            parser.error('--adapter-path requires --backend mlx')
        infer = lambda payload: chat(args.endpoint, payload)
        model_manifest = None
    start = datetime.now(timezone.utc)
    results_dir = ROOT / 'results'
    results_dir.mkdir(exist_ok=True)
    run_id = start.strftime('%Y%m%dT%H%M%S%fZ')
    output_path = results_dir / f'{run_id}.json'
    text_path = results_dir / f'{run_id}.txt'
    run = {
        'label': args.label, 'started_utc': start.isoformat(),
        'model': args.model, 'backend': args.backend,
        'endpoint': args.endpoint if args.backend == 'api' else None,
        'suite_sha256': hashlib.sha256(suite_bytes).hexdigest(),
        'suite_path': str(suite_path),
        'suite_provenance': suite.get('provenance'),
        'source_model': json.loads(model_manifest.read_text()) if model_manifest else None,
        'settings': {'temperature': 0, 'seed': 42, 'stream': False},
        'notes': [
            'Each case uses a separate conversation. No forced JSON mode.',
            'Exploratory suite; expected labels are supplied by the suite, not independently adjudicated.',
            'Keep these cases out of fine-tuning data.',
            'For MLX training, also baseline the exact MLX training weights before tuning.',
            'Fixed settings aid comparison; identical output across backends is not guaranteed.',
        ],
        'results': [],
    }
    if args.adapter_path:
        adapter_dir = args.adapter_path.resolve()
        run['adapter'] = {
            'path': str(adapter_dir),
            'config': json.loads((adapter_dir / 'adapter_config.json').read_text()),
            'weights_sha256': hashlib.sha256((adapter_dir / 'adapters.safetensors').read_bytes()).hexdigest(),
        }

    def save():
        output_path.write_text(json.dumps(run, indent=2, ensure_ascii=False) + '\n')
        sections = [f"Run: {run['label']}\nModel: {args.model}\nSuite: {suite_path}\nSettings: {run['settings']}\n"]
        if 'summary' in run:
            sections.append('SUMMARY:\n' + json.dumps(run['summary'], indent=2))
            mismatches = [item for item in run['results']
                          if item.get('correct') is False or 'error' in item]
            sections.append('MISMATCHES AND ERRORS:\n' + '\n'.join(
                f"{item['id']} | expected={item.get('expected_intent', '')} | "
                f"response={item.get('response', item.get('error', ''))}"
                for item in mismatches))
        for item in run['results']:
            sections.append(
                f"[{item['id']}]\nPROMPT: {item['prompt']}\n"
                + (f"EXPECTED INTENT: {item['expected_intent']}\n" if 'expected_intent' in item else '')
                + f"RESPONSE:\n{item.get('response', item.get('error', ''))}\n"
            )
        text_path.write_text('\n'.join(sections) + '\n')

    for index, case in enumerate(cases, 1):
        system = suite['intent_system'] if case['kind'] == 'intent' else suite['general_system']
        payload = {
            'model': args.model,
            'messages': [{'role': 'system', 'content': system},
                         {'role': 'user', 'content': case['prompt']}],
            **run['settings'],
            'max_tokens': 128 if case['kind'] == 'intent' else 256,
        }
        record = {**case, 'request': payload}
        tick = time.monotonic()
        try:
            raw = infer(payload)
            choice = raw['choices'][0]
            answer = choice['message'].get('content') or ''
            record.update(response=answer, finish_reason=choice.get('finish_reason'),
                          usage=raw.get('usage'), raw_response=raw)
            if case['kind'] == 'intent':
                try:
                    parsed = json.loads(answer)
                    record['valid_json'] = True
                except json.JSONDecodeError:
                    parsed = None
                    record['valid_json'] = False
                schema_ok = (isinstance(parsed, dict) and set(parsed) == {'intent'}
                             and isinstance(parsed.get('intent'), str)
                             and parsed['intent'] in suite['allowed_intents'])
                record['valid_schema'] = schema_ok
                record['correct'] = bool(schema_ok and parsed['intent'] == case['expected_intent'])
        except Exception as error:
            record['error'] = str(error)
        record['elapsed_seconds'] = round(time.monotonic() - tick, 3)
        run['results'].append(record)
        save()
        status = ('ERROR' if 'error' in record else
                  'SAVED' if case['kind'] != 'intent' else
                  'PASS' if record.get('correct') else 'MISMATCH')
        print(f"{index}/{len(cases)} {case['id']}: {status}", flush=True)
        if args.prompt:
            print(record.get('response', record.get('error', '')), flush=True)

    intent_results = [r for r in run['results'] if r['kind'] == 'intent']
    run['summary'] = {
        'total_cases': len(cases),
        'completed_responses': sum('response' in r for r in run['results']),
        'errors': sum('error' in r for r in run['results']),
        'intent_cases': len(intent_results),
        'correct_intent_and_schema': sum(r.get('correct', False) for r in intent_results),
        'valid_json': sum(r.get('valid_json', False) for r in intent_results),
        'valid_schema': sum(r.get('valid_schema', False) for r in intent_results),
    }
    count = len(intent_results)
    run['summary']['intent_accuracy_percent'] = (
        round(100 * run['summary']['correct_intent_and_schema'] / count, 2) if count else None)
    for field, output_key in [('expected_intent', 'by_intent'), ('origin', 'by_origin')]:
        groups = {}
        for item in intent_results:
            key = item.get(field, 'handwritten' if field == 'origin' else 'unknown')
            group = groups.setdefault(key, {'total': 0, 'correct': 0})
            group['total'] += 1
            group['correct'] += bool(item.get('correct'))
        for group in groups.values():
            group['accuracy_percent'] = round(100 * group['correct'] / group['total'], 2)
        run['summary'][output_key] = groups
    run['finished_utc'] = datetime.now(timezone.utc).isoformat()
    save()
    print(json.dumps(run['summary'], indent=2))
    print('Responses:', text_path)
    print('Complete record:', output_path)


if __name__ == '__main__':
    main()
