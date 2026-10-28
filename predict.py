"""Try the local classifier; opt into the experimental adapter for comparison."""
import argparse
import json
from pathlib import Path

from baseline import LocalMLX

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('message', help='Customer message to classify.')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--base', action='store_true', help='Use the original MLX weights (the default).')
    selection.add_argument('--adapter', action='store_true', help='Use experimental adapter 001, which regressed in evaluation.')
    parser.add_argument('--general', action='store_true', help='Use the general-assistant system prompt instead.')
    args = parser.parse_args()
    status = json.loads((ROOT / 'experiment-status.json').read_text())
    suite = json.loads((ROOT / 'baseline-large-prompts.json').read_text())
    adapter = status.get('adapter_path') if args.adapter else None
    if args.adapter and (not adapter or not (Path(adapter) / 'adapters.safetensors').is_file()):
        parser.error('The trained adapter is not ready yet.')
    runner = LocalMLX(status['model_path'], adapter)
    result = runner.chat({
        'model': status['model_path'],
        'messages': [
            {'role': 'system', 'content': suite['general_system'] if args.general else suite['intent_system']},
            {'role': 'user', 'content': args.message},
        ],
        'temperature': 0, 'seed': 42, 'max_tokens': 256 if args.general else 128,
    })
    print(result['choices'][0]['message']['content'])


if __name__ == '__main__':
    main()
