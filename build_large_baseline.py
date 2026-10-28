"""Build a fixed, balanced evaluation suite from pinned Bitext data plus our original cases."""
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import urllib.request

ROOT = Path(__file__).resolve().parent
REPO = 'bitext/Bitext-customer-support-llm-chatbot-training-dataset'
REVISION = '430d1a89bd93bd1fa23c16f29dd53e73f0087443'
FILENAME = 'Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv'
SEED = 42


def normalise(text):
    text = re.sub(r'\{\{.*?\}\}', ' ENTITY ', text)
    return ' '.join(re.findall(r'\w+', text.casefold()))


def tokens(text):
    return set(normalise(text).split())


def similarity(left, right):
    return len(left & right) / max(len(left | right), 1)


def main():
    suite = json.loads((ROOT / 'baseline-prompts.json').read_text())
    source_dir = ROOT / 'evaluation-source'
    source_dir.mkdir(exist_ok=True)
    source_path = source_dir / FILENAME
    source_url = f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{FILENAME}'
    if not source_path.exists():
        temporary = source_path.with_suffix('.download')
        with urllib.request.urlopen(source_url, timeout=120) as response, temporary.open('wb') as target:
            shutil.copyfileobj(response, target)
        temporary.rename(source_path)
    rows = list(csv.DictReader(source_path.open(encoding='utf-8-sig', newline='')))
    if not rows or not {'instruction', 'intent', 'flags'}.issubset(rows[0]):
        raise ValueError('Unexpected source dataset columns.')
    labels = suite['allowed_intents']
    if set(row['intent'] for row in rows) != set(labels):
        raise ValueError('Source intents do not match the existing evaluation task.')
    rng = random.Random(SEED)
    candidates = {}
    conflicting = set()
    for index, row in enumerate(rows):
        key = normalise(row['instruction'])
        if key in candidates and candidates[key][1]['intent'] != row['intent']:
            conflicting.add(key)
        elif key not in candidates:
            candidates[key] = (index, row)
    original_cases = [{**case, 'origin': 'handwritten'} for case in suite['cases']]
    reserved = {normalise(case['prompt']) for case in original_cases}
    cases = []
    for label in labels:
        pool = [(key, index, row) for key, (index, row) in candidates.items()
                if row['intent'] == label and key not in conflicting
                and key not in reserved and 'W' not in row['flags']]
        rng.shuffle(pool)
        selected = []
        # Select diverse wording, including three typo-tagged and three colloquial examples.
        for flag, target_count in [('Z', 3), ('Q', 6), (None, 10)]:
            while len(selected) < target_count:
                eligible = [item for item in pool if flag is None or flag in item[2]['flags']]
                if not eligible:
                    raise ValueError(f'Not enough unique candidates for {label}, flag={flag}')
                known = [tokens(item[2]['instruction']) for item in selected]
                pick = min(eligible, key=lambda item: max(
                    (similarity(tokens(item[2]['instruction']), prior) for prior in known), default=0))
                selected.append(pick)
                pool.remove(pick)
        for number, (key, index, row) in enumerate(selected, 1):
            reserved.add(key)
            cases.append({
                'id': f'bitext-{label}-{number:02}', 'kind': 'intent',
                'origin': 'bitext', 'prompt': row['instruction'],
                'expected_intent': label, 'source_row_index_zero_based': index,
                'source_flags': row['flags'],
            })
    rng.shuffle(cases)
    suite['cases'] = cases + original_cases
    suite['description'] = 'Fixed pre-training evaluation: 270 Bitext examples, 27 original intent cases, 3 general cases.'
    suite['provenance'] = {
        'dataset': REPO, 'revision': REVISION, 'filename': FILENAME,
        'url': source_url, 'license': 'cdla-sharing-1.0', 'seed': SEED,
        'source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
        'sampling': '10 unique instructions per intent; prefer varied wording, include typo and colloquial tags; exclude W-tagged examples.',
        'limitations': [
            'Public synthetic examples may have been seen during the original model training.',
            'Source labels can be ambiguous; inspect mismatches before treating them as model errors.',
            'Exact normalised evaluation instructions must be excluded from future training and validation.',
            'Near paraphrases may remain elsewhere in the source; apply grouping when preparing training data.',
        ],
    }
    output = ROOT / 'baseline-large-prompts.json'
    if output.exists():
        raise FileExistsError('Large suite already exists; preserve its evaluation identity.')
    output.write_text(json.dumps(suite, indent=2, ensure_ascii=False) + '\n')
    (source_dir / 'reserved-instructions.json').write_text(json.dumps({
        'normalisation': 'Replace {{...}} placeholders with ENTITY, casefold, extract word tokens, join with spaces.',
        'keys': sorted(reserved), 'suite': str(output),
    }, indent=2) + '\n')
    assert len(cases) == 270
    assert len({normalise(case['prompt']) for case in suite['cases']}) == 300
    assert suite['intent_system'] == json.loads((ROOT / 'baseline-prompts.json').read_text())['intent_system']
    print(f'Source rows: {len(rows)}; conflicting normalised instructions excluded: {len(conflicting)}')
    print(f'Wrote {len(suite["cases"])} cases to {output}')


if __name__ == '__main__':
    main()
