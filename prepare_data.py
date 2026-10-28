"""Prepare balanced training and validation examples while reserving the baseline suite."""
import csv
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import random
import re

ROOT = Path(__file__).resolve().parent
TRAIN_PER_INTENT = 40
VALID_PER_INTENT = 10
SEED = 42


def normalise(text):
    text = re.sub(r'\{\{.*?\}\}', ' ENTITY ', text)
    return ' '.join(re.findall(r'\w+', text.casefold()))


def representation(text):
    key = normalise(text)
    return key, set(key.split())


def near_duplicate(left, right):
    a, a_words = left
    b, b_words = right
    if a == b:
        return True
    if len(a_words & b_words) / max(len(a_words | b_words), 1) >= 0.8:
        return True
    if 2 * min(len(a), len(b)) / max(len(a) + len(b), 1) < 0.9:
        return False
    matcher = SequenceMatcher(None, a, b, autojunk=False)
    return matcher.quick_ratio() >= 0.9 and matcher.ratio() >= 0.9


def main():
    data_dir = ROOT / 'data'
    outputs = [data_dir / name for name in ('train.jsonl', 'valid.jsonl', 'split-manifest.json')]
    if any(path.exists() for path in outputs):
        raise FileExistsError('Prepared data already exists; preserve the saved split.')
    suite_path = ROOT / 'baseline-large-prompts.json'
    suite_bytes = suite_path.read_bytes()
    suite = json.loads(suite_bytes)
    provenance = suite['provenance']
    source_path = ROOT / 'evaluation-source' / provenance['filename']
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if source_hash != provenance['source_sha256']:
        raise ValueError('Source data checksum differs from the baseline source.')
    with source_path.open(encoding='utf-8-sig', newline='') as source:
        rows = list(csv.DictReader(source))
    representations = [representation(row['instruction']) for row in rows]
    test_representations = [representation(case['prompt']) for case in suite['cases']]
    label_for_key = {}
    conflicting = set()
    for row, (key, _) in zip(rows, representations):
        previous = label_for_key.setdefault(key, row['intent'])
        if previous != row['intent']:
            conflicting.add(key)
    rng = random.Random(SEED)
    train_indices = []
    valid_indices = []
    rejection_counts = {'test_overlap': 0, 'selected_overlap': 0, 'conflicting_or_W_tag': 0}
    # Every selected example must differ from the fixed evaluation set and from
    # all other selected examples under the documented similarity heuristic.
    selected_representations = []
    for label in suite['allowed_intents']:
        candidates = [i for i, row in enumerate(rows) if row['intent'] == label]
        rng.shuffle(candidates)
        selected = []
        for index in candidates:
            row = rows[index]
            rep = representations[index]
            if rep[0] in conflicting or 'W' in row['flags']:
                rejection_counts['conflicting_or_W_tag'] += 1
                continue
            if any(near_duplicate(rep, test) for test in test_representations):
                rejection_counts['test_overlap'] += 1
                continue
            if any(near_duplicate(rep, prior) for prior in selected_representations):
                rejection_counts['selected_overlap'] += 1
                continue
            selected.append(index)
            selected_representations.append(rep)
            if len(selected) == TRAIN_PER_INTENT + VALID_PER_INTENT:
                break
        if len(selected) < TRAIN_PER_INTENT + VALID_PER_INTENT:
            raise ValueError(f'Only {len(selected)} sufficiently different examples for {label}. No output written.')
        rng.shuffle(selected)
        train_indices.extend(selected[:TRAIN_PER_INTENT])
        valid_indices.extend(selected[TRAIN_PER_INTENT:])
        print(f'{label}: {TRAIN_PER_INTENT} train, {VALID_PER_INTENT} validation', flush=True)
    rng.shuffle(train_indices)
    rng.shuffle(valid_indices)
    assert not set(train_indices) & set(valid_indices)
    assert len(train_indices) == 1080 and len(valid_indices) == 270
    test_keys = {rep[0] for rep in test_representations}
    assert all(representations[i][0] not in test_keys for i in train_indices + valid_indices)

    def record(index):
        row = rows[index]
        return {'messages': [
            {'role': 'system', 'content': suite['intent_system']},
            {'role': 'user', 'content': row['instruction']},
            {'role': 'assistant', 'content': json.dumps({'intent': row['intent']}, separators=(',', ':'))},
        ]}

    data_dir.mkdir(exist_ok=True)
    split_details = {}
    for filename, indices in [('train.jsonl', train_indices), ('valid.jsonl', valid_indices)]:
        payload = ''.join(json.dumps(record(i), ensure_ascii=False) + '\n' for i in indices)
        path = data_dir / filename
        path.write_text(payload)
        split_details[filename] = {
            'count': len(indices), 'sha256': hashlib.sha256(payload.encode()).hexdigest(),
            'source_row_indices_zero_based': indices,
        }
    manifest = {
        'source': provenance, 'seed': SEED,
        'evaluation_suite': str(suite_path),
        'evaluation_suite_sha256': hashlib.sha256(suite_bytes).hexdigest(),
        'system_prompt': suite['intent_system'],
        'per_intent': {'train': TRAIN_PER_INTENT, 'validation': VALID_PER_INTENT},
        'splits': split_details, 'candidate_rejections': rejection_counts,
        'overlap_policy': {
            'exact': 'Normalise case, punctuation, spaces and {{...}} placeholders.',
            'near': 'Reject token-set Jaccard >= 0.8 or normalised character SequenceMatcher ratio >= 0.9.',
            'applies_to': 'All selected examples versus evaluation, and all pairs within/between training and validation.',
            'limitation': 'This lexical heuristic cannot rule out all semantic paraphrases or prior model exposure.',
        },
        'training_started': False,
    }
    outputs[2].write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'training': len(train_indices), 'validation': len(valid_indices),
                      'rejections': rejection_counts}, indent=2))


if __name__ == '__main__':
    main()
