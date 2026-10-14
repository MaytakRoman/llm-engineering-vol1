import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path


def canonical(text):
    return unicodedata.normalize('NFC', text).strip()


def make_rows():
    rows = []
    for group in range(100):
        text = (f'Документ {group:03d}. Датчик измеряет давление '
                f'в контуре номер {group:03d}.')
        for version in range(3):
            value = text if version < 2 else text + '  '
            rows.append({'id': f'{group:03d}-{version}',
                         'group': group, 'text': value})
    return rows


def exact_dedup(rows):
    seen, accepted, log = {}, [], []
    for row in sorted(rows, key=lambda x: x['id']):
        value = canonical(row['text'])
        key = hashlib.sha256(value.encode('utf-8')).hexdigest()
        duplicate = key in seen
        if not duplicate:
            seen[key] = row['id']
            accepted.append(dict(row, text=value))
        log.append({'document_id': row['id'],
                    'canonical_hash': key,
                    'decision': 'duplicate' if duplicate else 'accept',
                    'representative': seen[key],
                    'pipeline_version': 'toy-v1'})
    return accepted, log


def group_split(rows, seed=42):
    groups = sorted({r['group'] for r in rows})
    random.Random(seed).shuffle(groups)
    train_groups = set(groups[:80])
    train = [r for r in rows if r['group'] in train_groups]
    test = [r for r in rows if r['group'] not in train_groups]
    assert not ({r['group'] for r in train}
                & {r['group'] for r in test})
    return train, test


def overlap(train, test):
    return len({r['group'] for r in train}
               & {r['group'] for r in test})


def redact_email_template(text):
    # Учебный детектор строк, похожих на адреса.
    pattern = r'[A-Za-z0-9_.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'
    return re.sub(pattern, '<EMAIL>', text)


def run(output='toy_pipeline'):
    out = Path(output)
    out.mkdir(exist_ok=True)
    rows = make_rows()
    shuffled = rows.copy()
    random.Random(42).shuffle(shuffled)
    bad_overlap = overlap(shuffled[:240], shuffled[240:])
    unique, log = exact_dedup(rows)
    train, test = group_split(unique)
    assert len(rows) == 300
    assert len(unique) == 100
    assert bad_overlap == 47
    assert len(train) == 80 and len(test) == 20
    synthetic = 'Контакт: user@example.invalid.'
    assert '<EMAIL>' in redact_email_template(synthetic)
    for name, records in [('train', train), ('test', test),
                          ('decisions', log)]:
        with (out/(name+'.jsonl')).open('w', encoding='utf-8') as f:
            for row in records:
                f.write(json.dumps(row, ensure_ascii=False)+'\n')
    result = {'raw': len(rows), 'unique': len(unique),
              'row_split_overlap': bad_overlap,
              'group_split_overlap': overlap(train, test)}
    (out/'summary.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding='utf-8')
    print(result)


if __name__ == '__main__':
    run()
