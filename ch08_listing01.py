"""Групповое разбиение и аудит схемы искусственных инструкций."""
import hashlib
import json

def group_split(group):
    bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16)
    return 'test' if bucket % 5 == 0 else 'train'

def validate(record):
    assert isinstance(record['group'], str)
    messages = record['messages']
    assert messages and messages[-1]['role'] == 'assistant'
    assert messages[-1]['content'].strip()
    assert all(m['role'] in ['system', 'user', 'assistant']
               and isinstance(m['content'], str) for m in messages)

def run():
    records = []
    for group in range(100):
        for paraphrase in range(3):
            records.append({'group': f'doc-{group}',
              'messages': [
               {'role': 'user', 'content': f'Question {paraphrase}'},
               {'role': 'assistant', 'content': f'Answer {group}'}]})
    seen = {'train': set(), 'test': set()}
    counts = {'train': 0, 'test': 0}
    for record in records:
        validate(record)
        split = group_split(record['group'])
        seen[split].add(record['group']); counts[split] += 1
    assert not (seen['train'] & seen['test'])
    bad = {'group': 'bad', 'messages':
           [{'role': 'assistant', 'content': ' '}]}
    try:
        validate(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('empty answer accepted')
    print(json.dumps(counts, sort_keys=True))
    print('group overlap', len(seen['train'] & seen['test']))

if __name__ == '__main__':
    run()
