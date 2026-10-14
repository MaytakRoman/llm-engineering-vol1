"""Локальная публикация состояния; согласование участников задаётся отдельно."""
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory


def digest(data):
    return hashlib.sha256(data).hexdigest()


def atomic_write(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def publish(root, state):
    # Пример предполагает одного писателя и новый каталог состояния.
    root.mkdir()
    payload = json.dumps(state, sort_keys=True).encode('utf-8')
    atomic_write(root/'state.json', payload)
    manifest = {'format': 1, 'files': {'state.json': digest(payload)}}
    atomic_write(root/'manifest.json', json.dumps(manifest).encode())


def restore(root):
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest['format'] != 1:
        raise ValueError('Unsupported format')
    for name, expected in manifest['files'].items():
        if digest((root/name).read_bytes()) != expected:
            raise ValueError('Checksum mismatch')
    return json.loads((root/'state.json').read_text())


def run():
    state = {'weights': [1., 2.], 'moment1': [0.1, 0.2],
             'moment2': [0.01, 0.04], 'step': 7,
             'tokens': 1835008, 'data_cursor': {'shard': 3, 'offset': 17},
             'rng_seed': 42, 'recipe_version': 'toy-v1'}
    with TemporaryDirectory() as folder:
        root = Path(folder)/'checkpoint-7'
        publish(root, state)
        assert restore(root) == state
        (root/'state.json').write_bytes(b'corrupted')
        try:
            restore(root)
        except ValueError:
            print('Corrupted checkpoint rejected')
        else:
            raise AssertionError('Corruption accepted')
        incomplete = Path(folder)/'incomplete'
        incomplete.mkdir()
        (incomplete/'state.json').write_text('{}')
        try:
            restore(incomplete)
        except FileNotFoundError:
            print('Incomplete checkpoint rejected')
        else:
            raise AssertionError('Incomplete checkpoint accepted')


if __name__ == '__main__':
    run()
