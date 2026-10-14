"""Синтетическое точное извлечение: проверка протокола оценки."""
import random
from math import sqrt

def wilson(correct, count, z=1.96):
    if count <= 0:
        raise ValueError('positive count required')
    p = correct / count
    den = 1 + z*z / count
    center = (p + z*z / (2*count)) / den
    radius = z * sqrt(p*(1-p)/count + z*z/(4*count*count)) / den
    return center - radius, center + radius

def case(length, position, seed):
    rng = random.Random(seed)
    words = [str(rng.randrange(10000)) for _ in range(length)]
    secret = f'K{rng.getrandbits(64):016x}'
    offset = min(length - 1, int(position * length))
    words[offset] = 'answer=' + secret
    return words, secret

def retrieve(words, limit=None):
    visible = words if limit is None else words[-limit:]
    matches = [w.split('=', 1)[1] for w in visible
               if w.startswith('answer=')]
    return matches[-1] if matches else None

def run():
    for length in [256, 1024, 4096]:
        row = []
        for position in [.1, .5, .9]:
            cases = [case(length, position, s) for s in range(50)]
            full = sum(retrieve(w) == a for w, a in cases)
            tail = sum(retrieve(w, 512) == a for w, a in cases)
            assert full == 50
            row.append(tail / 50)
        print('length', length, 'tail accuracy', row)
    low, high = wilson(45, 50)
    assert low < .9 < high
    print('Wilson 45/50', low, high)

if __name__ == '__main__':
    run()
