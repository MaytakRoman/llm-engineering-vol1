"""Память состояния без активаций и временной сборки шардов."""
from math import sqrt


def memory(parameters, replicas):
    n, d = parameters, replicas
    if min(n, d) <= 0:
        raise ValueError('Invalid shape')
    return {'replicated': 16*n, 'zero1': n*(4+12/d),
            'zero2': n*(2+14/d), 'zero3': 16*n/d}


def global_clip(shards, threshold):
    squared_norm = sum(value*value for shard in shards
                       for value in shard)
    norm = sqrt(squared_norm)
    factor = min(1., threshold/norm) if norm else 1.
    return [[factor*v for v in shard] for shard in shards], norm


def run():
    for replicas in [1, 8, 16, 32, 64]:
        values = memory(10**9, replicas)
        print(replicas, {k: v/2**30 for k, v in values.items()})
        assert values['zero3'] <= values['zero2']
        assert values['zero2'] <= values['zero1']
    clipped, norm = global_clip([[3.], [4.]], 2.)
    assert norm == 5.
    assert abs(clipped[0][0] - 1.2) < 1e-12
    assert abs(clipped[1][0] - 1.6) < 1e-12
    assert global_clip([[0.], [0.]], 2.)[0] == [[0.], [0.]]
    save_seconds, mtbf_seconds = 60., 86400.
    optimum = sqrt(2*save_seconds*mtbf_seconds)
    assert abs(optimum - 3219.9378876) < 1e-6
    print('Checkpoint interval, seconds:', optimum)


if __name__ == '__main__':
    run()
