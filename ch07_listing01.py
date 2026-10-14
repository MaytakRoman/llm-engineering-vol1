"""Блочное причинное внимание с онлайн-нормализацией: сравнение с плотным эталоном."""
import numpy as np

def attention(q, k, v, tile=7):
    out = np.zeros_like(v)
    for i in range(len(q)):
        maximum, total = -np.inf, 0.0
        acc = np.zeros(v.shape[1])
        for start in range(0, i + 1, tile):
            end = min(start + tile, i + 1)
            scores = k[start:end] @ q[i] / np.sqrt(q.shape[1])
            new_max = max(maximum, scores.max())
            old_scale = np.exp(maximum - new_max)
            weights = np.exp(scores - new_max)
            acc = old_scale * acc + weights @ v[start:end]
            total = old_scale * total + weights.sum()
            maximum = new_max
        out[i] = acc / total
    return out

def dense(q, k, v):
    scores = q @ k.T / np.sqrt(q.shape[1])
    scores[np.triu_indices(len(q), 1)] = -np.inf
    weights = np.exp(scores - scores.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    return weights @ v

def run():
    rng = np.random.default_rng(71)
    q, k, v = [rng.normal(size=(31, 16)) for _ in range(3)]
    reference = dense(q, k, v)
    for tile in [1, 7, 16, 64]:
        np.testing.assert_allclose(attention(q, k, v, tile),
                                   reference, atol=1e-12)
    altered = v.copy(); altered[-1] += 100
    np.testing.assert_allclose(dense(q, k, altered)[:-1],
                               reference[:-1], atol=1e-12)
    print('max error', abs(attention(q, k, v) - reference).max())

if __name__ == '__main__':
    run()
