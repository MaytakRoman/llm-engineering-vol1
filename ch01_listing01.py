import numpy as np


def stable_softmax(scores):
    shifted = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=-1, keepdims=True)


def full_attention(x, wq, wk, wv, causal=True):
    q, k, v = x @ wq, x @ wk, x @ wv
    scores = q @ k.T / np.sqrt(q.shape[-1])
    if causal:
        forbidden = np.triu_indices(len(x), k=1)
        scores[forbidden] = -np.inf
    return stable_softmax(scores) @ v


def incremental_attention(x, wq, wk, wv):
    keys, values, outputs = [], [], []
    for row in x:
        keys.append(row @ wk)
        values.append(row @ wv)
        q = row @ wq
        k = np.stack(keys)
        v = np.stack(values)
        scores = q @ k.T / np.sqrt(q.shape[-1])
        outputs.append(stable_softmax(scores) @ v)
    return np.stack(outputs)


def run_checks():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(6, 8))
    wq, wk, wv = [rng.normal(size=(8, 8)) for _ in range(3)]
    full = full_attention(x, wq, wk, wv)
    incremental = incremental_attention(x, wq, wk, wv)
    np.testing.assert_allclose(full, incremental,
                               atol=1e-12, rtol=1e-12)
    changed = x.copy()
    changed[4:] += 100
    causal_changed = full_attention(changed, wq, wk, wv)
    causal_error = np.max(np.abs(full[:4] - causal_changed[:4]))
    assert causal_error == 0.0
    open_base = full_attention(x, wq, wk, wv, causal=False)
    open_changed = full_attention(changed, wq, wk, wv, causal=False)
    negative_error = np.max(np.abs(open_base[:4] - open_changed[:4]))
    assert negative_error > 1e-6
    print('causal_error', causal_error)
    print('incremental_error', np.max(np.abs(full-incremental)))
    print('negative_control_error', negative_error)


if __name__ == '__main__':
    run_checks()
