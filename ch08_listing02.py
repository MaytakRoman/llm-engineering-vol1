"""Маска потерь и адаптация малого ранга: численные проверки."""
import numpy as np

def masked_loss(logits, targets, mask):
    shifted = logits - logits.max(axis=-1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))
    nll = -logp[np.arange(len(targets)), targets]
    if mask.sum() == 0:
        raise ValueError('empty supervised target')
    return (nll * mask).sum() / mask.sum()

def run():
    rng = np.random.default_rng(81)
    logits = rng.normal(size=(7, 5)); targets = np.arange(7) % 5
    mask = np.array([0, 0, 0, 1, 1, 1, 1])
    original = masked_loss(logits, targets, mask)
    changed = logits.copy(); changed[:3] += rng.normal(size=(3, 5))*50
    assert masked_loss(changed, targets, mask) == original
    d_in, d_out, rank = 12, 10, 3
    x = rng.normal(size=(8, d_in))
    w = rng.normal(size=(d_out, d_in))
    a = rng.normal(size=(rank, d_in)); b = np.zeros((d_out, rank))
    np.testing.assert_allclose(x @ (w + b @ a).T, x @ w.T)
    target = rng.normal(size=(8, d_out))
    error = x @ (w + b @ a).T - target
    gradient = error.T @ x / len(x)
    grad_a, grad_b = b.T @ gradient, gradient @ a.T
    assert np.linalg.norm(grad_a) == 0
    assert np.linalg.norm(grad_b) > 0
    eps = 1e-6
    def loss(bb):
        residual = x @ (w + bb @ a).T - target
        return .5 * (residual**2).sum() / len(x)
    perturb = np.zeros_like(b); perturb[0, 0] = eps
    numerical = (loss(b + perturb) - loss(b - perturb)) / (2*eps)
    np.testing.assert_allclose(numerical, grad_b[0, 0], rtol=1e-5)
    print('masked loss', original)
    print('LoRA parameters', rank*(d_in+d_out))
    print('full parameters', d_in*d_out)

if __name__ == '__main__':
    run()
