"""Обучение линейного остатка ранга два на синтетических данных."""
import numpy as np

def run():
    rng = np.random.default_rng(82)
    din, dout, rank = 8, 6, 2
    x = rng.normal(size=(128, din))
    base = rng.normal(size=(dout, din))*.1
    truth = rng.normal(size=(dout,rank)) @ rng.normal(size=(rank,din))*.1
    target = x @ (base+truth).T
    a = rng.normal(size=(rank,din))*.1
    b = np.zeros((dout,rank))
    start = None
    for step in range(2000):
        error = x @ (base+b@a).T-target
        value = .5*(error**2).sum()/len(x)
        if start is None:
            start = value
        gradient = error.T@x/len(x)
        ga, gb = b.T@gradient, gradient@a.T
        a -= .05*ga; b -= .05*gb
    final = .5*((x@(base+b@a).T-target)**2).sum()/len(x)
    holdout = rng.normal(size=(64,din))
    relative = np.linalg.norm(holdout@(truth-b@a).T)
    relative /= np.linalg.norm(holdout@truth.T)
    assert final < start/10000
    assert relative < .01
    np.testing.assert_allclose(x@base.T+(x@a.T)@b.T,
                               x@(base+b@a).T, atol=1e-12)
    print('start', start, 'final', final)
    print('holdout relative', relative)
    print('trainable', rank*(din+dout))

if __name__ == '__main__':
    run()
