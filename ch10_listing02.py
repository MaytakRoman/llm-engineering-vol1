import numpy as np

def speculative_distribution(p, q):
    accepted = np.minimum(p, q)
    rejection = 1-accepted.sum()
    residual = np.maximum(p-q, 0)
    if rejection > 1e-12:
        residual /= residual.sum()
    else:
        residual[:] = 0
    return accepted + rejection*residual

pairs = [(np.array([.2, .5, .3]), np.array([.6, .2, .2])),
         (np.array([0., 1., 0.]), np.array([.5, 0., .5])),
         (np.array([.2, .5, .3]), np.array([.2, .5, .3]))]
for p, q in pairs:
    np.testing.assert_allclose(speculative_distribution(p, q), p,
                               atol=1e-12)
    print("acceptance", np.minimum(p, q).sum())
