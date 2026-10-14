import numpy as np

a = np.array([1, 1, 0, 0, 1, 0, 1, 0, 1, 0])
b = np.array([1, 0, 0, 0, 1, 1, 1, 0, 1, 0])
observed = np.mean(a == b)
chance = a.mean()*b.mean() + (1-a.mean())*(1-b.mean())
kappa = (observed-chance)/(1-chance)
assert abs(observed-0.8) < 1e-12
assert abs(kappa-0.6) < 1e-12
print("agreement", observed, "kappa", kappa)
