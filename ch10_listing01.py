"""Групповая симметричная квантизация и ошибка линейного выхода."""
import numpy as np

def quantize(w, bits, group):
    limit = 2**(bits-1)-1
    restored = np.zeros_like(w)
    for start in range(0, w.shape[1], group):
        block = w[:, start:start+group]
        scale = np.maximum(abs(block).max(axis=1, keepdims=True), 1e-12)
        scale /= limit
        integer = np.clip(np.rint(block/scale), -limit, limit)
        restored[:, start:start+group] = integer * scale
    return restored

def run():
    rng = np.random.default_rng(101)
    w = rng.normal(size=(12, 64)); x = rng.normal(size=(20, 64))
    for bits in [4, 8]:
        for group in [16, 64]:
            restored = quantize(w, bits, group)
            relative = np.linalg.norm(x @ (w-restored).T)
            relative /= np.linalg.norm(x @ w.T)
            print('bits', bits, 'group', group, 'relative error', relative)

if __name__ == '__main__':
    run()
