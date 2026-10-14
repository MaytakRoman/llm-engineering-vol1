"""Векторный AdamW: epsilon вне корня и проверка возобновления."""
import copy
import numpy as np
from pathlib import Path
import json

class AdamW:

    def __init__(self, shape, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.step = 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def update(self, weights, gradient, learning_rate, decay):
        if not np.isfinite(gradient).all():
            raise ValueError('Nonfinite gradient; state is unchanged')
        self.step += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradient ** 2
        m = self.m / (1 - self.beta1 ** self.step)
        v = self.v / (1 - self.beta2 ** self.step)
        decayed = (1 - learning_rate * decay) * weights
        return decayed - learning_rate * m / (np.sqrt(v) + self.epsilon)

    def state(self):
        return copy.deepcopy(self.__dict__)

    def load(self, state):
        self.__dict__ = copy.deepcopy(state)

def run():
    opt = AdamW((2,))
    initial = np.array([2.0, -3.0])
    zero = np.zeros(2)
    changed = opt.update(initial, zero, 0.01, 0.1)
    np.testing.assert_allclose(changed, initial * 0.999)
    weights = np.array([3.0, -2.0])
    curvature = np.array([1.0, 10.0])
    opt = AdamW(weights.shape)
    trace = []
    for step in range(200):
        gradient = curvature * weights
        before = weights.copy()
        weights = opt.update(weights, gradient, 0.05, 0.01)
        loss = float(0.5 * np.sum(curvature * weights ** 2))
        relative = float(
            np.linalg.norm(weights - before) /
            (np.linalg.norm(before) + 1e-12)
        )
        trace.append([step + 1, loss, relative])
        if step == 99:
            saved_weights = weights.copy()
            saved_state = opt.state()
    restored = AdamW(weights.shape)
    restored.load(saved_state)
    resumed = saved_weights
    for step in range(100):
        resumed = restored.update(resumed, curvature * resumed, 0.05, 0.01)
    np.testing.assert_array_equal(resumed, weights)
    assert trace[-1][1] < trace[0][1]
    Path('results').mkdir(exist_ok=True)
    Path('results/optimizer_result.json').write_text(json.dumps(trace))
    print('Final synthetic loss:', trace[-1][1])
    print('Resume is identical')
if __name__ == '__main__':
    run()
