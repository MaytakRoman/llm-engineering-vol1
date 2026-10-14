"""Двухвариантный DPO, проверка градиента и ограничение отношения в PPO."""
import math
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def loss(theta, reference, beta):
    return np.logaddexp(0, -beta * (theta-reference))

def ppo(ratio, advantage, epsilon=.2):
    clipped = np.clip(ratio, 1-epsilon, 1+epsilon)
    return min(ratio*advantage, clipped*advantage)

def run():
    reference, theta, beta = -.3, -.3, .5
    start = loss(theta, reference, beta)
    for step in range(100):
        grad = -beta * sigmoid(-beta*(theta-reference))
        epsilon = 1e-6
        numerical = (loss(theta+epsilon, reference, beta)
                     - loss(theta-epsilon, reference, beta)) / (2*epsilon)
        np.testing.assert_allclose(numerical, grad, atol=1e-8)
        theta -= .1 * grad
    assert loss(theta, reference, beta) < start
    assert sigmoid(theta) > sigmoid(reference)
    assert math.isclose(ppo(1.5, 1), 1.2, rel_tol=1e-12)
    assert math.isclose(ppo(.5, -1), -.8, rel_tol=1e-12)
    rewards = np.array([0., 1., 1., 0.])
    advantages = (rewards-rewards.mean()) / rewards.std()
    assert abs(advantages.mean()) < 1e-12
    assert np.all(np.isfinite(advantages))
    print('initial DPO loss', start)
    print('final DPO loss', loss(theta, reference, beta))
    print('preferred action probability', sigmoid(theta))

if __name__ == '__main__':
    run()
