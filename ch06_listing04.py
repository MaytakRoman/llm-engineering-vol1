"""Горизонт в токенах и точное уменьшение скалярного веса."""
from math import log, exp


def half_tokens(beta, batch_tokens):
    assert 0 < beta < 1 and batch_tokens > 0
    return log(0.5) / log(beta) * batch_tokens


def beta_for_batch(beta, old_batch, new_batch):
    return beta ** (new_batch / old_batch)


def main():
    old_batch = 131_072
    new_batch = 262_144
    beta = 0.999
    transferred = beta_for_batch(beta, old_batch, new_batch)
    original_horizon = half_tokens(beta, old_batch)
    new_horizon = half_tokens(transferred, new_batch)
    assert abs(original_horizon / new_horizon - 1) < 1e-10
    print("beta", transferred)
    print("half tokens", original_horizon)
    old_steps = 1000
    new_steps = 500
    rate = 0.001
    decay = 0.1
    old_factor = (1 - rate * decay) ** old_steps
    new_decay = (1 - (1 - rate * decay) ** 2) / rate
    new_factor = (1 - rate * new_decay) ** new_steps
    assert abs(old_factor - new_factor) < 1e-12
    print("decay", new_decay, "factor", old_factor)
    # Положительная корреляция координат меняет дисперсию суммы.
    rho = 0.1
    for width in [16, 64, 256]:
        variance = 1 + (width - 1) * rho
        assert variance > 1
        print("width", width, "normalized variance", variance)


if __name__ == "__main__":
    main()
