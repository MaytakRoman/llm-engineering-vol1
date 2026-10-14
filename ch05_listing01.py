"""Эталон на CPU для нормировки и тензорного разбиения."""
import numpy as np


def gradient_sum(theta, x, y):
    error = x @ theta - y
    return x.T @ error, len(x)


def rank_mesh(data_degree, tensor_degree):
    ranks = np.arange(data_degree * tensor_degree)
    mesh = ranks.reshape(data_degree, tensor_degree)
    tensor_groups = [list(row) for row in mesh]
    data_groups = [list(col) for col in mesh.T]
    assert sorted(sum(tensor_groups, [])) == list(ranks)
    assert sorted(sum(data_groups, [])) == list(ranks)
    return tensor_groups, data_groups


def run():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(8, 4))
    y = rng.normal(size=8)
    theta = rng.normal(size=4)
    full, count = gradient_sum(theta, x, y)
    full /= count
    g1, t1 = gradient_sum(theta, x[:2], y[:2])
    g2, t2 = gradient_sum(theta, x[2:], y[2:])
    correct = (g1 + g2) / (t1 + t2)
    wrong = (g1 / t1 + g2 / t2) / 2
    np.testing.assert_allclose(correct, full, atol=1e-12)
    assert np.linalg.norm(wrong - full) > 1e-6
    # DDP усредняет два локальных градиента.
    ddp_local = [2*g1/(t1+t2), 2*g2/(t1+t2)]
    np.testing.assert_allclose(sum(ddp_local)/2, full)
    matrix = rng.normal(size=(4, 6))
    expected = x @ matrix
    column = np.concatenate([x @ matrix[:, :3],
                             x @ matrix[:, 3:]], axis=1)
    row = x[:, :2] @ matrix[:2] + x[:, 2:] @ matrix[2:]
    np.testing.assert_allclose(column, expected)
    np.testing.assert_allclose(row, expected)
    tensor_groups, data_groups = rank_mesh(4, 2)
    print('Correct global gradient:', correct)
    print('Wrong averaging error:', np.linalg.norm(wrong-full))
    print('Tensor groups:', tensor_groups)
    print('Data groups:', data_groups)


if __name__ == '__main__':
    run()
