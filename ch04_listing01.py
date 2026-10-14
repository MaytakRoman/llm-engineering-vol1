"""Подгонка искусственной поверхности; языковая модель не обучается."""
import json
from pathlib import Path
import numpy as np


def surface(n, d):
    return 1.5 + 0.6 / np.sqrt(n) + 0.4 / np.sqrt(d)


def fit(points):
    n, d, observed = np.array(points).T
    best = None
    for alpha in np.linspace(0.35, 0.65, 31):
        for beta in np.linspace(0.35, 0.65, 31):
            design = np.column_stack([
                np.ones_like(n), n ** (-alpha), d ** (-beta)
            ])
            coef = np.linalg.lstsq(design, observed, rcond=None)[0]
            if np.any(coef <= 0):
                continue
            predicted = design @ coef
            error = float(np.mean((observed - predicted) ** 2))
            candidate = (error, alpha, beta, coef)
            if best is None or error < best[0]:
                best = candidate
    if best is None:
        raise ValueError("No admissible positive fit")
    error, alpha, beta, coef = best
    return dict(mse=error, alpha=float(alpha), beta=float(beta),
                E=float(coef[0]), A=float(coef[1]), B=float(coef[2]))


def predict(n, d, model):
    return (model['E'] + model['A'] / n ** model['alpha']
            + model['B'] / d ** model['beta'])


def run():
    points = []
    for n in [1., 2., 4., 8.]:
        for d in [2., 4., 8., 16.]:
            noise = 0.001 * np.sin(n + d)
            points.append([n, d, float(surface(n, d) + noise)])
    holdout = points[-1]
    model = fit(points[:-1])
    forecast = predict(holdout[0], holdout[1], model)
    assert abs(forecast - holdout[2]) < 0.01
    exact = fit([[n, d, float(surface(n, d))]
                 for n in [1., 2., 4., 8.]
                 for d in [2., 4., 8., 16.]])
    assert abs(exact['alpha'] - 0.5) < 1e-10
    assert abs(exact['beta'] - 0.5) < 1e-10
    optimum = 0.6 / 0.4 * np.sqrt(100.)
    assert abs(optimum - 15.) < 1e-10
    result = dict(model=model, points=points, holdout=holdout,
                  forecast=forecast, absolute_error=abs(forecast-holdout[2]))
    Path('results').mkdir(exist_ok=True)
    Path('results/scaling_result.json').write_text(
        json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    run()
