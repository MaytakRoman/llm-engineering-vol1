"""Нормированные расписания; качество языковой модели не измеряется."""
from math import cos, pi, log1p, exp
import json
from pathlib import Path


def schedule(progress, kind='cosine', warm=0.05, minimum=0.1):
    x = min(1., max(0., progress))
    if x <= warm:
        return x/warm
    if kind == 'cosine':
        z = (x-warm)/(1-warm)
        return minimum+(1-minimum)*(1+cos(pi*z))/2
    if kind == 'stable':
        if x <= 0.8:
            return 1.
        z = (x-0.8)/0.2
        return minimum+(1-minimum)*(1+cos(pi*z))/2
    raise ValueError('Unknown schedule')


def run():
    result = {}
    for kind in ['cosine', 'stable']:
        values = [schedule(t/1000, kind) for t in range(1001)]
        assert values[0] == 0.
        assert abs(values[-1]-0.1) < 1e-12
        assert min(values) >= 0. and max(values) <= 1.
        tail = values[800:]
        assert all(a >= b for a, b in zip(tail, tail[1:]))
        rates = [0.001*v for v in values[1:]]
        decay = 0.1
        exact = exp(sum(log1p(-decay*rate) for rate in rates))
        approximate = exp(-decay*sum(rates))
        assert abs(exact-approximate) < 1e-5
        result[kind] = dict(area=sum(rates), exact=exact,
                            approximate=approximate, values=values)
        print(kind, result[kind]['area'], exact, approximate)
    assert result['stable']['exact'] < result['cosine']['exact']
    Path('results').mkdir(exist_ok=True)
    Path('results/schedule_result.json').write_text(json.dumps(result))


if __name__ == '__main__':
    run()
