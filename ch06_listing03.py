"""Дисперсия выходной проекции; полная схема μP не реализована."""
import json
from pathlib import Path
import numpy as np

def run():
    rng = np.random.default_rng(42)
    results = []
    for width in [16, 64, 256, 1024]:
        raw, normalized = [], []
        for chunk in range(40):
            hidden = rng.normal(size=(500, width))
            readout = rng.normal(size=(500, width))
            output = np.sum(hidden * readout, axis=1)
            raw.extend(output)
            normalized.extend(output / np.sqrt(width))
        raw_variance = float(np.var(raw, ddof=1))
        normalized_variance = float(np.var(normalized, ddof=1))
        assert abs(normalized_variance - 1.0) < 0.06
        assert abs(raw_variance / width - 1.0) < 0.06
        results.append(dict(
            width=width, raw=raw_variance,
            normalized=normalized_variance
        ))
    Path('results').mkdir(exist_ok=True)
    Path('results/width_result.json').write_text(json.dumps(results))
    print(json.dumps(results, indent=2))
if __name__ == '__main__':
    run()
