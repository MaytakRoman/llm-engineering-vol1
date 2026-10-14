"""Решётка Unigram на символах; обученный токенизатор не используется."""
from math import exp, log

TEXT = "абв"
PROB = {
    "а": 0.20, "б": 0.15, "в": 0.15,
    "аб": 0.25, "бв": 0.20, "абв": 0.05,
}
assert abs(sum(PROB.values()) - 1.0) < 1e-12


def solve(text, prob):
    size = len(text)
    best = [float("-inf")] * (size + 1)
    mass = [0.0] * (size + 1)
    previous = [None] * (size + 1)
    best[0], mass[0] = 0.0, 1.0
    for end in range(1, size + 1):
        for token, weight in prob.items():
            start = end - len(token)
            if start < 0 or text[start:end] != token:
                continue
            score = best[start] + log(weight)
            mass[end] += mass[start] * weight
            if score > best[end]:
                best[end] = score
                previous[end] = (start, token)
    if previous[size] is None:
        raise ValueError("No complete segmentation")
    result, end = [], size
    while end:
        end, token = previous[end]
        result.append(token)
    return result[::-1], exp(best[size]), mass[size]


def enumerate_paths(text, prob):
    if not text:
        return [([], 1.0)]
    paths = []
    for token, weight in prob.items():
        if text.startswith(token):
            for tail, score in enumerate_paths(
                    text[len(token):], prob):
                paths.append(([token] + tail, weight * score))
    return paths


if __name__ == "__main__":
    tokens, score, total = solve(TEXT, PROB)
    paths = enumerate_paths(TEXT, PROB)
    assert tokens == ["абв"]
    assert abs(score - 0.05) < 1e-12
    assert abs(total - 0.132) < 1e-12
    assert abs(total - sum(p for _, p in paths)) < 1e-12
    assert "".join(tokens) == TEXT
    print(tokens, score, total)
    for path, probability in paths:
        print(path, probability, probability / total)
