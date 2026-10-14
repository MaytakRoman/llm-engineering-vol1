"""Синтетический стратифицированный аудит и упаковка документов."""
from math import sqrt


def wilson(successes, total, z=1.96):
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Invalid counts")
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    radius = z * sqrt(
        p * (1 - p) / total + z * z / (4 * total**2)
    ) / denom
    return center - radius, center + radius


def pack(documents, capacity):
    rows, row, group = [], [], []
    for doc_id, tokens in enumerate(documents):
        if len(tokens) > capacity:
            raise ValueError("Split long documents first")
        if len(row) + len(tokens) > capacity:
            rows.append((row, group))
            row, group = [], []
        row.extend(tokens)
        group.extend([doc_id] * len(tokens))
    if row:
        rows.append((row, group))
    return rows


def masks(group):
    attention = [
        [int(j <= i and group[i] == group[j])
         for j in range(len(group))]
        for i in range(len(group))
    ]
    # Выход позиции i предсказывает вход позиции i+1.
    loss = [
        int(i + 1 < len(group) and group[i] == group[i+1])
        for i in range(len(group))
    ]
    return attention, loss


if __name__ == "__main__":
    strata = [(0.8, 4, 200), (0.2, 30, 200)]
    estimate = sum(w * bad / n for w, bad, n in strata)
    assert abs(estimate - 0.046) < 1e-12
    print("Weighted bad fraction:", estimate)
    for weight, bad, total in strata:
        print(weight, bad / total, wilson(bad, total))
    rows = pack([[11, 12, 13], [21, 22]], 8)
    tokens, groups = rows[0]
    attention, loss = masks(groups)
    assert attention[3][0] == 0
    assert attention[4][3] == 1
    assert loss == [1, 1, 0, 1, 0]
    assert sum(loss) == 3
    print(tokens, groups, attention, loss)
