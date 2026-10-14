from collections import Counter

RU = [
    'Модель обрабатывает документ и формирует ответ.',
    'Проверка качества данных выполняется до обучения.',
    'Архитектура системы зависит от бюджета и нагрузки.',
    'Токенизатор сохраняет исходный текст без потерь.'
]
EN = [
    'The model processes a document and generates an answer.',
    'Data quality is checked before training.',
    'System architecture depends on budget and workload.',
    'The tokenizer preserves the original text without loss.'
]
TEST_RU = [
    'Модель сохраняет документ и проверяет ответ.',
    'Бюджет обучения зависит от данных.',
    'Проверка нагрузки выполняется до обработки текста.'
]
TEST_EN = [
    'The model preserves a document and checks the answer.',
    'Training budget depends on data.',
    'Workload is checked before processing text.'
]


def merge(seq, pair):
    out, i = [], 0
    while i < len(seq):
        if i+1 < len(seq) and (seq[i], seq[i+1]) == pair:
            out.append(seq[i] + seq[i+1])
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


def train(texts, steps=120):
    seqs = [[bytes([b]) for b in t.encode('utf-8')]
            for t in texts]
    rules = []
    for _ in range(steps):
        counts = Counter(pair for seq in seqs
                         for pair in zip(seq, seq[1:]))
        if not counts:
            break
        pair = min(counts, key=lambda x: (-counts[x], x))
        rules.append(pair)
        seqs = [merge(seq, pair) for seq in seqs]
    return rules


def encode(text, rules):
    seq = [bytes([b]) for b in text.encode('utf-8')]
    for pair in rules:
        seq = merge(seq, pair)
    restored = b''.join(seq).decode('utf-8')
    assert restored == text
    return seq


if __name__ == '__main__':
    for ru_repeat, en_repeat in [(9, 1), (5, 5), (1, 9)]:
        rules = train(RU*ru_repeat + EN*en_repeat)
        lengths = [sum(len(encode(t, rules)) for t in tests)
                   for tests in [TEST_RU, TEST_EN]]
        print(ru_repeat, en_repeat, lengths)
