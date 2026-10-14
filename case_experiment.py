"""Полный контрольный опыт: данные, обучение, оценка и сохранение."""
import hashlib
import json
import os
import platform
import re
from pathlib import Path

os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
rng = np.random.default_rng(20261006)
CONTEXT, EMBEDDING, HIDDEN, VOCAB = 128, 8, 64, 256
labels = ('NET', 'DSK', 'ACC')
cues = ('сеть', 'диск', 'доступ')
forms = (
    'Не работает {c} на узле {n}.',
    'На устройстве {n} недоступен {c}.',
    'Проверьте {c}, устройство {n}.',
    'Узел {n}: отказ, компонент {c}.',
    'Компонент {c} требует проверки: узел {n}.',
    'Сообщение с узла {n}: проблема, {c}.',
    'Узел {n}, компонент {c} неисправен.',
    'После запуска на узле {n} не работает {c}.',
    'Обнаружен отказ: {c}; устройство {n}.',
    'Просьба проверить узел {n}: недоступен {c}.',
)
sets = [[], [], []]
for f, form in enumerate(forms):
    split = 0 if f < 6 else 1 if f < 8 else 2
    for j, c in enumerate(cues):
        for n in range(8):
            prompt = 'Заявка: ' + form.format(c=c, n=n) + '\nКласс: '
            sets[split].append((prompt.encode(), labels[j].encode() + b'\n',
                f, j, n))
embedding_weights = rng.normal(0, 0.08, (VOCAB, EMBEDDING))
hidden_weights = rng.normal(0, 0.03, (CONTEXT * EMBEDDING, HIDDEN))
hidden_bias = np.zeros(HIDDEN)
output_weights = rng.normal(0, 0.03, (HIDDEN, VOCAB))
output_bias = np.zeros(VOCAB)
params = [embedding_weights, hidden_weights, hidden_bias, output_weights,
    output_bias]
first_moments = [np.zeros_like(x) for x in params]
second_moments = [np.zeros_like(x) for x in params]
step = 0

def contexts(records, answer_only):
    xx = []
    yy = []
    for prefix, answer, *_ in records:
        text = prefix + answer
        for i in range(1, len(text)):
            if answer_only and i < len(prefix):
                continue
            win = list(text[max(0, i - CONTEXT):i])
            xx.append([0] * (CONTEXT - len(win)) + win)
            yy.append(text[i])
    return np.array(xx), np.array(yy)

def forward(x):
    z = embedding_weights[x].reshape(len(x), -1)
    h = np.tanh(z @ hidden_weights + hidden_bias)
    a = h @ output_weights + output_bias
    a -= a.max(1, keepdims=True)
    p = np.exp(a)
    p /= p.sum(1, keepdims=True)
    return z, h, p

def update(x, y):
    global step
    z, h, p = forward(x)
    loss = -np.log(p[np.arange(len(y)), y]).mean()
    g = p.copy()
    g[np.arange(len(y)), y] -= 1
    g /= len(y)
    gu = h.T @ g
    gv = g.sum(0)
    dh = g @ output_weights.T * (1 - h * h)
    gw = z.T @ dh
    gb = dh.sum(0)
    de = np.zeros_like(embedding_weights)
    np.add.at(de, x.ravel(), (dh @ hidden_weights.T).reshape(-1, EMBEDDING))
    grads = [de, gw, gb, gu, gv]
    step += 1
    for a, g, m, s in zip(params, grads, first_moments, second_moments):
        np.clip(g, -1, 1, out=g)
        m *= 0.9
        m += 0.1 * g
        s *= 0.999
        s += 0.001 * g * g
        corrected_first = m / (1 - 0.9 ** step)
        corrected_second = s / (1 - 0.999 ** step)
        a -= 0.003 * corrected_first / (np.sqrt(corrected_second) + 1e-8)
    return float(loss)

def mean_loss(records):
    x, y = contexts(records, True)
    total = 0
    for i in range(0, len(y), 128):
        _, _, p = forward(x[i:i + 128])
        ii = np.arange(len(p))
        total += float(-np.log(p[ii, y[i:i + 128]]).sum())
    return total / len(y)

def predict(prefix):
    scores = []
    for label in labels:
        rec = [(prefix, label.encode() + b'\n')]
        x, y = contexts(rec, True)
        _, _, p = forward(x)
        scores.append(float(np.log(p[np.arange(len(y)), y]).sum()))
    return int(np.argmax(scores))
initial = mean_loss(sets[2])
trace = []
x, y = contexts(sets[0], False)
for it in range(300):
    ids = rng.integers(len(y), size=128)
    loss = update(x[ids], y[ids])
    if it % 50 == 0:
        trace.append(['pretrain', it, loss])
x, y = contexts(sets[0], True)
for it in range(600):
    ids = rng.integers(len(y), size=128)
    loss = update(x[ids], y[ids])
    if it % 100 == 0:
        trace.append(['adapt', it, loss])
conf = np.zeros((3, 3), int)
preds = []
for rec in sets[2]:
    guess = predict(rec[0])
    conf[rec[3], guess] += 1
    preds.append(guess)
val = sum(predict(r[0]) == r[3] for r in sets[1]) / len(sets[1])
corpus_records = [
    [(p.decode('utf-8'), a.decode('utf-8'), f, j, n)
     for p, a, f, j, n in split]
    for split in sets
]
corpus_json = json.dumps(
    corpus_records, ensure_ascii=False, separators=(',', ':')
).encode('utf-8')
result = {
    'seed': 20261006,
    'context': CONTEXT,
    'embedding': EMBEDDING,
    'hidden': HIDDEN,
    'vocab': VOCAB,
    'parameters': sum(a.size for a in params),
    'split': list(map(len, sets)),
    'initial_answer_nll': initial,
    'final_answer_nll': mean_loss(sets[2]),
    'validation_accuracy': val,
    'test_correct': int(np.trace(conf)),
    'test_total': len(sets[2]),
    'confusion': conf.tolist(),
    'trace': trace,
    'corpus_sha256': hashlib.sha256(corpus_json).hexdigest(),
}
Path('results').mkdir(exist_ok=True)
Path('results/case_result.json').write_text(json.dumps(result, indent=2))
np.savez('results/case_weights.npz', E=embedding_weights, W=hidden_weights,
    b=hidden_bias, U=output_weights, v=output_bias)
challenge_forms = [
    ('Нет соединения с сервером на узле {n}.', 0, 'синоним'),
    ('Не читаются файлы с накопителя на узле {n}.', 1, 'синоним'),
    ('Учетная запись не позволяет войти на узле {n}.', 2, 'синоним'),
    ('Сеть исправна, но диск неисправен на узле {n}.', 1, 'отрицание'),
    ('Диск исправен, но доступ закрыт на узле {n}.', 2, 'отрицание'),
    ('Доступ разрешен, но сеть неисправна на узле {n}.', 0, 'отрицание'),
    ('Узел {n} не получает пакеты от сервера.', 0, 'описание'),
    ('Узел {n} выдает ошибку чтения секторов.', 1, 'описание'),
    ('Узел {n} отклоняет вход пользователя.', 2, 'описание'),
]

def keyword_rule(prompt):
    text = prompt.decode('utf-8').lower()
    matches = [
        i for i, cue in enumerate(cues)
        if re.search(r'\b' + re.escape(cue) + r'\b', text)
    ]
    return matches[0] if len(matches) == 1 else -1
challenge = []
for family, (form, target_class, kind) in enumerate(challenge_forms):
    for node in range(8):
        prompt = ('Заявка: ' + form.format(n=node) + '\nКласс: ').encode()
        challenge.append({
            'family': family, 'node': node, 'kind': kind,
            'target': target_class, 'prompt': prompt.decode(),
            'network': predict(prompt), 'rule': keyword_rule(prompt),
        })
result['rule_original_correct'] = sum(
    keyword_rule(r[0]) == r[3] for r in sets[2]
)
result['challenge'] = {}
for kind in ('синоним', 'отрицание', 'описание'):
    rows = [r for r in challenge if r['kind'] == kind]
    result['challenge'][kind] = {
        'total': len(rows),
        'network_correct': sum(r['network'] == r['target'] for r in rows),
        'rule_correct': sum(r['rule'] == r['target'] for r in rows),
        'rule_abstain': sum(r['rule'] == -1 for r in rows),
    }
result['challenge_records'] = challenge
result['environment'] = {
    'python': platform.python_version(), 'numpy': np.__version__,
}
Path('results/case_result.json').write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8'
)
summary = {key: value for key, value in result.items()
           if key != 'challenge_records'}
print(json.dumps(summary, ensure_ascii=False, indent=2))
