import re
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np

weights = np.load('results/case_weights.npz')
E, W, b, U, v = [weights[k] for k in ('E','W','b','U','v')]
C = W.shape[0] // E.shape[1]
labels = ('NET', 'DSK', 'ACC')
cues = ('сеть', 'диск', 'доступ')

def score(pre, label):
    text = pre + label.encode() + b'\n'
    total = 0.0
    for i in range(len(pre), len(text)):
        win = list(text[max(0, i-C):i])
        x = np.array([0]*(C-len(win)) + win)
        h = np.tanh(E[x].reshape(-1) @ W + b)
        a = h @ U + v
        a -= a.max()
        total += a[text[i]] - np.log(np.exp(a).sum())
    return float(total)

def service(text):
    if not isinstance(text, str) or not text.strip():
        return {'status': 'invalid', 'class': None}
    if len(text.encode()) > 300:
        return {'status': 'invalid', 'class': None}
    found = [c for c in cues
             if re.search(r'\b' + re.escape(c) + r'\b',
                          text.lower())]
    if len(found) != 1:
        return {'status': 'review', 'class': None}
    pre = ('Заявка: ' + text + '\nКласс: ').encode()
    values = [score(pre, label) for label in labels]
    return {'status': 'accepted',
            'class': labels[int(np.argmax(values))]}

for text in ('', 'x'*301):
    assert service(text)['status'] == 'invalid'
for text in ('неизвестный узел', 'сеть и диск'):
    assert service(text)['status'] == 'review'
assert service('Не работает сеть на узле 3.')['class'] == 'NET'
# Слово «недоступен» не создаёт второго компонента.
assert service('На устройстве 1 недоступен сеть.')['class'] == 'NET'
assert service('На устройстве 1 недоступен диск.')['class'] == 'DSK'
assert service('недоступен')['status'] == 'review'
print('controller: 8/8')
