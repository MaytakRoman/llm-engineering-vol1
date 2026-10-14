"""Контрфактические пары выявляют предпочтение длины у косвенной метрики."""
import numpy as np

def proxy(correct, length):
    return 2*correct + .02*length

def target(correct, length):
    return correct - .0001*max(length-50,0)

def run():
    pairs = [((1,30),(1,200)), ((1,30),(0,200)),
             ((1,150),(0,30)), ((0,30),(0,200))]
    disagreements = 0
    for left,right in pairs:
        p = proxy(*left) > proxy(*right)
        t = target(*left) > target(*right)
        disagreements += p != t
        print(left,right,'proxy left',p,'target left',t)
    assert disagreements == 3

if __name__ == '__main__':
    run()
