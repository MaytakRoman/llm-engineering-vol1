"""Число страниц KV-кэша и внутренний остаток выделенной памяти."""
from math import ceil

def pages(lengths, block):
    if block <= 0 or any(n < 0 for n in lengths):
        raise ValueError('invalid length or block')
    return sum(ceil(n/block) for n in lengths)

def run():
    lengths = [17, 33, 65, 129]
    actual = sum(lengths)
    for block in [8,16,32,64]:
        allocated = pages(lengths, block) * block
        print('block', block, 'allocated', allocated,
              'waste', allocated-actual)
        assert actual <= allocated < actual+len(lengths)*block

if __name__ == '__main__':
    run()
