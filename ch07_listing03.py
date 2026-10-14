"""Допуск запросов по полезной памяти KV-кэша с резервом продолжения."""
from dataclasses import dataclass

@dataclass
class Request:
    prompt: int
    output: int
    priority: int = 0

def admit(requests, kv_per_token, capacity):
    selected, rejected, used = [], [], 0
    for request in sorted(requests, key=lambda x: -x.priority):
        reserve = (request.prompt+request.output)*kv_per_token
        if used+reserve <= capacity:
            selected.append(request); used += reserve
        else:
            rejected.append(request)
    return selected, rejected, used

def run():
    # 32 слоя, 8 KV-голов, размерность 128, BF16.
    per_token = 2*32*8*128*2
    assert per_token == 131072
    requests = [Request(8192,2048,1), Request(16384,4096),
                Request(32768,4096), Request(4096,1024,2)]
    usable = 6*2**30
    selected, rejected, used = admit(requests, per_token, usable)
    assert used <= usable
    assert len(selected)+len(rejected) == len(requests)
    print('bytes per token', per_token)
    print('admitted', [(r.prompt,r.output) for r in selected])
    print('rejected', [(r.prompt,r.output) for r in rejected])
    print('reserved GiB', used/2**30)
    # Резерв только под запрос скрывает будущий рост состояния.
    prompt_only = sum(r.prompt*per_token for r in selected)
    assert prompt_only < used
    print('missing output reserve GiB', (used-prompt_only)/2**30)

if __name__ == '__main__':
    run()
