"""Расчёт спецификации; обучение на GPU не выполняется."""
from math import ceil


def parameter_count(layers=24, width=2048, mlp=4096, vocab=32000):
    return (vocab*width + layers*(4*width**2 + 3*width*mlp
                                 + 2*width) + width)


def plan(gpus):
    if gpus not in [8, 16, 32, 64]:
        raise ValueError('Unsupported tutorial topology')
    dp, tp, pp, micro, sequence = gpus, 1, 1, 1, 2048
    accumulation = 128//dp
    batch = dp*micro*accumulation*sequence
    parameters = parameter_count()
    target_tokens = 20*10**9
    operations = 6*parameters*target_tokens
    seconds = operations/(gpus*10**14*0.4)
    assert batch == 262144
    assert parameters == 1072269312
    return dict(gpus=gpus, dp=dp, tp=tp, pp=pp,
                accumulation=accumulation, batch_tokens=batch,
                parameters=parameters, steps=ceil(target_tokens/batch),
                state_gib=16*parameters/2**30,
                operations=operations, ideal_hours=seconds/3600)


if __name__ == '__main__':
    for gpu_count in [8, 16, 32, 64]:
        print(plan(gpu_count))
