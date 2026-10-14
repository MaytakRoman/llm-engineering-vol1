"""Расчёт ресурсной заявки; измерения оборудования не выполняются."""
from math import ceil


def budget(parameters, tokens, gpus, peak_flops, utilization,
           batch_tokens, rate_per_gpu_hour):
    if min(parameters, tokens, gpus, peak_flops, batch_tokens) <= 0:
        raise ValueError("Inputs must be positive")
    if not 0 < utilization <= 1:
        raise ValueError("Invalid utilization")
    operations = 6 * parameters * tokens
    seconds = operations / (gpus * peak_flops * utilization)
    steps = ceil(tokens / batch_tokens)
    return dict(operations=operations, hours=seconds / 3600,
                gpu_hours=seconds * gpus / 3600,
                steps=steps, full_batch_tokens=steps * batch_tokens,
                cost=seconds * gpus / 3600 * rate_per_gpu_hour)


def run():
    for gpus in [8, 16, 32, 64]:
        result = budget(10**9, 20*10**9, gpus,
                        10**14, 0.4, 262144, 1.)
        assert result['operations'] == 12 * 10**19
        assert result['steps'] == 76294
        assert result['full_batch_tokens'] == 20000014336
        assert abs(result['gpu_hours'] - 833.3333333333) < 1e-6
        print(gpus, result)
    extra_train_cost = 12000.
    saving_per_token = 0.000002
    break_even = extra_train_cost / saving_per_token
    assert abs(break_even - 6e9) < 1e-3
    print('Break even tokens:', break_even)


if __name__ == '__main__':
    run()
