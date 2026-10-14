import math

def cost(rate, count, hours, utilization, tokens, overhead):
    if tokens <= 0 or not 0 < utilization <= 1:
        raise ValueError("invalid denominator")
    gpu = rate*count*hours/utilization
    return (gpu+overhead)/tokens*1e6

# Все тарифы и денежные значения условны.
for utilization in (.25, .5, .75):
    print(utilization, cost(2, 4, 10, utilization, 20_000_000, 80))
assert math.isclose(cost(2, 4, 10, .5, 20_000_000, 80), 12,
                    rel_tol=1e-12)
