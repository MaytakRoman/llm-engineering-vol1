"""Детерминированная очередь одного сервера; GPU не моделируется."""
import numpy as np

def fcfs(arrivals, durations):
    finish = 0.; waiting = []; completion = []
    for arrival,duration in zip(arrivals,durations):
        start = max(arrival,finish)
        waiting.append(start-arrival)
        finish = start+duration
        completion.append(finish-arrival)
    return np.array(waiting),np.array(completion)

def run():
    durations = np.array([.1]*9+[1.])*10
    durations = np.tile(durations,10)
    # Одинаковые времена обслуживания, два темпа поступления.
    for interval in [2.5,1.5]:
        arrivals = np.arange(len(durations))*interval
        waiting,completion = fcfs(arrivals,durations)
        assert np.all(waiting >= 0)
        np.testing.assert_allclose(completion,waiting+durations)
        print('arrival interval',interval,
              'mean waiting',waiting.mean(),
              'p95 completion',np.quantile(completion,.95))
    isolated_wait,done = fcfs([0,20],[1,2])
    assert np.all(isolated_wait == 0)
    np.testing.assert_allclose(done,[1,2])

if __name__ == '__main__':
    run()
