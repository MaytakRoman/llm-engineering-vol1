"""Учебная последовательная очередь обменов; измерения GPU не выполняются."""
from math import ceil


def ring_seconds(byte_count, ranks, latency, bandwidth):
    assert ranks >= 2 and byte_count >= 0
    assert latency >= 0 and bandwidth > 0
    stages = 2 * (ranks - 1)
    transferred = 2 * (ranks - 1) / ranks * byte_count
    return stages * latency + transferred / bandwidth


def finish_time(ready, durations):
    assert len(ready) == len(durations)
    assert all(b >= a for a, b in zip(ready, ready[1:]))
    end = 0.0
    for release, duration in zip(ready, durations):
        assert duration >= 0
        end = max(end, release) + duration
    return end


def main():
    ranks = 8
    gradient_bytes = 2_000_000_000
    latency = 5e-6
    bandwidth = 25e9
    backward = 0.25
    for bucket_bytes in [25_000_000, 100_000_000]:
        count = ceil(gradient_bytes / bucket_bytes)
        sizes = [bucket_bytes] * (count - 1)
        sizes += [gradient_bytes - sum(sizes)]
        assert sum(sizes) == gradient_bytes
        durations = [ring_seconds(n, ranks, latency,
                                  bandwidth) for n in sizes]
        ready = [backward * (i + 1) / count
                 for i in range(count)]
        overlapped = finish_time(ready, durations)
        serial = finish_time([backward] * count, durations)
        assert backward <= overlapped <= serial
        print(count, sum(durations),
              overlapped - backward, serial - backward)
    assert abs(finish_time([0.1, 0.2], [0.04, 0.04])
               - 0.24) < 1e-12
    assert abs(finish_time([0.1, 0.1], [0.04, 0.04])
               - 0.18) < 1e-12


if __name__ == "__main__":
    main()
