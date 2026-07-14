import time
from collections import defaultdict


class DSPBenchmark:
    """
    Measures execution time of each effect in the DSP chain.
    """

    def __init__(self, sample_rate, block_size):
        self.sample_rate = sample_rate
        self.block_size = block_size

        self.deadline_ms = (
            block_size / sample_rate
        ) * 1000.0

        self.times = defaultdict(float)
        self.calls = defaultdict(int)

    def measure(self, name, func, *args, **kwargs):
        start = time.perf_counter()

        result = func(*args, **kwargs)

        elapsed = (
            time.perf_counter() - start
        ) * 1000.0

        self.times[name] += elapsed
        self.calls[name] += 1

        return result

    def report(self):
        print("\nDSP Benchmark")
        print("-" * 45)

        total = 0.0

        for name in sorted(self.times):
            avg = (
                self.times[name]
                / self.calls[name]
            )

            total += avg

            print(
                f"{name:<15}"
                f"{avg:7.3f} ms"
            )

        print("-" * 45)
        print(
            f"{'TOTAL':<15}"
            f"{total:7.3f} ms"
        )

        print(
            f"{'DEADLINE':<15}"
            f"{self.deadline_ms:7.3f} ms"
        )

        usage = (
            total
            / self.deadline_ms
            * 100.0
        )

        print(
            f"{'CPU':<15}"
            f"{usage:6.1f}%"
        )

        print("-" * 45)