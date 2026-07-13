import math

import numpy as np


class DelayLine:
    """
    Circular delay line supporting fractional delay
    using linear interpolation.
    """

    def __init__(self, max_delay_samples):
        self.size = int(max_delay_samples)

        if self.size <= 1:
            raise ValueError(
                "Delay line must contain at least two samples."
            )

        self.buffer = np.zeros(
            self.size,
            dtype=np.float32,
        )

        self.write_index = 0

    def reset(self):
        self.buffer.fill(0.0)
        self.write_index = 0

    def write(self, sample):
        self.buffer[self.write_index] = float(sample)

        self.write_index += 1

        if self.write_index >= self.size:
            self.write_index = 0

    def read(self, delay_samples):
        """
        Read any delay, including fractional delays.

        Example:
            125.35 samples
        """

        delay_samples = float(delay_samples)

        if delay_samples < 0:
            delay_samples = 0.0

        if delay_samples > self.size - 2:
            delay_samples = self.size - 2

        read_position = (
            self.write_index
            - delay_samples
            - 1.0
        )

        while read_position < 0:
            read_position += self.size

        index0 = int(math.floor(read_position))
        index1 = (index0 + 1) % self.size

        fraction = read_position - index0

        sample0 = self.buffer[index0]
        sample1 = self.buffer[index1]

        return (
            sample0 * (1.0 - fraction)
            + sample1 * fraction
        )