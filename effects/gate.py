import numpy as np


class NoiseGate:
    def __init__(self, threshold=0.008):
        self.threshold = float(threshold)

    def set_threshold(self, threshold):
        self.threshold = max(0.0, float(threshold))

    def process_inplace(self, buffer):
        level = np.abs(buffer)

        quiet = level < self.threshold

        if np.any(quiet):
            buffer[quiet] *= (
                level[quiet] / self.threshold
            )

    def process(self, signal):
        output = np.asarray(
            signal,
            dtype=np.float32,
        ).copy()

        self.process_inplace(output)
        return output