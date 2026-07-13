import numpy as np


class AllPassStage:
    """
    First-order all-pass filter stage for normalized float32 audio.
    """

    def __init__(self, coefficient=0.0):
        self.coefficient = 0.0
        self.x1 = 0.0
        self.y1 = 0.0

        self.set_coefficient(coefficient)

    def set_coefficient(self, coefficient):
        self.coefficient = float(
            np.clip(coefficient, -0.99, 0.99)
        )

    def reset(self):
        self.x1 = 0.0
        self.y1 = 0.0

    def process_sample(self, sample):
        x = float(sample)

        y = (
            -self.coefficient * x
            + self.x1
            + self.coefficient * self.y1
        )

        self.x1 = x
        self.y1 = y

        return y

    def process(self, signal):
        signal = np.asarray(signal, dtype=np.float32)
        output = np.empty_like(signal)

        for index, sample in enumerate(signal):
            output[index] = self.process_sample(sample)

        return output