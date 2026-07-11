import numpy as np


class Biquad:
    """
    Stateful second-order IIR filter operating entirely on float32 audio.
    """

    def __init__(
        self,
        b0=1.0,
        b1=0.0,
        b2=0.0,
        a1=0.0,
        a2=0.0,
    ):
        self.set_coefficients(b0, b1, b2, a1, a2)
        self.reset()

    def set_coefficients(self, b0, b1, b2, a1, a2):
        self.b0 = float(b0)
        self.b1 = float(b1)
        self.b2 = float(b2)
        self.a1 = float(a1)
        self.a2 = float(a2)

    def reset(self):
        self.x1 = 0.0
        self.x2 = 0.0
        self.y1 = 0.0
        self.y2 = 0.0

    def process_sample(self, sample):
        x = float(sample)

        y = (
            self.b0 * x
            + self.b1 * self.x1
            + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2
        )

        self.x2 = self.x1
        self.x1 = x

        self.y2 = self.y1
        self.y1 = y

        return y

    def process(self, signal):
        signal = np.asarray(signal, dtype=np.float32)
        output = np.empty_like(signal)

        for index, sample in enumerate(signal):
            output[index] = self.process_sample(sample)

        return output