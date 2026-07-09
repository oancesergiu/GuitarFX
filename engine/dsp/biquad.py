import numpy as np


class Biquad:
    def __init__(self, b0=1.0, b1=0.0, b2=0.0, a1=0.0, a2=0.0):
        self.set_coefficients(b0, b1, b2, a1, a2)
        self.reset()

    def set_coefficients(self, b0, b1, b2, a1, a2):
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        self.a1 = a1
        self.a2 = a2

    def reset(self):
        self.x1 = 0.0
        self.x2 = 0.0
        self.y1 = 0.0
        self.y2 = 0.0

    def process(self, signal):
        signal = signal.astype(np.float32)
        output = np.zeros_like(signal)

        for i, x in enumerate(signal):
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

            output[i] = y

        return output