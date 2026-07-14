import numpy as np
from scipy.signal import lfilter


class Biquad:
    """
    Stateful second-order IIR filter for normalized float32 audio.

    Block processing uses SciPy's compiled lfilter implementation.
    Sample processing uses the equivalent transposed Direct Form II.
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

        self.b = np.array(
            [self.b0, self.b1, self.b2],
            dtype=np.float64,
        )

        self.a = np.array(
            [1.0, self.a1, self.a2],
            dtype=np.float64,
        )

    def reset(self):
        # Transposed Direct Form II filter state.
        self.state = np.zeros(2, dtype=np.float64)

    def process_sample(self, sample):
        x = float(sample)

        y = self.b0 * x + self.state[0]

        new_state_0 = (
            self.b1 * x
            - self.a1 * y
            + self.state[1]
        )

        new_state_1 = (
            self.b2 * x
            - self.a2 * y
        )

        self.state[0] = new_state_0
        self.state[1] = new_state_1

        return y

    def process(self, signal):
        signal = np.asarray(signal, dtype=np.float32)

        output, self.state = lfilter(
            self.b,
            self.a,
            signal,
            zi=self.state,
        )

        return output.astype(np.float32, copy=False)