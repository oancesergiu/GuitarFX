import numpy as np


class FIRConvolver:
    def __init__(self, impulse_response):
        ir = np.asarray(
            impulse_response,
            dtype=np.float32,
        )

        if ir.ndim != 1:
            raise ValueError("Impulse response must be one-dimensional.")

        peak = np.max(np.abs(ir))

        if peak > 0:
            ir /= peak

        self.ir = ir

        self.state = np.zeros(
            len(ir) - 1,
            dtype=np.float32,
        )

    def reset(self):
        self.state.fill(0.0)

    def process(self, signal):
        signal = np.asarray(
            signal,
            dtype=np.float32,
        )

        output = np.convolve(
            signal,
            self.ir,
            mode="full",
        )

        overlap = len(self.state)

        if overlap:
            output[:overlap] += self.state

            self.state[:] = output[-overlap:]

        return output[: len(signal)]