import numpy as np


class Delay:
    def __init__(self, sample_rate=44100, delay_ms=350, feedback=0.35, mix=0.35):
        self.sample_rate = sample_rate
        self.delay_samples = int(sample_rate * delay_ms / 1000)
        self.feedback = feedback
        self.mix = mix

        self.buffer = np.zeros(self.delay_samples, dtype=np.float32)
        self.index = 0

    def process(self, signal):
        signal = signal.astype(np.float32)
        output = np.zeros_like(signal)

        for i in range(len(signal)):
            delayed = self.buffer[self.index]

            output[i] = signal[i] * (1.0 - self.mix) + delayed * self.mix

            self.buffer[self.index] = signal[i] + delayed * self.feedback

            self.index += 1
            if self.index >= self.delay_samples:
                self.index = 0

        return np.clip(output, -32768, 32767).astype(np.int16)