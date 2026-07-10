import numpy as np


class Delay:
    def __init__(
        self,
        sample_rate=44100,
        delay_ms=350,
        feedback=0.35,
        mix=0.35,
    ):
        self.sample_rate = sample_rate
        self.feedback = float(feedback)
        self.mix = float(mix)

        self.set_delay_ms(delay_ms)

    def set_delay_ms(self, delay_ms):
        self.delay_ms = float(delay_ms)
        self.delay_samples = max(
            1,
            int(self.sample_rate * self.delay_ms / 1000.0)
        )

        # Recreate the buffer when delay time changes
        self.buffer = np.zeros(self.delay_samples, dtype=np.float32)
        self.index = 0

    def set_feedback(self, feedback):
        self.feedback = float(np.clip(feedback, 0.0, 0.95))

    def set_mix(self, mix):
        self.mix = float(np.clip(mix, 0.0, 1.0))

    def process(self, signal):
        signal = signal.astype(np.float32)
        output = np.zeros_like(signal)

        for i in range(len(signal)):
            delayed = self.buffer[self.index]

            dry = signal[i] * (1.0 - self.mix)
            wet = delayed * self.mix
            output[i] = dry + wet

            self.buffer[self.index] = (
                signal[i] + delayed * self.feedback
            )

            self.index += 1

            if self.index >= self.delay_samples:
                self.index = 0

        return np.clip(output, -32768, 32767).astype(np.int16)