import numpy as np


class Delay:
    """
    Stateful feedback delay using normalized float32 audio.
    """

    def __init__(
        self,
        sample_rate=44100,
        delay_ms=350.0,
        feedback=0.35,
        mix=0.30,
    ):
        self.sample_rate = int(sample_rate)

        self.feedback = 0.0
        self.mix = 0.0

        self.set_feedback(feedback)
        self.set_mix(mix)
        self.set_delay_ms(delay_ms)

    def set_delay_ms(self, delay_ms):
        self.delay_ms = max(1.0, float(delay_ms))

        self.delay_samples = max(
            1,
            int(
                self.sample_rate
                * self.delay_ms
                / 1000.0
            ),
        )

        self.buffer = np.zeros(
            self.delay_samples,
            dtype=np.float32,
        )

        self.index = 0

    def set_feedback(self, feedback):
        self.feedback = float(
            np.clip(feedback, 0.0, 0.95)
        )

    def set_mix(self, mix):
        self.mix = float(
            np.clip(mix, 0.0, 1.0)
        )

    def reset(self):
        self.buffer.fill(0.0)
        self.index = 0

    def process(self, signal):
        signal = np.asarray(signal, dtype=np.float32)
        output = np.empty_like(signal)

        dry_mix = 1.0 - self.mix

        for sample_index, sample in enumerate(signal):
            delayed_sample = self.buffer[self.index]

            output[sample_index] = (
                sample * dry_mix
                + delayed_sample * self.mix
            )

            self.buffer[self.index] = (
                sample
                + delayed_sample * self.feedback
            )

            self.index += 1

            if self.index >= self.delay_samples:
                self.index = 0

        return np.clip(
            output,
            -1.0,
            1.0,
        ).astype(np.float32)