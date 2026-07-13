import numpy as np

from engine.dsp.delay_line import DelayLine


class Delay:
    def __init__(
        self,
        sample_rate=44100,
        delay_ms=350.0,
        feedback=0.35,
        mix=0.30,
        max_delay_ms=2000.0,
    ):
        self.sample_rate = int(sample_rate)

        max_delay_samples = int(
            sample_rate * max_delay_ms / 1000.0
        )

        self.delay_line = DelayLine(max_delay_samples)

        self.feedback = 0.0
        self.mix = 0.0

        self.set_feedback(feedback)
        self.set_mix(mix)
        self.set_delay_ms(delay_ms)

    def reset(self):
        self.delay_line.reset()

    def set_delay_ms(self, delay_ms):
        self.delay_ms = float(delay_ms)

        self.delay_samples = max(
            1,
            int(
                self.sample_rate
                * self.delay_ms
                / 1000.0
            ),
        )

    def set_feedback(self, feedback):
        self.feedback = float(
            np.clip(feedback, 0.0, 0.95)
        )

    def set_mix(self, mix):
        self.mix = float(
            np.clip(mix, 0.0, 1.0)
        )

    def process(self, signal):
        signal = np.asarray(signal, dtype=np.float32)
        output = np.empty_like(signal)

        dry = 1.0 - self.mix

        for i, sample in enumerate(signal):
            delayed = self.delay_line.read(
                self.delay_samples
            )

            output[i] = (
                sample * dry
                + delayed * self.mix
            )

            self.delay_line.write(
                sample
                + delayed * self.feedback
            )

        return np.clip(
            output,
            -1.0,
            1.0,
        ).astype(np.float32)