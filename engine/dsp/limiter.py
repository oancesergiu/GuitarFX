import numpy as np


class Limiter:
    """
    Simple output limiter for normalized float32 audio.

    Keeps peaks below the threshold and applies a short release
    so gain changes are smoother.
    """

    def __init__(
        self,
        sample_rate=44100,
        threshold=0.92,
        release_ms=80.0,
        makeup_gain=1.0,
    ):
        self.sample_rate = float(sample_rate)
        self.threshold = float(np.clip(threshold, 0.1, 1.0))
        self.makeup_gain = float(np.clip(makeup_gain, 0.0, 2.0))

        release_seconds = max(0.001, float(release_ms) / 1000.0)

        self.release_coefficient = np.exp(
            -1.0 / (release_seconds * self.sample_rate)
        )

        self.current_gain = 1.0

    def set_threshold(self, threshold):
        self.threshold = float(np.clip(threshold, 0.1, 1.0))

    def set_makeup_gain(self, makeup_gain):
        self.makeup_gain = float(np.clip(makeup_gain, 0.0, 2.0))

    def reset(self):
        self.current_gain = 1.0

    def process(self, signal):
        signal = np.asarray(signal, dtype=np.float32)
        output = np.empty_like(signal)

        for index, sample in enumerate(signal):
            sample = float(sample) * self.makeup_gain
            level = abs(sample)

            if level > self.threshold:
                target_gain = self.threshold / level
                self.current_gain = min(
                    self.current_gain,
                    target_gain,
                )
            else:
                self.current_gain = (
                    self.release_coefficient * self.current_gain
                    + (1.0 - self.release_coefficient)
                )

            output[index] = sample * self.current_gain

        return np.clip(
            output,
            -1.0,
            1.0,
        ).astype(np.float32)