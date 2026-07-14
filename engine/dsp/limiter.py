import math

import numpy as np


class Limiter:
    """
    Fast block-based limiter for normalized float32 audio.

    Supports both:

        process_inplace(buffer)

    and

        process(signal)

    so it works with the new EffectRack while remaining
    backwards compatible.
    """

    def __init__(
        self,
        sample_rate=44100,
        threshold=0.92,
        release_ms=80.0,
        makeup_gain=1.0,
        block_size=512,
    ):
        self.sample_rate = float(sample_rate)
        self.block_size = max(1, int(block_size))

        self.threshold = float(
            np.clip(threshold, 0.1, 1.0)
        )

        self.makeup_gain = float(
            np.clip(makeup_gain, 0.0, 2.0)
        )

        self.current_gain = 1.0

        self.set_release(release_ms)

    def set_threshold(self, threshold):
        self.threshold = float(
            np.clip(threshold, 0.1, 1.0)
        )

    def set_makeup_gain(self, makeup_gain):
        self.makeup_gain = float(
            np.clip(makeup_gain, 0.0, 2.0)
        )

    def set_release(self, release_ms):
        release_seconds = max(
            0.001,
            float(release_ms) / 1000.0,
        )

        block_duration = (
            self.block_size / self.sample_rate
        )

        self.release_coefficient = math.exp(
            -block_duration / release_seconds
        )

    def reset(self):
        self.current_gain = 1.0

    def process_inplace(self, buffer):
        if buffer.size == 0:
            return

        peak = (
            float(np.max(np.abs(buffer)))
            * self.makeup_gain
        )

        if peak > self.threshold:
            target_gain = self.threshold / peak

            self.current_gain = min(
                self.current_gain,
                target_gain,
            )
        else:
            self.current_gain = (
                self.release_coefficient
                * self.current_gain
                + (1.0 - self.release_coefficient)
            )

        buffer *= (
            self.makeup_gain
            * self.current_gain
        )

        np.clip(
            buffer,
            -1.0,
            1.0,
            out=buffer,
        )

    def process(self, signal):
        output = np.asarray(
            signal,
            dtype=np.float32,
        ).copy()

        self.process_inplace(output)

        return output