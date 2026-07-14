import math

import numpy as np


class EnvelopeFollower:
    """
    Peak envelope follower for normalized float32 audio.

    Fast attack follows sudden peaks.
    Slower release lets the envelope fall smoothly.
    """

    def __init__(
        self,
        sample_rate=44100,
        attack_ms=5.0,
        release_ms=100.0,
    ):
        self.sample_rate = float(sample_rate)
        self.envelope = 0.0

        self.set_attack(attack_ms)
        self.set_release(release_ms)

    def _time_to_coefficient(self, time_ms):
        time_seconds = max(
            0.0001,
            float(time_ms) / 1000.0,
        )

        return math.exp(
            -1.0
            / (time_seconds * self.sample_rate)
        )

    def set_attack(self, attack_ms):
        self.attack_ms = float(attack_ms)
        self.attack_coefficient = self._time_to_coefficient(
            self.attack_ms
        )

    def set_release(self, release_ms):
        self.release_ms = float(release_ms)
        self.release_coefficient = self._time_to_coefficient(
            self.release_ms
        )

    def reset(self):
        self.envelope = 0.0

    def process_sample(self, sample):
        level = abs(float(sample))

        if level > self.envelope:
            coefficient = self.attack_coefficient
        else:
            coefficient = self.release_coefficient

        self.envelope = (
            coefficient * self.envelope
            + (1.0 - coefficient) * level
        )

        return self.envelope

    def process(self, signal):
        signal = np.asarray(signal, dtype=np.float32)
        output = np.empty_like(signal)

        for index, sample in enumerate(signal):
            output[index] = self.process_sample(sample)

        return output