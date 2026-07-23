import math

import numpy as np


class Compressor:
    """
    CPU-efficient block compressor for normalized float32 audio.

    Uses RMS level detection and a smoothly ramped gain across each
    audio block, avoiding Python sample-by-sample loops.
    """

    def __init__(
        self,
        sample_rate=44100,
        threshold_db=-18.0,
        ratio=4.0,
        attack_ms=5.0,
        release_ms=120.0,
        makeup_db=3.0,
        mix=1.0,
        block_size=256,
    ):
        self.sample_rate = float(sample_rate)
        self.block_size = max(1, int(block_size))

        self.threshold_db = float(threshold_db)
        self.ratio = max(1.0, float(ratio))
        self.makeup_db = float(makeup_db)
        self.mix = float(np.clip(mix, 0.0, 1.0))

        self.current_gain = 1.0

        self.set_attack(attack_ms)
        self.set_release(release_ms)

    @staticmethod
    def _db_to_amplitude(db):
        return 10.0 ** (float(db) / 20.0)

    @staticmethod
    def _amplitude_to_db(amplitude):
        return 20.0 * math.log10(
            max(float(amplitude), 1e-9)
        )

    def _time_to_block_coefficient(self, time_ms):
        time_seconds = max(
            0.0001,
            float(time_ms) / 1000.0,
        )

        block_seconds = (
            self.block_size / self.sample_rate
        )

        return math.exp(
            -block_seconds / time_seconds
        )

    def set_threshold(self, threshold_db):
        self.threshold_db = float(threshold_db)

    def set_ratio(self, ratio):
        self.ratio = max(1.0, float(ratio))

    def set_attack(self, attack_ms):
        self.attack_ms = max(
            0.1,
            float(attack_ms),
        )

        self.attack_coefficient = (
            self._time_to_block_coefficient(
                self.attack_ms
            )
        )

    def set_release(self, release_ms):
        self.release_ms = max(
            1.0,
            float(release_ms),
        )

        self.release_coefficient = (
            self._time_to_block_coefficient(
                self.release_ms
            )
        )

    def set_makeup(self, makeup_db):
        self.makeup_db = float(makeup_db)

    def set_mix(self, mix):
        self.mix = float(
            np.clip(mix, 0.0, 1.0)
        )

    def reset(self):
        self.current_gain = 1.0

    def process_inplace(self, buffer):
        signal = np.asarray(
            buffer,
            dtype=np.float32,
        )

        if signal.size == 0:
            return

        # RMS level of the current audio block.
        rms = float(
            np.sqrt(
                np.mean(signal * signal)
                + 1e-12
            )
        )

        level_db = self._amplitude_to_db(rms)

        gain_reduction_db = 0.0

        if level_db > self.threshold_db:
            compressed_db = (
                self.threshold_db
                + (
                    level_db
                    - self.threshold_db
                ) / self.ratio
            )

            gain_reduction_db = (
                compressed_db - level_db
            )

        target_gain = (
            self._db_to_amplitude(
                gain_reduction_db
                + self.makeup_db
            )
        )

        # Faster response when gain must decrease,
        # slower recovery when gain returns upward.
        if target_gain < self.current_gain:
            coefficient = self.attack_coefficient
        else:
            coefficient = self.release_coefficient

        smoothed_gain = (
            coefficient * self.current_gain
            + (1.0 - coefficient) * target_gain
        )

        # Ramp gain smoothly across the block to avoid clicks.
        gain_curve = np.linspace(
            self.current_gain,
            smoothed_gain,
            signal.size,
            dtype=np.float32,
        )

        wet = signal * gain_curve

        if self.mix >= 1.0:
            np.copyto(buffer, wet)
        elif self.mix <= 0.0:
            pass
        else:
            buffer *= 1.0 - self.mix
            buffer += wet * self.mix

        self.current_gain = smoothed_gain

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