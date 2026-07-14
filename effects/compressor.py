import numpy as np

from engine.dsp.envelope import EnvelopeFollower


class Compressor:
    """
    Feed-forward compressor for normalized float32 audio.
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
    ):
        self.sample_rate = int(sample_rate)

        self.threshold_db = float(threshold_db)
        self.ratio = max(1.0, float(ratio))
        self.makeup_db = float(makeup_db)
        self.mix = float(np.clip(mix, 0.0, 1.0))

        self.envelope = EnvelopeFollower(
            sample_rate=sample_rate,
            attack_ms=attack_ms,
            release_ms=release_ms,
        )

    @staticmethod
    def _amplitude_to_db(amplitude):
        return 20.0 * np.log10(max(float(amplitude), 1e-8))

    @staticmethod
    def _db_to_amplitude(db):
        return 10.0 ** (float(db) / 20.0)

    def set_threshold(self, threshold_db):
        self.threshold_db = float(threshold_db)

    def set_ratio(self, ratio):
        self.ratio = max(1.0, float(ratio))

    def set_attack(self, attack_ms):
        self.envelope.set_attack(attack_ms)

    def set_release(self, release_ms):
        self.envelope.set_release(release_ms)

    def set_makeup(self, makeup_db):
        self.makeup_db = float(makeup_db)

    def set_mix(self, mix):
        self.mix = float(np.clip(mix, 0.0, 1.0))

    def reset(self):
        self.envelope.reset()

    def process(self, signal):
        signal = np.asarray(signal, dtype=np.float32)
        output = np.empty_like(signal)

        makeup_gain = self._db_to_amplitude(self.makeup_db)
        dry_mix = 1.0 - self.mix

        for index, sample in enumerate(signal):
            envelope = self.envelope.process_sample(sample)
            envelope_db = self._amplitude_to_db(envelope)

            gain_reduction_db = 0.0

            if envelope_db > self.threshold_db:
                compressed_db = (
                    self.threshold_db
                    + (envelope_db - self.threshold_db) / self.ratio
                )

                gain_reduction_db = compressed_db - envelope_db

            gain = (
                self._db_to_amplitude(gain_reduction_db)
                * makeup_gain
            )

            wet = float(sample) * gain

            output[index] = (
                float(sample) * dry_mix
                + wet * self.mix
            )

        return np.clip(
            output,
            -1.0,
            1.0,
        ).astype(np.float32)