import math
import numpy as np

from engine.dsp.biquad import Biquad
from engine.dsp.cookbook import peaking_eq


class AutoWah:
    def __init__(
        self,
        sample_rate=44100,
        min_frequency=400.0,
        max_frequency=2200.0,
        rate_hz=1.5,
        resonance_q=3.0,
        boost_db=12.0,
        mix=0.7,
    ):
        self.sample_rate = float(sample_rate)

        self.min_frequency = float(min_frequency)
        self.max_frequency = float(max_frequency)

        self.rate_hz = float(rate_hz)
        self.resonance_q = float(resonance_q)
        self.boost_db = float(boost_db)

        self.mix = float(np.clip(mix, 0.0, 1.0))

        self.phase = 0.0

        self.filter = Biquad()

        # Update filter every N samples instead of every sample
        self.update_interval = 32
        self.sample_counter = 0

        self._update_filter()

    def _update_filter(self):
        lfo = 0.5 * (1.0 + math.sin(self.phase))

        center_frequency = (
            self.min_frequency
            + lfo * (self.max_frequency - self.min_frequency)
        )

        coefficients = peaking_eq(
            fc=center_frequency,
            fs=self.sample_rate,
            gain_db=self.boost_db,
            q=self.resonance_q,
        )

        self.filter.set_coefficients(*coefficients)

    def process(self, signal):
        signal = signal.astype(np.float32)
        output = np.zeros_like(signal)

        for i, sample in enumerate(signal):

            if self.sample_counter % self.update_interval == 0:
                self._update_filter()

            wet = self.filter.process_sample(sample)

            output[i] = (
                sample * (1.0 - self.mix)
                + wet * self.mix
            )

            self.phase += (
                2.0 * math.pi * self.rate_hz / self.sample_rate
            )

            if self.phase >= 2.0 * math.pi:
                self.phase -= 2.0 * math.pi

            self.sample_counter += 1

        return np.clip(output, -32768, 32767).astype(np.int16)

    def set_rate(self, rate_hz):
        self.rate_hz = float(rate_hz)

    def set_mix(self, mix):
        self.mix = float(np.clip(mix, 0.0, 1.0))

    def set_frequency_range(self, minimum, maximum):
        self.min_frequency = float(minimum)
        self.max_frequency = float(maximum)

    def set_resonance(self, q):
        self.resonance_q = float(q)

    def set_boost(self, boost_db):
        self.boost_db = float(boost_db)