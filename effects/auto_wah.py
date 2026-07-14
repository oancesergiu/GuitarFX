import math

import numpy as np
from scipy.signal import lfilter

from engine.dsp.lfo import LFO
from engine.dsp.cookbook import band_pass


class AutoWah:
    """
    CPU-efficient auto-wah for normalized float32 audio.

    The LFO and band-pass coefficients update once per audio block.
    The complete block is processed by scipy.signal.lfilter.
    """

    def __init__(
        self,
        sample_rate=44100,
        min_frequency=350.0,
        max_frequency=2200.0,
        rate_hz=1.2,
        resonance_q=4.0,
        mix=0.75,
        waveform="sine",
    ):
        self.sample_rate = float(sample_rate)
        self.min_frequency = float(min_frequency)
        self.max_frequency = float(max_frequency)
        self.resonance_q = float(resonance_q)

        self.mix = float(
            np.clip(mix, 0.0, 1.0)
        )

        self.lfo = LFO(
            sample_rate=self.sample_rate,
            rate_hz=rate_hz,
            waveform=waveform,
        )

        # Biquad state: two values for a second-order filter.
        self.filter_state = np.zeros(
            2,
            dtype=np.float64,
        )

    def _advance_lfo_for_block(self, block_size):
        lfo_value = self.lfo.next_sample()

        phase_step = (
            2.0
            * math.pi
            * self.lfo.rate_hz
            / self.sample_rate
        )

        self.lfo.phase += phase_step * max(
            0,
            block_size - 1,
        )

        self.lfo.phase %= 2.0 * math.pi

        return lfo_value

    def set_rate(self, rate_hz):
        self.lfo.set_rate(rate_hz)

    def set_waveform(self, waveform):
        self.lfo.set_waveform(waveform)

    def set_mix(self, mix):
        self.mix = float(
            np.clip(mix, 0.0, 1.0)
        )

    def set_frequency_range(self, minimum, maximum):
        minimum = max(20.0, float(minimum))
        maximum = min(
            self.sample_rate * 0.45,
            float(maximum),
        )

        if maximum <= minimum:
            raise ValueError(
                "Maximum wah frequency must exceed minimum."
            )

        self.min_frequency = minimum
        self.max_frequency = maximum

    def set_resonance(self, q):
        self.resonance_q = float(
            np.clip(q, 0.2, 20.0)
        )

    def reset(self):
        self.lfo.reset()
        self.filter_state.fill(0.0)

    def process_inplace(self, buffer):
        signal = np.asarray(
            buffer,
            dtype=np.float32,
        )

        if signal.size == 0:
            return

        lfo_value = self._advance_lfo_for_block(
            signal.size
        )

        # Logarithmic sweep sounds more natural than a linear one.
        frequency_ratio = (
            self.max_frequency
            / self.min_frequency
        )

        center_frequency = (
            self.min_frequency
            * (frequency_ratio ** lfo_value)
        )

        b0, b1, b2, a1, a2 = band_pass(
            fc=center_frequency,
            fs=self.sample_rate,
            q=self.resonance_q,
        )

        numerator = np.array(
            [b0, b1, b2],
            dtype=np.float64,
        )

        denominator = np.array(
            [1.0, a1, a2],
            dtype=np.float64,
        )

        wet, self.filter_state = lfilter(
            numerator,
            denominator,
            signal,
            zi=self.filter_state,
        )

        np.multiply(
            signal,
            1.0 - self.mix,
            out=buffer,
        )

        buffer += wet.astype(
            np.float32,
            copy=False,
        ) * self.mix

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