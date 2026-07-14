import math

import numpy as np
from scipy.signal import lfilter

from engine.dsp.lfo import LFO


class Phaser:
    """
    CPU-efficient multi-stage phaser for normalized float32 audio.

    The LFO and filter coefficients update once per audio block.
    Each all-pass stage is processed by scipy.signal.lfilter.
    """

    def __init__(
        self,
        sample_rate=44100,
        rate_hz=0.6,
        min_frequency=250.0,
        max_frequency=2200.0,
        stages=4,
        feedback=0.25,
        mix=0.5,
        waveform="sine",
    ):
        self.sample_rate = float(sample_rate)
        self.min_frequency = float(min_frequency)
        self.max_frequency = float(max_frequency)

        self.stage_count = max(1, int(stages))

        self.feedback = float(
            np.clip(feedback, -0.85, 0.85)
        )
        self.mix = float(
            np.clip(mix, 0.0, 1.0)
        )

        self.lfo = LFO(
            sample_rate=self.sample_rate,
            rate_hz=rate_hz,
            waveform=waveform,
        )

        # Spread the all-pass stages across different frequencies.
        self.stage_offsets = np.linspace(
            0.60,
            1.40,
            self.stage_count,
            dtype=np.float64,
        )

        # One filter-state value per first-order stage.
        self.stage_states = [
            np.zeros(1, dtype=np.float64)
            for _ in range(self.stage_count)
        ]

        # Feedback is carried between blocks.
        self.feedback_state = 0.0

    def _frequency_to_coefficient(self, frequency):
        frequency = float(
            np.clip(
                frequency,
                20.0,
                self.sample_rate * 0.45,
            )
        )

        tangent = math.tan(
            math.pi * frequency / self.sample_rate
        )

        return (1.0 - tangent) / (1.0 + tangent)

    def _advance_lfo_for_block(self, block_size):
        """
        Get one modulation value and advance the LFO by one block.
        """
        lfo_value = self.lfo.next_sample()

        # One sample was already advanced by next_sample().
        phase_step = (
            2.0
            * math.pi
            * self.lfo.rate_hz
            / self.sample_rate
        )

        self.lfo.phase += phase_step * max(0, block_size - 1)
        self.lfo.phase %= 2.0 * math.pi

        return lfo_value

    def set_rate(self, rate_hz):
        self.lfo.set_rate(rate_hz)

    def set_waveform(self, waveform):
        self.lfo.set_waveform(waveform)

    def set_feedback(self, feedback):
        self.feedback = float(
            np.clip(feedback, -0.85, 0.85)
        )

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
                "Maximum phaser frequency must exceed minimum."
            )

        self.min_frequency = minimum
        self.max_frequency = maximum

    def reset(self):
        self.lfo.reset()
        self.feedback_state = 0.0

        for state in self.stage_states:
            state.fill(0.0)

    def process_inplace(self, buffer):
        signal = np.asarray(buffer, dtype=np.float32)

        if signal.size == 0:
            return

        dry = signal.copy()

        lfo_value = self._advance_lfo_for_block(
            signal.size
        )

        base_frequency = (
            self.min_frequency
            + lfo_value
            * (self.max_frequency - self.min_frequency)
        )

        # Approximate feedback between blocks without a Python
        # sample loop.
        wet = signal.astype(
            np.float64,
            copy=True,
        )

        wet[0] += (
            self.feedback
            * self.feedback_state
        )

        for index, offset in enumerate(self.stage_offsets):
            stage_frequency = (
                base_frequency * float(offset)
            )

            coefficient = self._frequency_to_coefficient(
                stage_frequency
            )

            # First-order all-pass:
            # H(z) = (-c + z^-1) / (1 - c z^-1)
            numerator = np.array(
                [-coefficient, 1.0],
                dtype=np.float64,
            )

            denominator = np.array(
                [1.0, -coefficient],
                dtype=np.float64,
            )

            wet, self.stage_states[index] = lfilter(
                numerator,
                denominator,
                wet,
                zi=self.stage_states[index],
            )

        self.feedback_state = float(wet[-1])

        np.multiply(
            dry,
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