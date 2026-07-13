import math

import numpy as np

from engine.dsp.allpass import AllPassStage
from engine.dsp.lfo import LFO


class Phaser:
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

        self.feedback = float(
            np.clip(feedback, -0.95, 0.95)
        )
        self.mix = float(
            np.clip(mix, 0.0, 1.0)
        )

        self.lfo = LFO(
            sample_rate=sample_rate,
            rate_hz=rate_hz,
            waveform=waveform,
        )

        self.stages = [
            AllPassStage()
            for _ in range(max(1, int(stages)))
        ]

        self.stage_offsets = np.linspace(
            0.65,
            1.35,
            len(self.stages),
            dtype=np.float32,
        )

        self.feedback_sample = 0.0
        self.update_interval = 32
        self.sample_counter = 0

        self._update_coefficients(0.5)

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

    def _update_coefficients(self, lfo_value):
        base_frequency = (
            self.min_frequency
            + lfo_value
            * (self.max_frequency - self.min_frequency)
        )

        for stage, offset in zip(
            self.stages,
            self.stage_offsets,
        ):
            stage_frequency = base_frequency * float(offset)

            coefficient = self._frequency_to_coefficient(
                stage_frequency
            )

            stage.set_coefficient(coefficient)

    def set_rate(self, rate_hz):
        self.lfo.set_rate(rate_hz)

    def set_waveform(self, waveform):
        self.lfo.set_waveform(waveform)

    def set_feedback(self, feedback):
        self.feedback = float(
            np.clip(feedback, -0.95, 0.95)
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
        self.feedback_sample = 0.0
        self.sample_counter = 0
        self.lfo.reset()

        for stage in self.stages:
            stage.reset()

    def process(self, signal):
        signal = np.asarray(
            signal,
            dtype=np.float32,
        )

        output = np.empty_like(signal)
        dry_mix = 1.0 - self.mix

        for index, sample in enumerate(signal):
            lfo_value = self.lfo.next_sample()

            if (
                self.sample_counter
                % self.update_interval
                == 0
            ):
                self._update_coefficients(lfo_value)

            wet = (
                float(sample)
                + self.feedback
                * self.feedback_sample
            )

            for stage in self.stages:
                wet = stage.process_sample(wet)

            self.feedback_sample = wet

            output[index] = (
                float(sample) * dry_mix
                + wet * self.mix
            )

            self.sample_counter += 1

        return np.clip(
            output,
            -1.0,
            1.0,
        ).astype(np.float32)