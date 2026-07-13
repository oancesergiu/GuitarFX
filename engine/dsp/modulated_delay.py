import numpy as np

from engine.dsp.delay_line import DelayLine
from engine.dsp.lfo import LFO


class ModulatedDelay:
    """
    Reusable fractional modulated delay.

    Useful for chorus, flanger, and vibrato.
    """

    def __init__(
        self,
        sample_rate=44100,
        base_delay_ms=15.0,
        depth_ms=3.0,
        rate_hz=0.8,
        waveform="sine",
        feedback=0.0,
        max_delay_ms=50.0,
    ):
        self.sample_rate = float(sample_rate)

        self.base_delay_ms = float(base_delay_ms)
        self.depth_ms = float(depth_ms)
        self.feedback = float(
            np.clip(feedback, -0.95, 0.95)
        )

        max_delay_samples = int(
            self.sample_rate
            * max_delay_ms
            / 1000.0
        )

        self.delay_line = DelayLine(
            max_delay_samples=max_delay_samples
        )

        self.lfo = LFO(
            sample_rate=self.sample_rate,
            rate_hz=rate_hz,
            waveform=waveform,
        )

    def set_rate(self, rate_hz):
        self.lfo.set_rate(rate_hz)

    def set_waveform(self, waveform):
        self.lfo.set_waveform(waveform)

    def set_base_delay_ms(self, delay_ms):
        self.base_delay_ms = max(
            0.0,
            float(delay_ms),
        )

    def set_depth_ms(self, depth_ms):
        self.depth_ms = max(
            0.0,
            float(depth_ms),
        )

    def set_feedback(self, feedback):
        self.feedback = float(
            np.clip(feedback, -0.95, 0.95)
        )

    def reset(self):
        self.delay_line.reset()
        self.lfo.reset()

    def process_sample(self, sample):
        lfo_value = self.lfo.next_sample()

        modulation = (
            2.0 * lfo_value - 1.0
        )

        delay_ms = (
            self.base_delay_ms
            + modulation * self.depth_ms
        )

        delay_ms = max(0.0, delay_ms)

        delay_samples = (
            self.sample_rate
            * delay_ms
            / 1000.0
        )

        delayed = self.delay_line.read(
            delay_samples
        )

        self.delay_line.write(
            float(sample)
            + delayed * self.feedback
        )

        return delayed

    def process(self, signal):
        signal = np.asarray(
            signal,
            dtype=np.float32,
        )

        output = np.empty_like(signal)

        for index, sample in enumerate(signal):
            output[index] = self.process_sample(sample)

        return np.clip(
            output,
            -1.0,
            1.0,
        ).astype(np.float32)