import math


class LFO:
    """
    Reusable low-frequency oscillator.

    Output range:
        0.0 to 1.0
    """

    def __init__(
        self,
        sample_rate=44100,
        rate_hz=1.0,
        waveform="sine",
        phase=0.0,
    ):
        self.sample_rate = float(sample_rate)
        self.rate_hz = float(rate_hz)
        self.waveform = waveform
        self.phase = float(phase)

    def set_rate(self, rate_hz):
        self.rate_hz = max(0.0, float(rate_hz))

    def set_waveform(self, waveform):
        if waveform not in ("sine", "triangle"):
            raise ValueError(
                "LFO waveform must be 'sine' or 'triangle'."
            )

        self.waveform = waveform

    def reset(self, phase=0.0):
        self.phase = float(phase)

    def next_sample(self):
        if self.waveform == "sine":
            value = 0.5 * (
                1.0 + math.sin(self.phase)
            )

        else:
            normalized_phase = (
                self.phase / (2.0 * math.pi)
            )

            value = 1.0 - abs(
                4.0 * normalized_phase - 2.0
            )

        self.phase += (
            2.0
            * math.pi
            * self.rate_hz
            / self.sample_rate
        )

        if self.phase >= 2.0 * math.pi:
            self.phase -= 2.0 * math.pi

        return value