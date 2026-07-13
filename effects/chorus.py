import numpy as np

from engine.dsp.modulated_delay import ModulatedDelay


class Chorus:
    def __init__(
        self,
        sample_rate=44100,
        rate_hz=0.8,
        base_delay_ms=18.0,
        depth_ms=4.0,
        mix=0.35,
    ):
        self.mix = float(np.clip(mix, 0.0, 1.0))

        self.modulated_delay = ModulatedDelay(
            sample_rate=sample_rate,
            base_delay_ms=base_delay_ms,
            depth_ms=depth_ms,
            rate_hz=rate_hz,
            feedback=0.0,
            max_delay_ms=40.0,
        )

    def set_rate(self, rate_hz):
        self.modulated_delay.set_rate(rate_hz)

    def set_depth(self, depth_ms):
        self.modulated_delay.set_depth_ms(depth_ms)

    def set_mix(self, mix):
        self.mix = float(np.clip(mix, 0.0, 1.0))

    def reset(self):
        self.modulated_delay.reset()

    def process(self, signal):
        signal = np.asarray(signal, dtype=np.float32)

        wet = self.modulated_delay.process(signal)

        output = (
            signal * (1.0 - self.mix)
            + wet * self.mix
        )

# Leave headroom to prevent chorus peaks from clipping
        output *= 0.8

        return np.clip(
            output,
            -1.0,
            1.0,
        ).astype(np.float32)