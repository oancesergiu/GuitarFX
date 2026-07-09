import numpy as np

from engine.dsp.biquad import Biquad
from engine.dsp.cookbook import low_shelf, peaking_eq, high_shelf


class ThreeBandEQ:
    def __init__(
        self,
        sample_rate=44100,
        bass_db=0.0,
        mid_db=0.0,
        treble_db=0.0,
    ):
        self.sample_rate = sample_rate

        self.bass = Biquad(
            *low_shelf(
                fc=200,
                fs=sample_rate,
                gain_db=bass_db
            )
        )

        self.mid = Biquad(
            *peaking_eq(
                fc=1000,
                fs=sample_rate,
                gain_db=mid_db,
                q=1.0
            )
        )

        self.treble = Biquad(
            *high_shelf(
                fc=4000,
                fs=sample_rate,
                gain_db=treble_db
            )
        )

    def process(self, signal):
        signal = signal.astype(np.float32)

        signal = self.bass.process(signal)
        signal = self.mid.process(signal)
        signal = self.treble.process(signal)

        return np.clip(signal, -32768, 32767).astype(np.int16)