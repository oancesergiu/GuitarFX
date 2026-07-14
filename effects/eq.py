import numpy as np

from engine.dsp.biquad import Biquad
from engine.dsp.cookbook import high_shelf, low_shelf, peaking_eq


class ThreeBandEQ:
    def __init__(
        self,
        sample_rate=44100,
        bass_db=0.0,
        mid_db=0.0,
        treble_db=0.0,
        bass_frequency=200.0,
        mid_frequency=1000.0,
        treble_frequency=4000.0,
        mid_q=1.0,
    ):
        self.sample_rate = float(sample_rate)

        self.bass_frequency = float(bass_frequency)
        self.mid_frequency = float(mid_frequency)
        self.treble_frequency = float(treble_frequency)
        self.mid_q = float(mid_q)

        self.bass_db = float(bass_db)
        self.mid_db = float(mid_db)
        self.treble_db = float(treble_db)

        self.bass = Biquad()
        self.mid = Biquad()
        self.treble = Biquad()

        self._update_filters()

    def _update_filters(self):
        self.bass.set_coefficients(
            *low_shelf(
                fc=self.bass_frequency,
                fs=self.sample_rate,
                gain_db=self.bass_db,
            )
        )

        self.mid.set_coefficients(
            *peaking_eq(
                fc=self.mid_frequency,
                fs=self.sample_rate,
                gain_db=self.mid_db,
                q=self.mid_q,
            )
        )

        self.treble.set_coefficients(
            *high_shelf(
                fc=self.treble_frequency,
                fs=self.sample_rate,
                gain_db=self.treble_db,
            )
        )

    def set_bass(self, gain_db):
        self.bass_db = float(np.clip(gain_db, -12.0, 12.0))
        self._update_filters()

    def set_mid(self, gain_db):
        self.mid_db = float(np.clip(gain_db, -12.0, 12.0))
        self._update_filters()

    def set_treble(self, gain_db):
        self.treble_db = float(np.clip(gain_db, -12.0, 12.0))
        self._update_filters()

    def process_inplace(self, buffer):
        processed = self.bass.process(buffer)
        processed = self.mid.process(processed)
        processed = self.treble.process(processed)

        np.copyto(
            buffer,
            np.clip(processed, -1.0, 1.0),
        )


    def process(self, signal):
        output = np.asarray(
            signal,
            dtype=np.float32,
        ).copy()

        self.process_inplace(output)

        return output