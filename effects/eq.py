import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi


class ThreeBandEQ:
    def __init__(self, sample_rate=44100, bass_gain=1.0, mid_gain=1.0, treble_gain=1.0):
        self.sample_rate = sample_rate

        self.bass_gain = bass_gain
        self.mid_gain = mid_gain
        self.treble_gain = treble_gain

        # Frequency split points
        self.bass_cutoff = 250
        self.treble_cutoff = 4000

        # Filters
        self.b_bass, self.a_bass = butter(
            2, self.bass_cutoff, btype="low", fs=sample_rate
        )

        self.b_treble, self.a_treble = butter(
            2, self.treble_cutoff, btype="high", fs=sample_rate
        )

        self.b_mid_low, self.a_mid_low = butter(
            2, self.bass_cutoff, btype="high", fs=sample_rate
        )

        self.b_mid_high, self.a_mid_high = butter(
            2, self.treble_cutoff, btype="low", fs=sample_rate
        )

        # Filter memory/state
        self.zi_bass = lfilter_zi(self.b_bass, self.a_bass) * 0
        self.zi_treble = lfilter_zi(self.b_treble, self.a_treble) * 0
        self.zi_mid_low = lfilter_zi(self.b_mid_low, self.a_mid_low) * 0
        self.zi_mid_high = lfilter_zi(self.b_mid_high, self.a_mid_high) * 0

    def process(self, signal):
        signal = signal.astype(np.float32)

        bass, self.zi_bass = lfilter(
            self.b_bass, self.a_bass, signal, zi=self.zi_bass
        )

        treble, self.zi_treble = lfilter(
            self.b_treble, self.a_treble, signal, zi=self.zi_treble
        )

        mid, self.zi_mid_low = lfilter(
            self.b_mid_low, self.a_mid_low, signal, zi=self.zi_mid_low
        )

        mid, self.zi_mid_high = lfilter(
            self.b_mid_high, self.a_mid_high, mid, zi=self.zi_mid_high
        )

        output = (
            bass * self.bass_gain
            + mid * self.mid_gain
            + treble * self.treble_gain
        )

        return np.clip(output, -32768, 32767).astype(np.int16)