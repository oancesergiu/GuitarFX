import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi


class Overdrive:
    def __init__(
        self,
        gain=4.0,
        drive=0.45,
        level=0.75,
        tone=0.55,
        sample_rate=44100,
    ):
        self.sample_rate = int(sample_rate)

        self.gain = float(gain)
        self.drive = float(drive)
        self.level = float(level)
        self.tone = float(np.clip(tone, 0.0, 1.0))

        # Tighten low frequencies before clipping.
        self.hp_b, self.hp_a = butter(
            2,
            120.0,
            btype="highpass",
            fs=self.sample_rate,
        )
        self.hp_state = lfilter_zi(self.hp_b, self.hp_a) * 0.0

        # Mid-focused path before clipping.
        self.mid_b, self.mid_a = butter(
            2,
            [500.0, 1800.0],
            btype="bandpass",
            fs=self.sample_rate,
        )
        self.mid_state = lfilter_zi(self.mid_b, self.mid_a) * 0.0

        self._update_tone_filter()

    def _update_tone_filter(self):
        cutoff = 2500.0 + self.tone * 5000.0

        self.lp_b, self.lp_a = butter(
            2,
            cutoff,
            btype="lowpass",
            fs=self.sample_rate,
        )
        self.lp_state = lfilter_zi(self.lp_b, self.lp_a) * 0.0

    def set_gain(self, gain):
        self.gain = float(np.clip(gain, 0.1, 20.0))

    def set_drive(self, drive):
        self.drive = float(np.clip(drive, 0.03, 1.0))

    def set_level(self, level):
        self.level = float(np.clip(level, 0.0, 1.0))

    def set_tone(self, tone):
        self.tone = float(np.clip(tone, 0.0, 1.0))
        self._update_tone_filter()

    def process(self, signal):
        x = np.asarray(signal, dtype=np.float32)

        # Tight high-pass.
        x, self.hp_state = lfilter(
            self.hp_b,
            self.hp_a,
            x,
            zi=self.hp_state,
        )

        # Mid-focused parallel path.
        mids, self.mid_state = lfilter(
            self.mid_b,
            self.mid_a,
            x,
            zi=self.mid_state,
        )

        x = x + 0.22 * mids
        x *= self.gain

        normalized = x / self.drive

        stage1 = np.where(
            normalized >= 0.0,
            np.tanh(normalized),
            0.82 * np.tanh(normalized * 1.2),
        )

        stage2 = np.tanh(stage1 * 1.35)

        shaped = (
            0.90 * stage2
            + 0.10 * np.clip(x, -1.0, 1.0)
        )

        shaped, self.lp_state = lfilter(
            self.lp_b,
            self.lp_a,
            shaped,
            zi=self.lp_state,
        )

        shaped *= self.level

        return np.clip(shaped, -1.0, 1.0).astype(np.float32)