import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi


class Overdrive:
    def __init__(
        self,
        gain=4.0,
        drive=15000.0,
        level=22000.0,
        tone=0.55,
        sample_rate=44100,
    ):
        self.sample_rate = int(sample_rate)

        self.gain = float(gain)
        self.drive = float(drive)
        self.level = float(level)
        self.tone = float(np.clip(tone, 0.0, 1.0))

        # Tighten the low end before clipping.
        self.hp_b, self.hp_a = butter(
            2,
            120.0,
            btype="highpass",
            fs=self.sample_rate,
        )
        self.hp_state = lfilter_zi(self.hp_b, self.hp_a) * 0.0

        # Add a small mid emphasis before clipping.
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
        self.drive = float(np.clip(drive, 1000.0, 32000.0))

    def set_level(self, level):
        self.level = float(np.clip(level, 0.0, 32767.0))

    def set_tone(self, tone):
        self.tone = float(np.clip(tone, 0.0, 1.0))
        self._update_tone_filter()

    def process(self, signal):
        signal = np.asarray(signal, dtype=np.float32)

        # Normalize int16-style samples.
        x = signal / 32768.0

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

        # Blend a small mid boost into the dry signal.
        x = x + 0.22 * mids

        # Input gain.
        x *= self.gain

        threshold = max(self.drive / 32768.0, 0.03)
        normalized = x / threshold

        # First clipping stage: asymmetric soft clipping.
        stage1 = np.where(
            normalized >= 0.0,
            np.tanh(normalized),
            0.82 * np.tanh(normalized * 1.2),
        )

        # Second clipping stage: gentler saturation.
        stage2 = np.tanh(stage1 * 1.35)

        # Keep some attack and string definition.
        shaped = 0.90 * stage2 + 0.10 * np.clip(x, -1.0, 1.0)

        # Tone shaping after clipping.
        shaped, self.lp_state = lfilter(
            self.lp_b,
            self.lp_a,
            shaped,
            zi=self.lp_state,
        )

        output_gain = self.level / 32767.0
        shaped *= output_gain

        return np.clip(
            shaped * 32767.0,
            -32768,
            32767,
        ).astype(np.int16)