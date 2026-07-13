import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi


class Preamp:
    """
    Simple multi-stage guitar preamp model for normalized float32 audio.

    Signal chain:
        input high-pass
        -> first gain stage
        -> inter-stage tone shaping
        -> second gain stage
        -> presence filter
        -> master output
    """

    def __init__(
        self,
        sample_rate=44100,
        input_gain=3.0,
        second_stage_gain=2.2,
        bass_tightness=120.0,
        brightness=0.55,
        presence=0.35,
        master=0.75,
    ):
        self.sample_rate = int(sample_rate)

        self.input_gain = float(input_gain)
        self.second_stage_gain = float(second_stage_gain)
        self.bass_tightness = float(bass_tightness)
        self.brightness = float(np.clip(brightness, 0.0, 1.0))
        self.presence = float(np.clip(presence, 0.0, 1.0))
        self.master = float(np.clip(master, 0.0, 1.5))

        self._build_filters()

    def _build_filters(self):
        # Remove sub-bass and tighten palm-muted notes.
        self.hp_b, self.hp_a = butter(
            2,
            self.bass_tightness,
            btype="highpass",
            fs=self.sample_rate,
        )
        self.hp_state = lfilter_zi(self.hp_b, self.hp_a) * 0.0

        # Inter-stage low-pass. Lower cutoff gives a darker amp.
        stage_cutoff = 3200.0 + self.brightness * 4200.0

        self.stage_lp_b, self.stage_lp_a = butter(
            1,
            stage_cutoff,
            btype="lowpass",
            fs=self.sample_rate,
        )
        self.stage_lp_state = (
            lfilter_zi(self.stage_lp_b, self.stage_lp_a) * 0.0
        )

        # Presence path isolates upper mids/highs.
        presence_cutoff = 2200.0

        self.presence_b, self.presence_a = butter(
            1,
            presence_cutoff,
            btype="highpass",
            fs=self.sample_rate,
        )
        self.presence_state = (
            lfilter_zi(self.presence_b, self.presence_a) * 0.0
        )

    def set_input_gain(self, value):
        self.input_gain = float(np.clip(value, 0.1, 12.0))

    def set_second_stage_gain(self, value):
        self.second_stage_gain = float(np.clip(value, 0.1, 8.0))

    def set_brightness(self, value):
        self.brightness = float(np.clip(value, 0.0, 1.0))
        self._build_filters()

    def set_presence(self, value):
        self.presence = float(np.clip(value, 0.0, 1.0))

    def set_master(self, value):
        self.master = float(np.clip(value, 0.0, 1.5))

    def process(self, signal):
        x = np.asarray(signal, dtype=np.float32)

        # Input conditioning.
        x, self.hp_state = lfilter(
            self.hp_b,
            self.hp_a,
            x,
            zi=self.hp_state,
        )

        # Stage 1: asymmetric soft saturation.
        stage1_input = x * self.input_gain

        stage1 = np.where(
            stage1_input >= 0.0,
            np.tanh(stage1_input),
            0.88 * np.tanh(stage1_input * 1.15),
        )

        # Inter-stage filtering.
        stage1, self.stage_lp_state = lfilter(
            self.stage_lp_b,
            self.stage_lp_a,
            stage1,
            zi=self.stage_lp_state,
        )

        # Stage 2: slightly different nonlinearity.
        stage2_input = stage1 * self.second_stage_gain

        stage2 = (
            0.75 * np.tanh(stage2_input)
            + 0.25 * np.tanh(stage2_input * 0.45)
        )

        # Presence boost using a parallel high-frequency path.
        high_path, self.presence_state = lfilter(
            self.presence_b,
            self.presence_a,
            stage2,
            zi=self.presence_state,
        )

        output = stage2 + self.presence * 0.35 * high_path
        output *= self.master

        return np.clip(output, -1.0, 1.0).astype(np.float32)