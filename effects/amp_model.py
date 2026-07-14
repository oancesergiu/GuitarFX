import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi


AMP_MODELS = {
    "fender": {
        "preamp_gain": 1.8,
        "second_stage_gain": 1.2,
        "bass": 0.55,
        "middle": 0.35,
        "treble": 0.70,
        "presence": 0.35,
        "resonance": 0.20,
        "sag": 0.12,
        "master": 0.72,
        "input_highpass_hz": 70.0,
        "interstage_lowpass_hz": 7200.0,
        "asymmetry": 0.94,
    },
    "vox": {
        "preamp_gain": 2.7,
        "second_stage_gain": 1.5,
        "bass": 0.35,
        "middle": 0.55,
        "treble": 0.78,
        "presence": 0.42,
        "resonance": 0.15,
        "sag": 0.18,
        "master": 0.68,
        "input_highpass_hz": 85.0,
        "interstage_lowpass_hz": 6800.0,
        "asymmetry": 0.90,
    },
    "marshall": {
        "preamp_gain": 4.2,
        "second_stage_gain": 2.0,
        "bass": 0.48,
        "middle": 0.72,
        "treble": 0.58,
        "presence": 0.42,
        "resonance": 0.30,
        "sag": 0.20,
        "master": 0.64,
        "input_highpass_hz": 100.0,
        "interstage_lowpass_hz": 6200.0,
        "asymmetry": 0.86,
    },
    "5150": {
        "preamp_gain": 6.2,
        "second_stage_gain": 2.8,
        "bass": 0.52,
        "middle": 0.42,
        "treble": 0.62,
        "presence": 0.52,
        "resonance": 0.48,
        "sag": 0.10,
        "master": 0.56,
        "input_highpass_hz": 125.0,
        "interstage_lowpass_hz": 5700.0,
        "asymmetry": 0.82,
    },
    "mesa": {
        "preamp_gain": 5.7,
        "second_stage_gain": 2.6,
        "bass": 0.65,
        "middle": 0.30,
        "treble": 0.68,
        "presence": 0.48,
        "resonance": 0.55,
        "sag": 0.08,
        "master": 0.56,
        "input_highpass_hz": 115.0,
        "interstage_lowpass_hz": 5400.0,
        "asymmetry": 0.84,
    },
}


class AmpModel:
    """
    Lightweight block-based guitar amplifier model.

    Available personalities:
        fender
        vox
        marshall
        5150
        mesa

    Audio format:
        normalized mono float32
    """

    def __init__(
        self,
        sample_rate=44100,
        model="marshall",
    ):
        self.sample_rate = int(sample_rate)
        self.model = None
        self.sag_envelope = 0.0

        self.set_model(model)

    def _zero_state(self, numerator, denominator):
        return lfilter_zi(numerator, denominator) * 0.0

    def set_model(self, model):
        model = str(model).lower()

        if model not in AMP_MODELS:
            available = ", ".join(sorted(AMP_MODELS))
            raise ValueError(
                f"Unknown amp model '{model}'. "
                f"Available models: {available}"
            )

        settings = AMP_MODELS[model]

        self.model = model

        self.preamp_gain = float(settings["preamp_gain"])
        self.second_stage_gain = float(
            settings["second_stage_gain"]
        )

        self.bass = float(settings["bass"])
        self.middle = float(settings["middle"])
        self.treble = float(settings["treble"])

        self.presence = float(settings["presence"])
        self.resonance = float(settings["resonance"])
        self.sag = float(settings["sag"])
        self.master = float(settings["master"])

        self.input_highpass_hz = float(
            settings["input_highpass_hz"]
        )
        self.interstage_lowpass_hz = float(
            settings["interstage_lowpass_hz"]
        )
        self.asymmetry = float(settings["asymmetry"])

        self.sag_envelope = 0.0
        self._build_filters()

    def _build_filters(self):
        self.input_hp_b, self.input_hp_a = butter(
            2,
            self.input_highpass_hz,
            btype="highpass",
            fs=self.sample_rate,
        )
        # Pre-distortion bass path used to tighten or thicken the amp.
        self.pre_bass_b, self.pre_bass_a = butter(
            1,
            250.0,
            btype="lowpass",
            fs=self.sample_rate,
        )
        self.pre_bass_state = self._zero_state(
            self.pre_bass_b,
            self.pre_bass_a,
        )

        # Pre-distortion upper-mid/high path used for bite and clarity.
        self.pre_treble_b, self.pre_treble_a = butter(
            1,
            1200.0,
            btype="highpass",
            fs=self.sample_rate,
        )
        self.pre_treble_state = self._zero_state(
            self.pre_treble_b,
            self.pre_treble_a,
        )
        self.input_hp_state = self._zero_state(
            self.input_hp_b,
            self.input_hp_a,
        )

        self.interstage_lp_b, self.interstage_lp_a = butter(
            1,
            self.interstage_lowpass_hz,
            btype="lowpass",
            fs=self.sample_rate,
        )
        self.interstage_lp_state = self._zero_state(
            self.interstage_lp_b,
            self.interstage_lp_a,
        )

        self.bass_b, self.bass_a = butter(
            1,
            250.0,
            btype="lowpass",
            fs=self.sample_rate,
        )
        self.bass_state = self._zero_state(
            self.bass_b,
            self.bass_a,
        )

        self.mid_b, self.mid_a = butter(
            2,
            [350.0, 2200.0],
            btype="bandpass",
            fs=self.sample_rate,
        )
        self.mid_state = self._zero_state(
            self.mid_b,
            self.mid_a,
        )

        self.treble_b, self.treble_a = butter(
            1,
            2200.0,
            btype="highpass",
            fs=self.sample_rate,
        )
        self.treble_state = self._zero_state(
            self.treble_b,
            self.treble_a,
        )

        self.resonance_b, self.resonance_a = butter(
            2,
            130.0,
            btype="lowpass",
            fs=self.sample_rate,
        )
        self.resonance_state = self._zero_state(
            self.resonance_b,
            self.resonance_a,
        )

        self.presence_b, self.presence_a = butter(
            1,
            2800.0,
            btype="highpass",
            fs=self.sample_rate,
        )
        self.presence_state = self._zero_state(
            self.presence_b,
            self.presence_a,
        )

    def set_preamp_gain(self, value):
        self.preamp_gain = float(np.clip(value, 0.1, 12.0))

    def set_second_stage_gain(self, value):
        self.second_stage_gain = float(
            np.clip(value, 0.1, 8.0)
        )

    def set_bass(self, value):
        self.bass = float(np.clip(value, 0.0, 1.0))

    def set_middle(self, value):
        self.middle = float(np.clip(value, 0.0, 1.0))

    def set_treble(self, value):
        self.treble = float(np.clip(value, 0.0, 1.0))

    def set_presence(self, value):
        self.presence = float(np.clip(value, 0.0, 1.0))

    def set_resonance(self, value):
        self.resonance = float(np.clip(value, 0.0, 1.0))

    def set_sag(self, value):
        self.sag = float(np.clip(value, 0.0, 0.8))

    def set_master(self, value):
        self.master = float(np.clip(value, 0.0, 1.5))

    def reset(self):
        self.sag_envelope = 0.0
        self._build_filters()
    def _stage1_curve(self, signal):
        """
        Model-specific first preamp-stage nonlinearity.
        """
        x = np.asarray(signal, dtype=np.float64)

        if self.model == "fender":
            # Open and dynamic, with gentle symmetrical breakup.
            return (
                0.82 * np.tanh(x * 0.80)
                + 0.18 * np.tanh(x * 0.28)
            )

        if self.model == "vox":
            # Brighter and more asymmetric.
            positive = np.tanh(x * 1.05)
            negative = 0.88 * np.tanh(x * 1.22)

            return np.where(x >= 0.0, positive, negative)

        if self.model == "marshall":
            # Strong asymmetric upper-mid character.
            positive = np.tanh(x * 1.18)
            negative = 0.80 * np.tanh(x * 1.38)

            return np.where(x >= 0.0, positive, negative)

        if self.model == "5150":
            # Tight and compressed high-gain response.
            first = np.tanh(x * 1.45)
            second = np.tanh(first * 1.65)

            return 0.35 * first + 0.65 * second

        if self.model == "mesa":
            # Thick cascading saturation.
            first = np.tanh(x * 1.28)
            second = np.tanh((first + 0.10 * x) * 1.55)

            return 0.42 * first + 0.58 * second

        return np.tanh(x)


    def _stage2_curve(self, signal):
        """
        Model-specific second gain-stage and power-stage character.
        """
        x = np.asarray(signal, dtype=np.float64)

        if self.model == "fender":
            return (
                0.76 * np.tanh(x * 0.72)
                + 0.24 * x / (1.0 + np.abs(x))
            )

        if self.model == "vox":
            return (
                0.72 * np.tanh(x * 1.00)
                + 0.28 * np.tanh(x * 0.35)
            )

        if self.model == "marshall":
            positive = np.tanh(x * 1.08)
            negative = 0.86 * np.tanh(x * 1.20)

            return np.where(x >= 0.0, positive, negative)

        if self.model == "5150":
            compressed = np.tanh(x * 1.45)
            return np.tanh(compressed * 1.30)

        if self.model == "mesa":
            first = np.tanh(x * 1.18)
            return (
                0.30 * first
                + 0.70 * np.tanh(first * 1.48)
            )

        return np.tanh(x)

    def _pre_emphasis(self, signal):
        """
        Shape the signal before distortion.
        """

        if self.model == "fender":
            return signal

        if self.model == "marshall":
            return (
                signal
                + 0.18 * self.treble_path
                - 0.10 * self.bass_path
            )

        if self.model == "5150":
            return (
                signal
                + 0.24 * self.treble_path
                - 0.22 * self.bass_path
            )

        if self.model == "mesa":
            return (
                signal
                + 0.08 * self.bass_path
                + 0.10 * self.mid_path
            )

        if self.model == "vox":
            return (
                signal
                + 0.20 * self.treble_path
            )

        return signal

    def _apply_pre_emphasis(
    self,
    signal,
    bass_path,
    treble_path,
    ):
        """
        Shape the signal before distortion.

        This changes which frequencies hit the nonlinear stages,
        strongly affects the amp's character.
        """

        if self.model == "fender":
            return (
                signal
                - 0.04 * bass_path
                + 0.03 * treble_path
            )

        if self.model == "vox":
            return (
                signal
                - 0.10 * bass_path
                + 0.18 * treble_path
            )

        if self.model == "marshall":
            return (
                signal
                - 0.12 * bass_path
                + 0.14 * treble_path
            )

        if self.model == "5150":
            return (
                signal
                - 0.24 * bass_path
                + 0.18 * treble_path
            )

        if self.model == "mesa":
            return (
                signal
                + 0.06 * bass_path
                + 0.07 * treble_path
            )

        return signal

    def process_inplace(self, buffer):
        x = np.asarray(buffer, dtype=np.float32)

        if x.size == 0:
            return

        x_filtered, self.input_hp_state = lfilter(
            self.input_hp_b,
            self.input_hp_a,
            x,
            zi=self.input_hp_state,
        )

        pre_bass, self.pre_bass_state = lfilter(
            self.pre_bass_b,
            self.pre_bass_a,
            x_filtered,
            zi=self.pre_bass_state,
        )

        pre_treble, self.pre_treble_state = lfilter(
            self.pre_treble_b,
            self.pre_treble_a,
            x_filtered,
            zi=self.pre_treble_state,
        )

        voiced_input = self._apply_pre_emphasis(
            x_filtered,
            pre_bass,
            pre_treble,
        )

        stage1_input = voiced_input * self.preamp_gain
        stage1 = self._stage1_curve(stage1_input)

        stage1, self.interstage_lp_state = lfilter(
            self.interstage_lp_b,
            self.interstage_lp_a,
            stage1,
            zi=self.interstage_lp_state,
        )

        stage2_input = stage1 * self.second_stage_gain
        stage2 = self._stage2_curve(stage2_input)

        low, self.bass_state = lfilter(
            self.bass_b,
            self.bass_a,
            stage2,
            zi=self.bass_state,
        )

        mids, self.mid_state = lfilter(
            self.mid_b,
            self.mid_a,
            stage2,
            zi=self.mid_state,
        )

        highs, self.treble_state = lfilter(
            self.treble_b,
            self.treble_a,
            stage2,
            zi=self.treble_state,
        )

        tone = (
            stage2 * 0.30
            + low * (0.12 + self.bass * 0.42)
            + mids * (0.12 + self.middle * 0.55)
            + highs * (0.08 + self.treble * 0.38)
        )

        block_level = float(
            np.sqrt(np.mean(tone * tone) + 1e-12)
        )

        smoothing = (
            0.30
            if block_level > self.sag_envelope
            else 0.025
        )

        self.sag_envelope += smoothing * (
            block_level - self.sag_envelope
        )

        sag_gain = 1.0 - self.sag * np.clip(
            self.sag_envelope * 2.0,
            0.0,
            1.0,
        )

        power_input = tone * sag_gain * 1.35

        power_stage = (
            0.78 * np.tanh(power_input)
            + 0.22 * np.tanh(power_input * 0.35)
        )

        low_resonance, self.resonance_state = lfilter(
            self.resonance_b,
            self.resonance_a,
            power_stage,
            zi=self.resonance_state,
        )

        high_presence, self.presence_state = lfilter(
            self.presence_b,
            self.presence_a,
            power_stage,
            zi=self.presence_state,
        )

        output = (
            power_stage
            + low_resonance * self.resonance * 0.30
            + high_presence * self.presence * 0.25
        )

        output *= self.master

        np.copyto(
            buffer,
            np.clip(output, -1.0, 1.0).astype(
                np.float32,
                copy=False,
            ),
        )

    def process(self, signal):
        output = np.asarray(
            signal,
            dtype=np.float32,
        ).copy()

        self.process_inplace(output)
        return output