import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi


AMP_MODELS = {
    "fender": {
        "preamp_gain_min": 1.2,
        "preamp_gain_max": 3.6,
        "stage2_gain_min": 1.0,
        "stage2_gain_max": 1.9,

        "sag_attack_ms": 35.0,
        "sag_release_ms": 240.0,

        "bass": 0.55,
        "middle": 0.35,
        "treble": 0.70,
        "presence": 0.33,
        "resonance": 0.20,
        "sag": 0.12,
        "master": 0.72,
        "input_highpass_hz": 70.0,
        "interstage_lowpass_hz": 6000.0,
        "asymmetry": 0.65,
    },
    "vox": {
        "preamp_gain_min": 1.5,
        "preamp_gain_max": 4.7,
        "stage2_gain_min": 1.0,
        "stage2_gain_max": 2.4,

        "sag_attack_ms": 22.0,
        "sag_release_ms": 190.0,

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
        "preamp_gain_min": 1.8,
        "preamp_gain_max": 6.6,
        "stage2_gain_min": 1.1,
        "stage2_gain_max": 3.0,

        "sag_attack_ms": 18.0,
        "sag_release_ms": 150.0,

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
        "preamp_gain_min": 2.5,
        "preamp_gain_max": 9.0,
        "stage2_gain_min": 1.4,
        "stage2_gain_max": 4.0,

        "sag_attack_ms": 8.0,
        "sag_release_ms": 90.0,

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
        "preamp_gain_min": 2.3,
        "preamp_gain_max": 8.3,
        "stage2_gain_min": 1.3,
        "stage2_gain_max": 3.8,

        "sag_attack_ms": 12.0,
        "sag_release_ms": 120.0,

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
        
        self.preamp_gain_min = float(settings["preamp_gain_min"])
        self.preamp_gain_max = float(settings["preamp_gain_max"])

        self.stage2_gain_min = float(settings["stage2_gain_min"])
        self.stage2_gain_max = float(settings["stage2_gain_max"])

        # Default amp gain at about 60%
        self.set_gain(0.60)

        self.bass = float(settings["bass"])
        self.middle = float(settings["middle"])
        self.treble = float(settings["treble"])

        self.presence = float(settings["presence"])
        self.resonance = float(settings["resonance"])
        self.sag = float(settings["sag"])
        self.sag_attack_ms = float(
            settings["sag_attack_ms"]
        )

        self.sag_release_ms = float(
            settings["sag_release_ms"]
        )
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

    def set_gain(self, gain):
        gain = float(np.clip(gain, 0.0, 1.0))
        self.gain = gain

        # Model-specific gain-knob response.
        #
        # Exponents above 1.0 preserve more clean headroom.
        # Exponents below 1.0 bring gain in earlier.
        gain_exponents = {
            "fender": 1.35,
            "vox": 1.05,
            "marshall": 0.90,
            "mesa": 0.80,
            "5150": 0.70,
        }

        exponent = gain_exponents.get(
            self.model,
            1.0,
        )

        effective_gain = gain ** exponent

        self.preamp_gain = (
            self.preamp_gain_min
            + effective_gain
            * (
                self.preamp_gain_max
                - self.preamp_gain_min
            )
        )

        self.second_stage_gain = (
            self.stage2_gain_min
            + effective_gain
            * (
                self.stage2_gain_max
                - self.stage2_gain_min
            )
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
            # Mostly linear at normal playing levels, with gentle breakup
            # only when the stage is pushed harder.
            soft = np.tanh(x * 0.55)

            return (
                0.72 * x
                + 0.28 * soft
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
            # Preserve clean dynamics and add only mild power-stage rounding.
            rounded = x / (
                1.0 + 0.20 * np.abs(x)
            )

            return (
                0.82 * x
                + 0.18 * rounded
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
            # Clean, open, lots of headroom.
            return (
                signal
                - 0.02 * bass_path
                + 0.02 * treble_path
            )

        if self.model == "vox":
            # Chime: tighter bass and stronger upper mids.
            return (
                signal
                - 0.12 * bass_path
                + 0.22 * treble_path
            )

        if self.model == "marshall":
            # Crunch: cut low-end before distortion, emphasize mids.
            return (
                signal
                - 0.16 * bass_path
                + 0.12 * treble_path
            )

        if self.model == "5150":
            # Modern high gain: very tight bass, controlled highs.
            return (
                signal
                - 0.30 * bass_path
                + 0.12 * treble_path
            )

        if self.model == "mesa":
            # Smooth lead: not as bright as the 5150.
            return (
                signal
                - 0.20 * bass_path
                + 0.10 * treble_path
            )

        return signal

    def _apply_gain_topology(self, stage1):
        """
        Apply the model-specific preamp gain structure.
        """

        if self.model == "fender":
            # High-headroom clean topology.
            return stage1

        if self.model == "vox":
            # Earlier breakup, but less cascaded gain than a Marshall.
            stage2_input = stage1 * (
                0.75 + 0.45 * self.second_stage_gain
            )
            return self._stage2_curve(stage2_input)

        if self.model == "marshall":
            # Traditional cascaded crunch topology.
            stage2_input = stage1 * self.second_stage_gain
            return self._stage2_curve(stage2_input)

        if self.model == "5150":
            # Aggressive cascaded high-gain topology.
            stage2_input = stage1 * self.second_stage_gain
            stage2 = self._stage2_curve(stage2_input)

            return np.tanh(
                stage2 * 1.18
            )

        if self.model == "mesa":
            # Smoother cascaded saturation with some stage-one clarity.
            stage2_input = stage1 * self.second_stage_gain
            stage2 = self._stage2_curve(stage2_input)

            return (
                0.18 * stage1
                + 0.82 * stage2
            )

        return stage1

    def _power_amp(self, signal):
        if self.model == "fender":
            return (
                0.82 * np.tanh(signal * 0.85)
                + 0.18 * signal
            )

        if self.model == "vox":
            return (
                0.72 * np.tanh(signal * 1.05)
                + 0.28 * np.tanh(signal * 0.45)
            )

        if self.model == "marshall":
            return (
                0.82 * np.tanh(signal * 1.10)
                + 0.18 * np.tanh(signal * 0.60)
            )

        if self.model == "5150":
            return np.tanh(signal * 1.18)

        if self.model == "mesa":
            return (
                0.60 * np.tanh(signal)
                + 0.40 * np.tanh(signal * 0.55)
            )

        return np.tanh(signal)

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

        stage2 = self._apply_gain_topology(stage1)

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
        block_size = tone.size

        attack_coefficient = np.exp(
            -block_size
            / (
                self.sample_rate
                * self.sag_attack_ms
                / 1000.0
            )
        )

        release_coefficient = np.exp(
            -block_size
            / (
                self.sample_rate
                * self.sag_release_ms
                / 1000.0
            )
        )

        coefficient = (
            attack_coefficient
            if block_level > self.sag_envelope
            else release_coefficient
        )

        self.sag_envelope = (
            coefficient * self.sag_envelope
            + (1.0 - coefficient) * block_level
        )

        sag_amount = float(
            np.clip(
                self.sag_envelope * 2.0,
                0.0,
                1.0,
            )
        )

        sag_gain = 1.0 - 0.30 * self.sag * sag_amount

        power_input = tone * sag_gain * 1.35
        power_stage = self._power_amp(power_input)

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