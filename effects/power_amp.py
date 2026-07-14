import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi


class PowerAmp:
    """
    Lightweight guitar power-amp approximation for normalized float32 audio.

    Models:
    - power-tube-style soft saturation
    - dynamic sag
    - low-frequency resonance
    - high-frequency presence
    - master output level
    """

    def __init__(
        self,
        sample_rate=44100,
        drive=1.8,
        sag=0.20,
        resonance=0.20,
        presence=0.15,
        master=0.75,
    ):
        self.sample_rate = int(sample_rate)

        self.drive = float(drive)
        self.sag = float(np.clip(sag, 0.0, 0.8))
        self.resonance = float(np.clip(resonance, 0.0, 1.0))
        self.presence = float(np.clip(presence, 0.0, 1.0))
        self.master = float(np.clip(master, 0.0, 1.5))

        self.sag_envelope = 0.0

        self._build_filters()

    def _build_filters(self):
        # Low-frequency speaker/power-section resonance path.
        self.low_b, self.low_a = butter(
            2,
            140.0,
            btype="lowpass",
            fs=self.sample_rate,
        )
        self.low_state = lfilter_zi(
            self.low_b,
            self.low_a,
        ) * 0.0

        # Presence path in the upper-mid/high-frequency region.
        self.high_b, self.high_a = butter(
            1,
            2500.0,
            btype="highpass",
            fs=self.sample_rate,
        )
        self.high_state = lfilter_zi(
            self.high_b,
            self.high_a,
        ) * 0.0

    def set_drive(self, value):
        self.drive = float(np.clip(value, 0.1, 5.0))

    def set_sag(self, value):
        self.sag = float(np.clip(value, 0.0, 0.8))

    def set_resonance(self, value):
        self.resonance = float(np.clip(value, 0.0, 1.0))

    def set_presence(self, value):
        self.presence = float(np.clip(value, 0.0, 1.0))

    def set_master(self, value):
        self.master = float(np.clip(value, 0.0, 1.5))

    def reset(self):
        self.sag_envelope = 0.0
        self._build_filters()

    def process(self, signal):
        signal = np.asarray(signal, dtype=np.float32)

        # Track block energy for simple supply-sag behavior.
        block_level = float(np.sqrt(np.mean(signal * signal) + 1e-12))

        if block_level > self.sag_envelope:
            smoothing = 0.25
        else:
            smoothing = 0.02

        self.sag_envelope += smoothing * (
            block_level - self.sag_envelope
        )

        sag_gain = 1.0 - self.sag * np.clip(
            self.sag_envelope * 2.0,
            0.0,
            1.0,
        )

        driven = signal * self.drive * sag_gain

        # Two blended nonlinear responses give a softer,
        # more rounded saturation than a single clipper.
        saturated = (
            0.72 * np.tanh(driven)
            + 0.28 * np.tanh(driven * 0.45)
        )

        low_path, self.low_state = lfilter(
            self.low_b,
            self.low_a,
            saturated,
            zi=self.low_state,
        )

        high_path, self.high_state = lfilter(
            self.high_b,
            self.high_a,
            saturated,
            zi=self.high_state,
        )

        output = (
            saturated
            + self.resonance * 0.35 * low_path
            + self.presence * 0.30 * high_path
        )

        output *= self.master

        return np.clip(
            output,
            -1.0,
            1.0,
        ).astype(np.float32)