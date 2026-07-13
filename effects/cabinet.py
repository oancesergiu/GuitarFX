from pathlib import Path

import numpy as np
from scipy.io import wavfile

from engine.dsp.convolution import FIRConvolver


class Cabinet:
    """
    Cabinet IR effect with runtime impulse-response loading.
    """

    def __init__(
        self,
        ir_path,
        sample_rate=44100,
        output_level=0.9,
        max_ir_samples=2048,
    ):
        self.sample_rate = int(sample_rate)
        self.output_level = float(np.clip(output_level, 0.0, 2.0))
        self.max_ir_samples = max_ir_samples

        self.ir_path = None
        self.convolver = None

        self.load_ir(ir_path)

    def _read_ir(self, ir_path):
        path = Path(ir_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Cabinet IR was not found: {path}"
            )

        ir_sample_rate, impulse_response = wavfile.read(path)

        if int(ir_sample_rate) != self.sample_rate:
            raise ValueError(
                f"IR uses {ir_sample_rate} Hz, but the audio engine "
                f"uses {self.sample_rate} Hz."
            )

        if impulse_response.ndim == 2:
            impulse_response = np.mean(
                impulse_response.astype(np.float32),
                axis=1,
            )

        if np.issubdtype(impulse_response.dtype, np.integer):
            info = np.iinfo(impulse_response.dtype)
            scale = float(max(abs(info.min), info.max))

            impulse_response = (
                impulse_response.astype(np.float32) / scale
            )
        else:
            impulse_response = impulse_response.astype(np.float32)

        active_samples = np.flatnonzero(
            np.abs(impulse_response) > 1e-7
        )

        if active_samples.size == 0:
            raise ValueError(
                f"IR contains no usable audio: {path}"
            )

        impulse_response = impulse_response[active_samples[0]:]

        if self.max_ir_samples is not None:
            impulse_response = impulse_response[
                : int(self.max_ir_samples)
            ]

        if impulse_response.size == 0:
            raise ValueError(
                f"IR is empty after preparation: {path}"
            )

        return path, impulse_response

    def load_ir(self, ir_name):
        path = Path("irs") / ir_name

        path, impulse_response = self._read_ir(path)

        self.ir_path = path
        self.convolver = FIRConvolver(impulse_response)

        print(f"Loaded cabinet IR: {path.name}")

    def set_output_level(self, output_level):
        self.output_level = float(
            np.clip(output_level, 0.0, 2.0)
        )

    def reset(self):
        if self.convolver is not None:
            self.convolver.reset()

    def process(self, signal):
        signal = np.asarray(signal, dtype=np.float32)

        if self.convolver is None:
            return signal

        output = self.convolver.process(signal)
        output *= self.output_level

        return np.clip(
            output,
            -1.0,
            1.0,
        ).astype(np.float32)