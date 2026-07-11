from pathlib import Path

import numpy as np
from scipy.io import wavfile

from engine.dsp.convolution import FIRConvolver


class Cabinet:
    """
    Cabinet impulse-response effect using normalized float32 audio.
    """

    def __init__(
        self,
        ir_path,
        sample_rate=44100,
        output_level=0.9,
        max_ir_samples=2048,
    ):
        self.sample_rate = int(sample_rate)
        self.output_level = float(
            np.clip(output_level, 0.0, 2.0)
        )

        self.ir_path = Path(ir_path)
        self.max_ir_samples = max_ir_samples

        impulse_response = self._load_ir(self.ir_path)

        self.convolver = FIRConvolver(impulse_response)

    def _load_ir(self, ir_path):
        if not ir_path.exists():
            raise FileNotFoundError(
                f"Cabinet IR was not found: {ir_path}"
            )

        ir_sample_rate, impulse_response = wavfile.read(ir_path)

        if int(ir_sample_rate) != self.sample_rate:
            raise ValueError(
                f"IR uses {ir_sample_rate} Hz, but the audio engine "
                f"uses {self.sample_rate} Hz."
            )

        # Convert stereo IRs to mono.
        if impulse_response.ndim == 2:
            impulse_response = np.mean(
                impulse_response.astype(np.float32),
                axis=1,
            )

        # Convert PCM samples to normalized float32.
        if np.issubdtype(impulse_response.dtype, np.integer):
            maximum = float(
                max(
                    abs(np.iinfo(impulse_response.dtype).min),
                    np.iinfo(impulse_response.dtype).max,
                )
            )

            impulse_response = (
                impulse_response.astype(np.float32)
                / maximum
            )
        else:
            impulse_response = impulse_response.astype(np.float32)

        # Remove silence before the initial impulse.
        active_samples = np.flatnonzero(
            np.abs(impulse_response) > 1e-7
        )

        if active_samples.size == 0:
            raise ValueError(
                f"IR contains no usable audio: {ir_path}"
            )

        impulse_response = impulse_response[
            active_samples[0]:
        ]

        if self.max_ir_samples is not None:
            impulse_response = impulse_response[
                : int(self.max_ir_samples)
            ]

        if impulse_response.size == 0:
            raise ValueError(
                f"IR is empty after preparation: {ir_path}"
            )

        return impulse_response

    def set_output_level(self, output_level):
        self.output_level = float(
            np.clip(output_level, 0.0, 2.0)
        )

    def reset(self):
        self.convolver.reset()

    def process(self, signal):
        signal = np.asarray(signal, dtype=np.float32)

        output = self.convolver.process(signal)
        output *= self.output_level

        return np.clip(
            output,
            -1.0,
            1.0,
        ).astype(np.float32)