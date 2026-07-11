from pathlib import Path

import numpy as np
from scipy.io import wavfile

from engine.dsp.convolution import FIRConvolver


class Cabinet:
    def __init__(
        self,
        ir_path,
        sample_rate=44100,
        output_level=1.5,
        max_ir_samples=4096
    ):
        self.ir_path = Path(ir_path)
        self.sample_rate = int(sample_rate)
        self.output_level = float(output_level)

        ir_sample_rate, impulse_response = wavfile.read(self.ir_path)

        if ir_sample_rate != self.sample_rate:
            raise ValueError(
                f"IR sample rate is {ir_sample_rate} Hz, "
                f"but the audio engine uses {self.sample_rate} Hz."
            )

        # Convert stereo IRs to mono.
        if impulse_response.ndim == 2:
            impulse_response = np.mean(impulse_response, axis=1)

        # Convert PCM WAV data to floating point.
        if np.issubdtype(impulse_response.dtype, np.integer):
            maximum = float(np.iinfo(impulse_response.dtype).max)
            impulse_response = (
                impulse_response.astype(np.float32) / maximum
            )
        else:
            impulse_response = impulse_response.astype(np.float32)

        # Remove silence before the start of the impulse.
        active_samples = np.flatnonzero(
            np.abs(impulse_response) > 1e-6
        )

        if active_samples.size > 0:
            impulse_response = impulse_response[active_samples[0]:]

        # Keep the first portion initially to reduce CPU usage.
        if max_ir_samples is not None:
            impulse_response = impulse_response[:max_ir_samples]

        if impulse_response.size == 0:
            raise ValueError(f"IR file contains no audio: {self.ir_path}")

        self.convolver = FIRConvolver(impulse_response)

    def reset(self):
        self.convolver.reset()

    def process(self, signal):
        signal = np.asarray(signal, dtype=np.float32)

        output = self.convolver.process(signal)
        output *= self.output_level

        return np.clip(output, -32768, 32767).astype(np.int16)