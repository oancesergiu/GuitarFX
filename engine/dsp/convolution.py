import numpy as np
from scipy.fft import irfft, next_fast_len, rfft


class FIRConvolver:
    """
    Stateful FFT-based FIR convolver using overlap-add.

    Input and output are normalized mono float32 audio.
    """

    def __init__(self, impulse_response):
        ir = np.asarray(
            impulse_response,
            dtype=np.float32,
        ).flatten()

        if ir.size == 0:
            raise ValueError("Impulse response cannot be empty.")

        peak = float(np.max(np.abs(ir)))

        if peak > 0.0:
            ir = ir / peak

        self.ir = ir
        self.ir_length = ir.size

        self.overlap = np.zeros(
            max(0, self.ir_length - 1),
            dtype=np.float32,
        )

        self.block_size = None
        self.fft_size = None
        self.ir_spectrum = None

    def _prepare_fft(self, block_size):
        """
        Prepare the IR spectrum for the current audio block size.
        """
        self.block_size = int(block_size)

        convolution_length = (
            self.block_size
            + self.ir_length
            - 1
        )

        self.fft_size = next_fast_len(
            convolution_length
        )

        self.ir_spectrum = rfft(
            self.ir,
            n=self.fft_size,
        )

    def reset(self):
        self.overlap.fill(0.0)

    def process(self, signal):
        signal = np.asarray(
            signal,
            dtype=np.float32,
        )

        if signal.ndim != 1:
            raise ValueError(
                "FIRConvolver expects mono one-dimensional audio."
            )

        if signal.size == 0:
            return signal.copy()

        # Rebuild the FFT configuration only if block size changes.
        if self.block_size != signal.size:
            self._prepare_fft(signal.size)

        input_spectrum = rfft(
            signal,
            n=self.fft_size,
        )

        convolved = irfft(
            input_spectrum * self.ir_spectrum,
            n=self.fft_size,
        )

        valid_length = (
            signal.size
            + self.ir_length
            - 1
        )

        convolved = convolved[:valid_length]

        overlap_length = self.overlap.size

        if overlap_length > 0:
            convolved[:overlap_length] += self.overlap

        output = convolved[:signal.size].copy()

        if overlap_length > 0:
            new_overlap = convolved[
                signal.size:
                signal.size + overlap_length
            ]

            self.overlap.fill(0.0)
            self.overlap[:new_overlap.size] = new_overlap

        return output.astype(
            np.float32,
            copy=False,
        )