import numpy as np


class FIRConvolver:
    """
    Block FIR convolution with overlap-add state.

    Input and output use normalized float32 audio.
    """

    def __init__(self, impulse_response):
        impulse_response = np.asarray(
            impulse_response,
            dtype=np.float32,
        ).flatten()

        if impulse_response.size == 0:
            raise ValueError("Impulse response cannot be empty.")

        peak = float(np.max(np.abs(impulse_response)))

        if peak > 0.0:
            impulse_response = impulse_response / peak

        self.ir = impulse_response

        self.overlap = np.zeros(
            max(0, self.ir.size - 1),
            dtype=np.float32,
        )

    def reset(self):
        self.overlap.fill(0.0)

    def process(self, signal):
        signal = np.asarray(signal, dtype=np.float32)

        if signal.ndim != 1:
            raise ValueError("FIRConvolver expects mono one-dimensional audio.")

        convolved = np.convolve(
            signal,
            self.ir,
            mode="full",
        ).astype(np.float32)

        overlap_length = self.overlap.size

        if overlap_length > 0:
            add_length = min(overlap_length, convolved.size)

            convolved[:add_length] += self.overlap[:add_length]

        block_length = signal.size
        output = convolved[:block_length].copy()

        if overlap_length > 0:
            new_overlap = convolved[block_length:]

            self.overlap.fill(0.0)

            copy_length = min(
                new_overlap.size,
                self.overlap.size,
            )

            self.overlap[:copy_length] = new_overlap[:copy_length]

        return output