import numpy as np


INT16_SCALE = 32768.0


def int16_to_float32(signal):
    """
    Convert signed 16-bit PCM samples into normalized float32 samples.

    int16 range:
        -32768 ... 32767

    float32 range:
        approximately -1.0 ... +1.0
    """
    signal = np.asarray(signal, dtype=np.int16)
    return signal.astype(np.float32) / INT16_SCALE


def float32_to_int16(signal):
    """
    Convert normalized float32 samples back into signed 16-bit PCM.
    """
    signal = np.asarray(signal, dtype=np.float32)
    signal = np.clip(signal, -1.0, 32767.0 / INT16_SCALE)

    return (signal * INT16_SCALE).astype(np.int16)