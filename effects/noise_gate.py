import numpy as np


def noise_gate(signal, threshold=0.008):
    """
    Soft noise gate for normalized float32 audio.

    signal range:
        approximately -1.0 ... +1.0
    """
    signal = np.asarray(signal, dtype=np.float32)

    level = np.abs(signal)

    gain = np.ones_like(signal)

    quiet = level < threshold
    gain[quiet] = level[quiet] / threshold

    return signal * gain