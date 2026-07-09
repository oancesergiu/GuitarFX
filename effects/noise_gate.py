import numpy as np

def noise_gate(signal, threshold=250):
    signal = signal.astype(np.float32)

    # Soft gate instead of hard mute
    level = np.abs(signal)

    gain = np.ones_like(signal)
    gain[level < threshold] = level[level < threshold] / threshold

    return signal * gain