import numpy as np

def soft_clip(signal, gain=3.5, drive=16000.0, level=22000.0):
    signal = signal.astype(np.float32)

    # Drive stage
    signal = signal * gain

    # Smooth saturation
    signal = np.tanh(signal / drive)

    # Output level
    signal = signal * level

    return signal.astype(np.int16)