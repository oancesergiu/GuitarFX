from pathlib import Path

import numpy as np
from scipy.io.wavfile import write
from scipy.signal import butter, lfilter


SAMPLE_RATE = 44100
IR_LENGTH = 1024

OUTPUT_PATH = Path("irs/test_cabinet.wav")


def generate_cabinet_ir():
    impulse = np.zeros(IR_LENGTH, dtype=np.float32)
    impulse[0] = 1.0

    # Remove very low frequencies below approximately 80 Hz
    highpass_b, highpass_a = butter(
        2,
        80.0,
        btype="highpass",
        fs=SAMPLE_RATE,
    )

    # Roll off harsh guitar frequencies above approximately 5500 Hz
    lowpass_b, lowpass_a = butter(
        4,
        5500.0,
        btype="lowpass",
        fs=SAMPLE_RATE,
    )

    cabinet_ir = lfilter(highpass_b, highpass_a, impulse)
    cabinet_ir = lfilter(lowpass_b, lowpass_a, cabinet_ir)

    # Add a short resonance/decay to imitate speaker coloration
    time = np.arange(IR_LENGTH) / SAMPLE_RATE

    resonance = (
        0.18
        * np.sin(2.0 * np.pi * 1100.0 * time)
        * np.exp(-time * 90.0)
    )

    cabinet_ir += resonance.astype(np.float32)

    # Normalize safely
    maximum = np.max(np.abs(cabinet_ir))

    if maximum > 0:
        cabinet_ir /= maximum

    cabinet_ir *= 0.9

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    write(
        OUTPUT_PATH,
        SAMPLE_RATE,
        (cabinet_ir * 32767).astype(np.int16),
    )

    print(f"Created cabinet IR: {OUTPUT_PATH}")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Length: {IR_LENGTH} samples")


if __name__ == "__main__":
    generate_cabinet_ir()