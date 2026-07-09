import numpy as np

from config import (
    RATE,
    NOISE_GATE_THRESHOLD,
    INPUT_GAIN,
    DISTORTION_GAIN,
    DISTORTION_DRIVE,
    DISTORTION_LEVEL,
    DELAY_MS,
    DELAY_FEEDBACK,
    DELAY_MIX,
)

from effects.noise_gate import noise_gate
from effects.distortion import soft_clip
from effects.delay import Delay


delay = Delay(
    sample_rate=RATE,
    delay_ms=DELAY_MS,
    feedback=DELAY_FEEDBACK,
    mix=DELAY_MIX
)


def process(guitar):
    guitar = noise_gate(guitar, threshold=NOISE_GATE_THRESHOLD)

    guitar = guitar * INPUT_GAIN
    guitar = np.clip(guitar, -32768, 32767)

    guitar = soft_clip(
        guitar,
        gain=DISTORTION_GAIN,
        drive=DISTORTION_DRIVE,
        level=DISTORTION_LEVEL
    )

    guitar = delay.process(guitar)

    return guitar