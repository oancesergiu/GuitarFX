import numpy as np
from effects.distortion import soft_clip


def process(guitar):
    # Pre-gain
    guitar = guitar * 1.5
    guitar = np.clip(guitar, -32768, 32767)

    # Distortion
    guitar = soft_clip(
        guitar,
        gain=6.0,
        drive=10000.0,
        level=28000.0
    )

    return guitar