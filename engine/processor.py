import numpy as np
from effects.gain import apply_gain
from effects.distortion import soft_clip
from effects.noise_gate import noise_gate
from effects.delay import Delay

delay = Delay(
    sample_rate=44100,
    delay_ms=350,
    feedback=0.35,
    mix=0.30
)

def process(guitar):
    # 1. Noise gate before distortion
    guitar = noise_gate(guitar, threshold=250)

    # 2. Pre-gain
    guitar = guitar * 1.0

    # 3. Distortion
    guitar = soft_clip(
        guitar,
        gain=4.0,
        drive=16000.0,
        level=22000.0
    )

    guitar = delay.process(guitar)
    return guitar