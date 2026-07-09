from config import (
    RATE,
    NOISE_GATE_THRESHOLD,
    DISTORTION_GAIN,
    DISTORTION_DRIVE,
    DISTORTION_LEVEL,
    DELAY_MS,
    DELAY_FEEDBACK,
    DELAY_MIX,
    BASS_DB,
    MID_DB,
    TREBLE_DB,
)

from engine.effect_rack import EffectRack

from effects.gate import NoiseGate
from effects.overdrive import Overdrive
from effects.eq import ThreeBandEQ
from effects.delay import Delay


rack = EffectRack()

rack.add(NoiseGate(threshold=NOISE_GATE_THRESHOLD))

rack.add(
    Overdrive(
        gain=DISTORTION_GAIN,
        drive=DISTORTION_DRIVE,
        level=DISTORTION_LEVEL,
    )
)

rack.add(
    ThreeBandEQ(
        sample_rate=RATE,
        bass_db=BASS_DB,
        mid_db=MID_DB,
        treble_db=TREBLE_DB,
    )
)

delay = Delay(
    sample_rate=RATE,
    delay_ms=DELAY_MS,
    feedback=DELAY_FEEDBACK,
    mix=DELAY_MIX,
)

rack.add(delay)
#rack.remove(delay)



def process(guitar):
    return rack.process(guitar)