import numpy as np

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
from engine.preset_manager import PresetManager

from effects.gate import NoiseGate
from effects.overdrive import Overdrive
from effects.eq import ThreeBandEQ
from effects.auto_wah import AutoWah
from effects.cabinet import Cabinet
from effects.delay import Delay


rack = EffectRack()


gate = NoiseGate(
    threshold=NOISE_GATE_THRESHOLD,
)

overdrive = Overdrive(
    gain=DISTORTION_GAIN,
    drive=DISTORTION_DRIVE,
    level=DISTORTION_LEVEL,
    tone=0.55,
    sample_rate=RATE,
)

eq = ThreeBandEQ(
    sample_rate=RATE,
    bass_db=BASS_DB,
    mid_db=MID_DB,
    treble_db=TREBLE_DB,
)

auto_wah = AutoWah(
    sample_rate=RATE,
    min_frequency=350.0,
    max_frequency=2200.0,
    rate_hz=1.2,
    resonance_q=4.0,
    mix=0.75,
)

cabinet = Cabinet(
    ir_path="irs/celestion.wav",
    sample_rate=RATE,
    output_level=0.9,
    max_ir_samples=2048,
)

delay = Delay(
    sample_rate=RATE,
    delay_ms=DELAY_MS,
    feedback=DELAY_FEEDBACK,
    mix=DELAY_MIX,
)


rack.add(gate)
rack.add(overdrive)
rack.add(eq)
rack.add(auto_wah)
rack.add(cabinet)
rack.add(delay)


effects = {
    "gate": gate,
    "overdrive": overdrive,
    "eq": eq,
    "auto_wah": auto_wah,
    "cabinet": cabinet,
    "delay": delay,
}


presets = PresetManager(
    rack=rack,
    effects=effects,
    presets_dir="presets",
)

print(
    "Available presets:",
    presets.available_presets(),
)

# Change this name to select the startup preset.
presets.load("metal")


def process(guitar):
    guitar = np.asarray(
        guitar,
        dtype=np.float32,
    )

    return rack.process(guitar)