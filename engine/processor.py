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
from effects.preamp import Preamp
from effects.gate import NoiseGate
from effects.overdrive import Overdrive
from effects.eq import ThreeBandEQ
from effects.auto_wah import AutoWah
from effects.cabinet import Cabinet
from effects.delay import Delay
from effects.phaser import Phaser
from effects.chorus import Chorus
from engine.dsp.limiter import Limiter

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

preamp = Preamp(
    sample_rate=RATE,
    input_gain=2.2,
    second_stage_gain=1.6,
    bass_tightness=100.0,
    brightness=0.35,
    presence=0.10,
    master=0.65,
)

phaser = Phaser(
    sample_rate=RATE,
    rate_hz=0.6,
    min_frequency=300.0,
    max_frequency=1800.0,
    stages=4,
    feedback=0.25,
    mix=0.5,
)

chorus = Chorus(
    sample_rate=RATE,
    rate_hz=0.8,
    base_delay_ms=18.0,
    depth_ms=4.0,
    mix=0.35,
)

limiter = Limiter(
    sample_rate=RATE,
    threshold=0.92,
    release_ms=80.0,
    makeup_gain=1.0,
)

rack.add(gate)
rack.add(overdrive)
rack.add(eq)
rack.add(auto_wah)
rack.add(phaser)
rack.add(chorus)
rack.add(cabinet)
rack.add(delay)
rack.add(limiter)

effects = {
    "gate": gate,
    "overdrive": overdrive,
    "eq": eq,
    "auto_wah": auto_wah,
    "phaser": phaser,
    "cabinet": cabinet,
    "delay": delay,
    "chorus": chorus,
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

presets.load("clean")
rack.enable(chorus)


def process(guitar):
    guitar = np.asarray(
        guitar,
        dtype=np.float32,
    )

    return rack.process(guitar)