import numpy as np

from config import (
    RATE,
    BLOCK_FRAMES,
    INPUT_GAIN_DB,
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
from tools.benchmark import DSPBenchmark
from effects.compressor import Compressor
from effects.power_amp import PowerAmp
from effects.output_gain import OutputGain
from engine.amp_factory import AmpFactory
from effects.input_gain import InputGain

amp_model = AmpFactory.create(
    "marshall",
    sample_rate=RATE,
)

input_gain = InputGain(
    gain_db=INPUT_GAIN_DB,
)

output_gain = OutputGain(
    gain_db=0.0,
)

power_amp = PowerAmp(
    sample_rate=RATE,
    drive=1.8,
    sag=0.20,
    resonance=0.20,
    presence=0.15,
    master=0.75,
)

compressor = Compressor(
    sample_rate=RATE,
    threshold_db=-18.0,
    ratio=4.0,
    attack_ms=5.0,
    release_ms=120.0,
    makeup_db=3.0,
    mix=1.0,
    block_size=BLOCK_FRAMES,
)

benchmark = DSPBenchmark(
    sample_rate=RATE,
    block_size=BLOCK_FRAMES,
)

rack = EffectRack(
    benchmark=benchmark,
)


gate = NoiseGate(
    threshold=0.008,
)

overdrive = Overdrive(
    gain=3.5,
    drive=0.45,
    level=0.75,
    tone=0.55,
    sample_rate=RATE,
)

eq = ThreeBandEQ(
    sample_rate=RATE,
    bass_db=0.0,
    mid_db=0.0,
    treble_db=0.0,
)

delay = Delay(
    sample_rate=RATE,
    delay_ms=350.0,
    feedback=0.35,
    mix=0.30,
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
    ir_path="5150_metal.wav",
    sample_rate=RATE,
    output_level=0.9,
    max_ir_samples=2048,
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
    block_size=BLOCK_FRAMES,
)

limiter = Limiter(
    sample_rate=RATE,
    threshold=0.92,
    release_ms=80.0,
    makeup_gain=1.0,
    block_size=BLOCK_FRAMES,
)

rack.add(input_gain)
rack.add(gate)
rack.add(compressor)
rack.add(overdrive)
rack.add(amp_model)
rack.add(eq)
rack.add(auto_wah)
rack.add(phaser)
rack.add(chorus)
rack.add(cabinet)
rack.add(delay)
rack.add(output_gain)
rack.add(limiter)

effects = {
    "gate": gate,
    "compressor": compressor,
    "overdrive": overdrive,
    "amp_model": amp_model,
    "eq": eq,
    "power_amp": power_amp,
    "auto_wah": auto_wah,
    "phaser": phaser,
    "cabinet": cabinet,
    "delay": delay,
    "chorus": chorus,
    "output_gain": output_gain,
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

presets.load("blues")
rack.enable(amp_model)


def print_benchmark():
    benchmark.report()


def process(guitar):
    guitar = np.asarray(
        guitar,
        dtype=np.float32,
    )

    return rack.process(guitar)