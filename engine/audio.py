import subprocess
import numpy as np
from effects.distortion import soft_clip
from effects.gain import apply_gain
from engine.processor import process

RATE = 44100
CHANNELS = 2
BLOCK_FRAMES = 256
DEVICE = "hw:3,0"

GAIN = 5.0


def run_audio():
    arecord_cmd = [
        "arecord",
        "-D", DEVICE,
        "-f", "S16_LE",
        "-r", str(RATE),
        "-c", str(CHANNELS),
        "--buffer-size=1024",
        "--period-size=256",
    ]

    aplay_cmd = [
        "aplay",
        "-D", DEVICE,
        "-f", "S16_LE",
        "-r", str(RATE),
        "-c", str(CHANNELS),
        "--buffer-size=1024",
        "--period-size=256",
    ]

    arecord = subprocess.Popen(
        arecord_cmd,
        stdout=subprocess.PIPE,
        bufsize=0
    )

    aplay = subprocess.Popen(
        aplay_cmd,
        stdin=subprocess.PIPE,
        bufsize=0
    )

    print("Python ALSA audio engine running. Press Ctrl+C to stop.")

    bytes_per_frame = CHANNELS * 2
    block_bytes = BLOCK_FRAMES * bytes_per_frame

    try:
        while True:
            data = arecord.stdout.read(block_bytes)

            if len(data) != block_bytes:
                continue

            audio = np.frombuffer(data, dtype=np.int16).reshape(-1, 2)

            # Guitar is connected to input 2 of the UM2
            guitar = audio[:, 1].astype(np.float32)

           # Clean pre-gain
            guitar = process(guitar)

            # Distortion
            #guitar = soft_clip(
            #guitar,
            #gain=8.0,
            #drive=9000.0,
            #level=28000.0
            #)


            # Send the processed guitar to both headphone channels
            out = np.column_stack((guitar, guitar)).astype(np.int16)

            aplay.stdin.write(out.tobytes())
            aplay.stdin.flush()

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        print("Closing audio devices...")

        try:
            arecord.terminate()
            aplay.terminate()

            arecord.wait(timeout=2)
            aplay.wait(timeout=2)
        except Exception:
            pass

        print("Done.")