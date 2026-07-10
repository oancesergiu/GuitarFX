import subprocess
import numpy as np

from config import (
    RATE,
    CHANNELS,
    BLOCK_FRAMES,
    DEVICE,
    BUFFER_SIZE,
    PERIOD_SIZE,
)

from engine.processor import process


def run_audio():
    arecord_cmd = [
        "arecord",
        "-D", DEVICE,
        "-f", "S16_LE",
        "-r", str(RATE),
        "-c", str(CHANNELS),
        f"--buffer-size={BUFFER_SIZE}",
        f"--period-size={PERIOD_SIZE}",
    ]

    aplay_cmd = [
        "aplay",
        "-D", DEVICE,
        "-f", "S16_LE",
        "-r", str(RATE),
        "-c", str(CHANNELS),
        f"--buffer-size={BUFFER_SIZE}",
        f"--period-size={PERIOD_SIZE}",
    ]

    arecord = subprocess.Popen(
        arecord_cmd,
        stdout=subprocess.PIPE,
        bufsize=0,
    )

    aplay = subprocess.Popen(
        aplay_cmd,
        stdin=subprocess.PIPE,
        bufsize=0,
    )

    print("Python ALSA audio engine running. Press Ctrl+C to stop.")

    bytes_per_frame = CHANNELS * 2
    block_bytes = BLOCK_FRAMES * bytes_per_frame

    try:
        while True:
            data = arecord.stdout.read(block_bytes)

            if len(data) != block_bytes:
                continue

            audio = np.frombuffer(
                data,
                dtype=np.int16,
            ).reshape(-1, CHANNELS)

            # Guitar is connected to UM2 input 2
            guitar = audio[:, 1].astype(np.float32)

            guitar = process(guitar)

            output = np.column_stack(
                (guitar, guitar)
            ).astype(np.int16)

            aplay.stdin.write(output.tobytes())

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        print("Closing audio devices...")

        for pipe in (arecord.stdout, aplay.stdin):
            try:
                pipe.close()
            except Exception:
                pass

        for process_handle in (arecord, aplay):
            try:
                process_handle.terminate()
                process_handle.wait(timeout=2)
            except Exception:
                try:
                    process_handle.kill()
                except Exception:
                    pass

        print("Done.")