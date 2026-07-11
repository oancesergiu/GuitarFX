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

from engine.dsp.audio_format import (
    int16_to_float32,
    float32_to_int16,
)


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

    arecord_process = subprocess.Popen(
        arecord_cmd,
        stdout=subprocess.PIPE,
        bufsize=0,
    )

    aplay_process = subprocess.Popen(
        aplay_cmd,
        stdin=subprocess.PIPE,
        bufsize=0,
    )

    print(
        "Float32 DSP engine running. "
        "Press Ctrl+C to stop."
    )

    bytes_per_sample = 2
    bytes_per_frame = CHANNELS * bytes_per_sample
    block_bytes = BLOCK_FRAMES * bytes_per_frame

    try:
        while True:
            data = arecord_process.stdout.read(
                block_bytes
            )

            if len(data) != block_bytes:
                continue

            input_audio = np.frombuffer(
                data,
                dtype=np.int16,
            ).reshape(-1, CHANNELS)

            # Guitar is connected to UM2 input 2.
            guitar_int16 = input_audio[:, 1]

            # Convert once: PCM int16 -> normalized float32.
            guitar_float = int16_to_float32(
                guitar_int16
            )

            # Entire DSP rack now stays in float32.
            processed_float = process(
                guitar_float
            )

            # Convert once: normalized float32 -> PCM int16.
            output_mono = float32_to_int16(
                processed_float
            )

            output_stereo = np.column_stack(
                (output_mono, output_mono)
            ).astype(np.int16)

            aplay_process.stdin.write(
                output_stereo.tobytes()
            )

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        print("Closing audio devices...")

        for pipe in (
            arecord_process.stdout,
            aplay_process.stdin,
        ):
            try:
                pipe.close()
            except Exception:
                pass

        for child_process in (
            arecord_process,
            aplay_process,
        ):
            try:
                child_process.terminate()
                child_process.wait(timeout=2)

            except Exception:
                try:
                    child_process.kill()
                    child_process.wait(timeout=1)
                except Exception:
                    pass

        print("Done.")