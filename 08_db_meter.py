#!/usr/bin/env python
"""Dead-simple terminal dB meter for the microphone.

Usage:
    python 08_db_meter.py                 # USB PnP mic if present, else default
    python 08_db_meter.py --device 2      # pick an input device
    python 08_db_meter.py --list-devices
"""

import argparse
import sys

import numpy as np
import pyaudio

FS = 48000
CHUNK = 2400            # 50 ms -> 20 updates/s
PREFERRED_MIC = "USB PnP"
DB_MIN, DB_MAX = -70.0, 0.0   # meter range (dBFS)
BAR = 50


def pick_device(pa, requested):
    if requested is not None:
        return requested
    for i in range(pa.get_device_count()):
        d = pa.get_device_info_by_index(i)
        if d["maxInputChannels"] > 0 and PREFERRED_MIC.lower() in d["name"].lower():
            return i
    return pa.get_default_input_device_info()["index"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--list-devices", action="store_true")
    args = ap.parse_args()

    pa = pyaudio.PyAudio()
    if args.list_devices:
        for i in range(pa.get_device_count()):
            d = pa.get_device_info_by_index(i)
            if d["maxInputChannels"] > 0:
                print(f"[{i}] {d['name']}")
        pa.terminate()
        return

    dev = pick_device(pa, args.device)
    print(f"mic: [{dev}] {pa.get_device_info_by_index(dev)['name']}  — Ctrl-C to stop")
    stream = pa.open(format=pyaudio.paFloat32, channels=1, rate=FS, input=True,
                     input_device_index=dev, frames_per_buffer=CHUNK)
    peak = DB_MIN
    try:
        while True:
            x = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False),
                              dtype=np.float32)
            db = 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-10)
            peak = max(peak - 0.3, db)                     # slow-decay peak hold
            n = int(np.clip((db - DB_MIN) / (DB_MAX - DB_MIN), 0, 1) * BAR)
            p = int(np.clip((peak - DB_MIN) / (DB_MAX - DB_MIN), 0, 1) * BAR)
            bar = ("#" * n + "-" * (BAR - n))[:BAR]
            bar = bar[:p] + "|" + bar[p + 1:] if p < BAR else bar
            print(f"\r{db:7.1f} dBFS  [{bar}]  peak {peak:6.1f}  ", end="", flush=True)
    except KeyboardInterrupt:
        print()
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


if __name__ == "__main__":
    main()
