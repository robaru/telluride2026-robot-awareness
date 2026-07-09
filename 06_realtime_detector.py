#!/usr/bin/env python
"""Real-time drone detector — terminal only, no GUI.

Pipeline identical to 05_psd_profile_drone_detector.ipynb:
    0.1 s window every 0.05 s -> PSD -> log-f grid -> whiten (broadband gone)
    -> subtract calibrated background (fan lines gone) -> excess E(f)
    -> score = mean excess above 2.5 kHz
    -> must sustain THR_ON for MIN_ON_S -> DRONE (release below THR_OFF)

Usage:
    python 06_realtime_detector.py                    # mic; calibrates on first run
    python 06_realtime_detector.py --recal            # force a new calibration
    python 06_realtime_detector.py --cal-secs 30      # calibration length
    python 06_realtime_detector.py --device 2         # pick an input device
    python 06_realtime_detector.py --list-devices
    python 06_realtime_detector.py --wav "recordings with big drone/recordings-01.wav"
                                                      # replay a file through the same code

Calibration (the background/fan fingerprint) is saved to drone_cal.npz next to this
script and reloaded on the next start. Recalibrate whenever the room or fan changes.
Keep the drone OFF while calibrating.
"""

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import scipy.signal as sig

# ---------- the knobs (keep in sync with the notebook) ----------
FS         = 44100
WIN_S      = 0.1              # analysis window (s)
HOP_S      = 0.05             # hop between decisions (s)
NPERSEG    = 2048             # STFT segment
FMIN, FMAX = 700.0, 16000.0   # analysis band: above the fan/low-frequency mess
PPO        = 96               # points per octave of the log-frequency grid
WHITEN_OCT = 1 / 3            # width of the whitening median filter (octaves)
SCORE_FMIN = 2500.0           # scoring band floor
SMOOTH_N   = 5                # median smoothing of the score (~0.25 s)
THR_ON, THR_OFF = 0.95, 0.80  # detection hysteresis (dB of mean excess)
MIN_ON_S   = 1.0              # score must sustain THR_ON this long before flagging

LOGF = FMIN * 2 ** (np.arange(int(np.log2(FMAX / FMIN) * PPO)) / PPO)
MED_BINS = int(PPO * WHITEN_OCT) // 2 * 2 + 1
SCORE_BAND = LOGF > SCORE_FMIN
MIN_ON_N = int(round(MIN_ON_S / HOP_S))
HOP = int(HOP_S * FS)

CAL_FILE = Path(__file__).resolve().parent / "drone_cal.npz"


# ---------- pipeline (same math as the notebook) ----------

def robust_log_psd(win):
    """Log-PSD on the log-f grid, averaged across the window's few STFT frames."""
    f, _, Z = sig.stft(win, FS, nperseg=NPERSEG, noverlap=NPERSEG // 2)
    P = np.mean(np.abs(Z) ** 2, axis=1)
    return np.interp(LOGF, f, 10 * np.log10(P + 1e-18))


def whiten(L):
    """Keep only narrowband structure: dB above the local (1/3-octave median) floor."""
    return L - sig.medfilt(L, MED_BINS)


def make_comb_template():
    """Synthetic comb (harmonics of 350 Hz + RPM wobble) for the f0 readout."""
    rng = np.random.default_rng(0)
    tt = np.arange(int(8.0 * FS)) / FS
    wob = sig.sosfiltfilt(sig.butter(2, 2.0, fs=FS, output="sos"),
                          rng.standard_normal(len(tt)))
    phase = 2 * np.pi * np.cumsum(350.0 * (1 + 0.02 * wob / wob.std())) / FS
    synth = sum(np.sin(k * phase + rng.uniform(0, 2 * np.pi)) for k in range(2, 30))
    T = np.maximum(whiten(robust_log_psd(synth)), 0)
    T -= T.mean()
    T /= np.linalg.norm(T)
    shifts = np.arange(-int(PPO * np.log2(350 / 200)), int(PPO * np.log2(550 / 350)) + 1)
    bank = []
    for s in shifts:
        Ts = np.roll(T, s)
        if s > 0:
            Ts[:s] = 0
        elif s < 0:
            Ts[s:] = 0
        bank.append(Ts)
    return np.array(bank), 350.0 * 2 ** (shifts / PPO)


TPL_BANK, F0_AXIS = make_comb_template()


class LiveDroneDetector:
    """Feed HOP-sized chunks; returns a metrics dict once the buffer is full."""

    def __init__(self, w_bg):
        self.w_bg = w_bg
        self.buf = np.zeros(int(WIN_S * FS))
        self.fill = 0
        self.recent = deque(maxlen=SMOOTH_N)
        self.streak = 0
        self.on = False

    def feed(self, chunk):
        k = len(chunk)
        self.buf = np.roll(self.buf, -k)
        self.buf[-k:] = chunk
        self.fill += k
        if self.fill < len(self.buf):
            return None
        E = np.maximum(whiten(robust_log_psd(self.buf)) - self.w_bg, 0)
        self.recent.append(E[SCORE_BAND].mean())
        s = float(np.median(self.recent))
        if not self.on:                     # arm only after MIN_ON_S of sustained score
            self.streak = self.streak + 1 if s >= THR_ON else 0
            self.on = self.streak >= MIN_ON_N
        else:                               # release quickly once the drone is gone
            self.on = s >= THR_OFF
            if not self.on:
                self.streak = 0
        En = E - E.mean()
        corr = TPL_BANK @ En / (np.linalg.norm(En) + 1e-12)
        return dict(score=s, on=self.on, streak=self.streak,
                    f0=float(F0_AXIS[np.argmax(corr)]), f0_corr=float(corr.max()),
                    level_db=20 * np.log10(np.sqrt(np.mean(chunk ** 2)) + 1e-9))


def calibrate_from_audio(x):
    """Background profile = median whitened spectrum over drone-free audio."""
    n, h = int(WIN_S * FS), int(0.25 * FS)
    W = [whiten(robust_log_psd(x[i:i + n])) for i in range(0, len(x) - n, h)]
    return np.median(W, axis=0)


# ---------- terminal output ----------

USE_COLOR = sys.stdout.isatty()


def paint(txt, code):
    return f"\033[{code}m{txt}\033[0m" if USE_COLOR else txt


def status_line(m):
    bar_n = int(np.clip(m["score"] / 2.0, 0, 1) * 20)
    bar = "#" * bar_n + "-" * (20 - bar_n)
    if m["on"]:
        state = paint("* DRONE ", "1;31")
        extra = f"f0 {m['f0']:4.0f} Hz (corr {m['f0_corr']:+.2f})"
    else:
        state = paint("  ---   ", "2")
        extra = (f"arming {m['streak']:2d}/{MIN_ON_N}" if m["streak"]
                 else " " * 12)
    return (f"{state} score {m['score']:5.2f} dB |{bar}| thr {THR_ON:.2f} "
            f"{extra:<22s} mic {m['level_db']:6.1f} dBFS  {m['ms']:4.1f} ms/hop")


def run_stream(chunks, det, realtime_note=""):
    """Common loop: consume HOP-sized float chunks, print status + events."""
    t_stream, n_det, t_on, last_draw = 0.0, 0, None, 0.0
    drone_total = 0.0
    print(f"listening{realtime_note} — Ctrl-C to stop. "
          f"THR {THR_ON}/{THR_OFF} dB, sustain {MIN_ON_S:.1f} s\n")
    try:
        for chunk in chunks:
            tic = time.perf_counter()
            m = det.feed(chunk)
            t_stream += len(chunk) / FS
            if m is None:
                continue
            m["ms"] = (time.perf_counter() - tic) * 1e3
            if m["on"] and t_on is None:
                t_on, n_det = t_stream, n_det + 1
                print("\r\033[K" if USE_COLOR else "\r" + " " * 100 + "\r", end="")
                print(paint(f"[{t_stream:7.2f} s] DRONE DETECTED   "
                            f"(score {m['score']:.2f} dB, f0 ~{m['f0']:.0f} Hz)", "1;31"))
            elif not m["on"] and t_on is not None:
                drone_total += t_stream - t_on
                print("\r\033[K" if USE_COLOR else "\r" + " " * 100 + "\r", end="")
                print(paint(f"[{t_stream:7.2f} s] clear             "
                            f"(was on {t_stream - t_on:.1f} s)", "1;32"))
                t_on = None
            if t_stream - last_draw >= 0.1:                 # redraw at 10 Hz
                print("\r" + status_line(m) + " ", end="", flush=True)
                last_draw = t_stream
    except KeyboardInterrupt:
        pass
    if t_on is not None:
        drone_total += t_stream - t_on
    print(f"\n\nsession: {t_stream:.1f} s audio, {n_det} detection(s), "
          f"{drone_total:.1f} s of drone")


# ---------- audio sources ----------

PREFERRED_MIC = "USB PnP"   # substring match; falls back to system default


def pick_device(pa, requested):
    """Explicit --device wins; otherwise prefer the USB PnP mic, then the default."""
    if requested is not None:
        return requested
    for i in range(pa.get_device_count()):
        d = pa.get_device_info_by_index(i)
        if d["maxInputChannels"] > 0 and PREFERRED_MIC.lower() in d["name"].lower():
            print(f"using mic [{i}] {d['name']}")
            return i
    d = pa.get_default_input_device_info()
    print(f"'{PREFERRED_MIC}' not found — using default mic [{d['index']}] {d['name']}")
    return d["index"]


def mic_chunks(device):
    import pyaudio
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paFloat32, channels=1, rate=FS, input=True,
                     input_device_index=pick_device(pa, device),
                     frames_per_buffer=HOP)
    try:
        while True:
            data = stream.read(HOP, exception_on_overflow=False)
            yield np.frombuffer(data, dtype=np.float32).astype(np.float64)
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


def wav_chunks(path, realtime=False):
    import soundfile as sf
    x, fs = sf.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if fs != FS:
        raise SystemExit(f"{path} is {fs} Hz, expected {FS} Hz")
    for i in range(0, len(x) - HOP, HOP):
        if realtime:
            time.sleep(HOP_S)
        yield x[i:i + HOP]


def list_devices():
    import pyaudio
    pa = pyaudio.PyAudio()
    print("input devices:")
    for i in range(pa.get_device_count()):
        d = pa.get_device_info_by_index(i)
        if d["maxInputChannels"] > 0:
            mark = " (default)" if i == pa.get_default_input_device_info()["index"] else ""
            if PREFERRED_MIC.lower() in d["name"].lower():
                mark += " <- preferred"
            print(f"  [{i}] {d['name']}  ({d['maxInputChannels']} ch, "
                  f"{d['defaultSampleRate']:.0f} Hz){mark}")
    pa.terminate()


def get_calibration(args):
    if args.wav and not CAL_FILE.exists():
        # replay convenience: calibrate from the file's own start (assumed drone-free)
        import soundfile as sf
        x, _ = sf.read(args.wav)
        if x.ndim > 1:
            x = x.mean(axis=1)
        n = min(len(x), int(args.cal_secs * FS))
        print(f"no {CAL_FILE.name}; calibrating from the first {n / FS:.0f} s of the file "
              "(assumed drone-free)")
        return calibrate_from_audio(x[:n])
    if CAL_FILE.exists() and not args.recal:
        w_bg = np.load(CAL_FILE)["w_bg"]
        if len(w_bg) != len(LOGF):
            raise SystemExit(f"{CAL_FILE.name} was made with different parameters — "
                             "run with --recal")
        print(f"loaded calibration from {CAL_FILE.name} (use --recal to redo it)")
        return w_bg
    # record a fresh calibration from the mic
    print(paint(f"CALIBRATING for {args.cal_secs:.0f} s — keep the drone OFF "
                "(fan/room noise is fine)...", "1;33"))
    buf = []
    for chunk in mic_chunks(args.device):
        buf.append(chunk)
        done = len(buf) * HOP_S
        print(f"\r  {done:5.1f} / {args.cal_secs:.0f} s", end="", flush=True)
        if done >= args.cal_secs:
            break
    print()
    w_bg = calibrate_from_audio(np.concatenate(buf))
    np.savez(CAL_FILE, w_bg=w_bg)
    print(f"calibration saved to {CAL_FILE.name}")
    return w_bg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wav", help="replay a wav file instead of the microphone")
    ap.add_argument("--realtime", action="store_true",
                    help="pace --wav replay at real speed")
    ap.add_argument("--device", type=int, default=None, help="input device index")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--recal", action="store_true", help="force a new calibration")
    ap.add_argument("--cal-secs", type=float, default=20.0,
                    help="calibration length in seconds (default 20)")
    args = ap.parse_args()

    if args.list_devices:
        list_devices()
        return

    w_bg = get_calibration(args)
    det = LiveDroneDetector(w_bg)
    if args.wav:
        run_stream(wav_chunks(args.wav, args.realtime), det,
                   realtime_note=f" to {Path(args.wav).name}")
    else:
        run_stream(mic_chunks(args.device), det)


if __name__ == "__main__":
    main()
