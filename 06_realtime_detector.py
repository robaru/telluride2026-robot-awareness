#!/usr/bin/env python
"""Real-time drone detector — terminal only, no GUI.

Pipeline identical to 05_psd_profile_drone_detector.ipynb:
    0.1 s window every 0.05 s -> PSD -> log-f grid -> whiten (broadband gone)
    -> subtract calibrated background (fan lines gone) -> excess E(f)
    -> score = mean excess in 3-7 kHz (speech dies by ~3.5 kHz, above 7 kHz only noise)
    -> must sustain THR_ON for MIN_ON_S -> DRONE (release below THR_OFF)

The thresholds are LEARNED during calibration: THR_ON = 2x the loudest score the
background sustains for 1 s. So calibrate with the room sounding like the demo —
people talking loudly is good, it raises the threshold to match.

DUCK REFLEX (wow over precision): while a drone is detected, duck when its 8-16 kHz
level RISES faster than SLOPE_DUCK dB/s (= it is charging; time-to-contact < ~9 s)
or sits CLOSE_DB above its level at detection (= it is near). Prints ">>> DUCK <<<"
by default; --spot actually sends take/duck/release to Spot over rosbridge.
Fly assertively at the robot — a faster approach triggers earlier and looks better.

Usage:
    python 06_realtime_detector.py                    # mic; 10-s calibration at startup
                                                      #   (keep the drone OFF for those 10 s)
    python 06_realtime_detector.py --load-cal         # skip calibration, reuse drone_cal.npz
                                                      #   (mid-demo restart, drone airborne)
    python 06_realtime_detector.py --spot             # actually send the duck to Spot
    python 06_realtime_detector.py --cal-secs 30      # longer calibration
    python 06_realtime_detector.py --device 2         # pick an input device
    python 06_realtime_detector.py --list-devices
    python 06_realtime_detector.py --wav "recordings with big drone/static.wav"
                                                      # replay a file through the same code

On the mic the calibration (background/fan fingerprint + learned threshold) is
redone at every start — demo rooms change — and saved to drone_cal.npz;
--load-cal reuses the saved one instead. Replay (--wav) loads the saved file,
or calibrates from the wav's own start with --recal.

2026-07-09 Roberto Barumerli
"""

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import scipy.signal as sig

# ---------- the knobs (keep in sync with the notebook) ----------
FS         = 48000
WIN_S      = 0.1              # analysis window (s)
HOP_S      = 0.05             # hop between decisions (s)
NPERSEG    = 2048             # STFT segment
FMIN, FMAX = 700.0, 16000.0   # analysis band: above the fan/low-frequency mess
PPO        = 96               # points per octave of the log-frequency grid
WHITEN_OCT = 1 / 3            # width of the whitening median filter (octaves)
SCORE_FMIN = 3000.0           # scoring band: drone comb strong, speech harmonics dead
SCORE_FMAX = 7000.0           #   ... and above this only noise floor (dilutes the score)
SMOOTH_N   = 5                # median smoothing of the score (~0.25 s)
MIN_ON_S   = 1.0              # score must sustain THR_ON this long before flagging
THR_ON_FACTOR  = 2.0          # THR_ON = factor x loudest 1-s-sustained calibration score
THR_ON_MIN     = 1.0          # ... but never below this (dB)
THR_OFF_FACTOR = 0.6          # THR_OFF = factor x THR_ON (release)

# ---------- duck reflex (wow over precision: react to "getting louder fast") ----------
LOOM_LO, LOOM_HI = 8000.0, 16000.0  # level band for looming: prop wash, crowd can't reach
SLOPE_DUCK = 1.0              # duck when level rises >= this (dB/s) ~ time-to-contact < 9 s
CLOSE_DB   = 5.0              # ...or when >= this much louder than at detection (proximity)
ARM_S      = 3.0              # wait this long after detection before ducking (takeoff ramp)
DUCK_HOLD_S = 5.0             # stay ducked at least this long, re-arm once level stops rising

LOGF = FMIN * 2 ** (np.arange(int(np.log2(FMAX / FMIN) * PPO)) / PPO)
MED_BINS = int(PPO * WHITEN_OCT) // 2 * 2 + 1
SCORE_BAND = (LOGF > SCORE_FMIN) & (LOGF < SCORE_FMAX)
LOOM_BAND = (LOGF > LOOM_LO) & (LOGF < LOOM_HI)
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

    def __init__(self, w_bg, thr_on):
        self.w_bg = w_bg
        self.thr_on = thr_on
        self.thr_off = THR_OFF_FACTOR * thr_on
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
        Li = robust_log_psd(self.buf)
        E = np.maximum(whiten(Li) - self.w_bg, 0)
        self.recent.append(E[SCORE_BAND].mean())
        s = float(np.median(self.recent))
        if not self.on:                     # arm only after MIN_ON_S of sustained score
            self.streak = self.streak + 1 if s >= self.thr_on else 0
            self.on = self.streak >= MIN_ON_N
        else:                               # release quickly once the drone is gone
            self.on = s >= self.thr_off
            if not self.on:
                self.streak = 0
        En = E - E.mean()
        corr = TPL_BANK @ En / (np.linalg.norm(En) + 1e-12)
        return dict(score=s, on=self.on, streak=self.streak,
                    f0=float(F0_AXIS[np.argmax(corr)]), f0_corr=float(corr.max()),
                    loom_level=float(Li[LOOM_BAND].mean()),
                    level_db=20 * np.log10(np.sqrt(np.mean(chunk ** 2)) + 1e-9))


class DuckReflex:
    """Wow-over-precision duck trigger. Feed (loom_level_dB, detected) per hop.

    Ducks when the drone is *getting louder fast* (level slope >= SLOPE_DUCK dB/s,
    i.e. closing with time-to-contact < ~8.7/SLOPE_DUCK s) or is simply much louder
    than when first detected (CLOSE_DB). Armed ARM_S after detection so the takeoff
    loudness ramp can't fake a charge. After a duck it holds DUCK_HOLD_S, then
    re-arms once the drone stops getting louder — repeated passes each get a duck.
    """

    def __init__(self, on_duck=None):
        self.on_duck = on_duck or (lambda: None)
        self.win = deque(maxlen=int(round(3.0 / HOP_S)))   # 3-s slope window
        self.since_on = 0.0
        self.ref = None
        self.ducked_at = None

    def feed(self, loom_level, detected):
        """Returns dict(slope, rel, ducked) while detected, else None."""
        if not detected:
            self.win.clear()
            self.since_on, self.ref, self.ducked_at = 0.0, None, None
            return None
        self.since_on += HOP_S
        self.win.append(loom_level)
        n = len(self.win)
        if self.ref is None and self.since_on >= 2.0:
            self.ref = float(np.mean(self.win))            # level at detection ~ takeoff
        # slope: mean of newest second minus mean of oldest second of the window
        k = int(round(1.0 / HOP_S))
        slope = ((np.mean(list(self.win)[-k:]) - np.mean(list(self.win)[:k]))
                 / ((n - k) * HOP_S)) if n > 2 * k else 0.0
        rel = (loom_level - self.ref) if self.ref is not None else 0.0
        if self.ducked_at is not None:                     # currently ducked
            self.ducked_at += HOP_S
            # stand up only when the drone has actually left: well below the
            # "close" level (3 dB hysteresis) and no longer climbing —
            # hovering overhead keeps the robot down
            if (self.ducked_at >= DUCK_HOLD_S and slope < 0.2
                    and rel < CLOSE_DB - 3.0):
                self.ducked_at = None
        elif (self.since_on >= ARM_S and slope > -0.2       # never duck at a leaver
              and (slope >= SLOPE_DUCK or rel >= CLOSE_DB)):
            self.ducked_at = 0.0
            self.on_duck()
        return dict(slope=float(slope), rel=float(rel), ducked=self.ducked_at is not None)


class SpotClient:
    """One persistent rosbridge connection: opened at startup (after calibration),
    closed on Ctrl-C. duck() only makes the Service call."""

    def __init__(self, host="192.168.167.163", port=9090):
        import roslibpy
        self._roslibpy = roslibpy
        print(f"connecting to Spot at {host}:{port} ...")
        self.client = roslibpy.Ros(host=host, port=port)
        self.client.run(timeout=10)
        if not self.client.is_connected:
            raise SystemExit(f"could not connect to Spot at {host}:{port} — "
                             "check the BrainAirWaves wifi / rosbridge")
        print("connected to Spot")

    def duck(self):
        import threading

        def _go():
            try:
                self._roslibpy.Service(self.client, "/D02/spot/duck",
                                       "std_srvs/Trigger").call(
                    self._roslibpy.ServiceRequest())
            except Exception as e:                          # demo must not die with Spot
                print(f"\n[spot] duck failed: {e}")

        threading.Thread(target=_go, daemon=True).start()   # audio loop never blocks

    def close(self):
        try:
            self.client.terminate()
            print("spot connection closed")
        except Exception:
            pass


def calibrate_from_audio(x):
    """(background profile, THR_ON) from drone-free audio — keep the drone OFF,
    but let the room sound like the demo: the threshold learns from it."""
    n, h = int(WIN_S * FS), int(0.25 * FS)
    W = [whiten(robust_log_psd(x[i:i + n])) for i in range(0, len(x) - n, h)]
    w_bg = np.median(W, axis=0)
    # score the calibration audio against its own profile, exactly like runtime
    hop = int(HOP_S * FS)
    s = [np.maximum(whiten(robust_log_psd(x[i:i + n])) - w_bg, 0)[SCORE_BAND].mean()
         for i in range(0, len(x) - n, hop)]
    s = sig.medfilt(s, SMOOTH_N)
    sustained = np.array([min(s[i:i + MIN_ON_N]) for i in range(len(s) - MIN_ON_N)])
    thr_on = max(THR_ON_FACTOR * sustained.max(), THR_ON_MIN)
    return w_bg, thr_on


# ---------- terminal output ----------

USE_COLOR = sys.stdout.isatty()


def paint(txt, code):
    return f"\033[{code}m{txt}\033[0m" if USE_COLOR else txt


def status_line(m, det, r):
    bar_n = int(np.clip(m["score"] / (2 * det.thr_on), 0, 1) * 20)
    bar = "#" * bar_n + "-" * (20 - bar_n)
    if m["on"]:
        state = paint("* DUCK! ", "1;33") if (r and r["ducked"]) else paint("* DRONE ", "1;31")
        extra = (f"f0 {m['f0']:4.0f} Hz  {r['slope']:+4.1f} dB/s" if r
                 else f"f0 {m['f0']:4.0f} Hz")
    else:
        state = paint("  ---   ", "2")
        extra = (f"arming {m['streak']:2d}/{MIN_ON_N}" if m["streak"]
                 else " " * 12)
    return (f"{state} score {m['score']:5.2f} dB |{bar}| thr {det.thr_on:.2f} "
            f"{extra:<22s} mic {m['level_db']:6.1f} dBFS  {m['ms']:4.1f} ms/hop")


def run_stream(chunks, det, reflex=None, realtime_note=""):
    """Common loop: consume HOP-sized float chunks, print status + events."""
    t_stream, n_det, t_on, last_draw = 0.0, 0, None, 0.0
    drone_total, n_duck, was_ducked = 0.0, 0, False
    print(f"listening{realtime_note} — Ctrl-C to stop. "
          f"THR {det.thr_on:.2f}/{det.thr_off:.2f} dB (learned from calibration), "
          f"sustain {MIN_ON_S:.1f} s"
          + (f" | duck: >{SLOPE_DUCK} dB/s or +{CLOSE_DB} dB" if reflex else "") + "\n")

    def event(txt, color):
        print("\r\033[K" if USE_COLOR else "\r" + " " * 100 + "\r", end="")
        print(paint(txt, color))

    try:
        for chunk in chunks:
            tic = time.perf_counter()
            m = det.feed(chunk)
            t_stream += len(chunk) / FS
            if m is None:
                continue
            m["ms"] = (time.perf_counter() - tic) * 1e3
            r = reflex.feed(m["loom_level"], m["on"]) if reflex else None
            if m["on"] and t_on is None:
                t_on, n_det = t_stream, n_det + 1
                event(f"[{t_stream:7.2f} s] DRONE DETECTED   "
                      f"(score {m['score']:.2f} dB, f0 ~{m['f0']:.0f} Hz)", "1;31")
            elif not m["on"] and t_on is not None:
                drone_total += t_stream - t_on
                event(f"[{t_stream:7.2f} s] clear             "
                      f"(was on {t_stream - t_on:.1f} s)", "1;32")
                t_on = None
            ducked = bool(r and r["ducked"])
            if ducked and not was_ducked:
                n_duck += 1
                event(f"[{t_stream:7.2f} s] >>> DUCK <<<      "
                      f"(closing {r['slope']:+.1f} dB/s, {r['rel']:+.1f} dB vs takeoff)",
                      "1;33")
            elif was_ducked and not ducked:
                event(f"[{t_stream:7.2f} s] threat passed — standing back up", "1;32")
            was_ducked = ducked
            if t_stream - last_draw >= 0.1:                 # redraw at 10 Hz
                print("\r" + status_line(m, det, r) + " ", end="", flush=True)
                last_draw = t_stream
    except KeyboardInterrupt:
        pass
    if t_on is not None:
        drone_total += t_stream - t_on
    print(f"\n\nsession: {t_stream:.1f} s audio, {n_det} detection(s), "
          f"{drone_total:.1f} s of drone, {n_duck} duck(s)")


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


def save_cal(w_bg, thr_on):
    np.savez(CAL_FILE, w_bg=w_bg, thr_on=thr_on)
    print(f"calibration saved to {CAL_FILE.name} (THR_ON = {thr_on:.2f} dB)")


def load_cal():
    if not CAL_FILE.exists():
        raise SystemExit(f"no {CAL_FILE.name} to load — run once without --load-cal")
    cal = np.load(CAL_FILE)
    if "thr_on" not in cal or len(cal["w_bg"]) != len(LOGF):
        raise SystemExit(f"{CAL_FILE.name} is from an older version or different "
                         "parameters — run without --load-cal to remake it")
    age_h = (time.time() - CAL_FILE.stat().st_mtime) / 3600
    note = paint(f"  <- {age_h:.1f} h old, from a possibly different room!", "1;33") \
        if age_h > 2 else f" ({age_h * 60:.0f} min old)"
    print(f"loaded calibration from {CAL_FILE.name} "
          f"(THR_ON = {float(cal['thr_on']):.2f} dB){note}")
    return cal["w_bg"], float(cal["thr_on"])


def get_calibration(args):
    if args.wav:
        # replay: reuse the saved calibration unless --recal (then take the file's
        # own start, assumed drone-free)
        if CAL_FILE.exists() and not args.recal:
            return load_cal()
        import soundfile as sf
        x, _ = sf.read(args.wav)
        if x.ndim > 1:
            x = x.mean(axis=1)
        n = min(len(x), int(args.cal_secs * FS))
        print(f"calibrating from the first {n / FS:.0f} s of the file (assumed drone-free)")
        w_bg, thr_on = calibrate_from_audio(x[:n])
        save_cal(w_bg, thr_on)
        return w_bg, thr_on
    if args.load_cal:
        # fast restart (e.g. mid-demo, drone already flying — do NOT recalibrate then)
        return load_cal()
    # default on the mic: fresh calibration every start — demo rooms change
    print(paint(f"CALIBRATING for {args.cal_secs:.0f} s — keep the drone OFF, but let "
                "the room sound like the demo (talking is GOOD, it sets the threshold)",
                "1;33"))
    buf = []
    for chunk in mic_chunks(args.device):
        buf.append(chunk)
        done = len(buf) * HOP_S
        print(f"\r  {done:5.1f} / {args.cal_secs:.0f} s", end="", flush=True)
        if done >= args.cal_secs:
            break
    print()
    w_bg, thr_on = calibrate_from_audio(np.concatenate(buf))
    save_cal(w_bg, thr_on)
    return w_bg, thr_on


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wav", help="replay a wav file instead of the microphone")
    ap.add_argument("--realtime", action="store_true",
                    help="pace --wav replay at real speed")
    ap.add_argument("--device", type=int, default=None, help="input device index")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--load-cal", action="store_true",
                    help="skip the startup calibration and reuse drone_cal.npz "
                         "(for a fast restart while the drone is already flying)")
    ap.add_argument("--recal", action="store_true",
                    help="with --wav: recalibrate from the file start instead of "
                         "loading drone_cal.npz")
    ap.add_argument("--cal-secs", type=float, default=10.0,
                    help="calibration length in seconds (default 10)")
    ap.add_argument("--spot", action="store_true",
                    help="actually send the duck command to Spot (default: print only)")
    ap.add_argument("--no-duck", action="store_true", help="disable the duck reflex")
    args = ap.parse_args()

    if args.list_devices:
        list_devices()
        return

    w_bg, thr_on = get_calibration(args)
    det = LiveDroneDetector(w_bg, thr_on)
    spot = SpotClient() if args.spot else None                # connect after calibration
    reflex = None if args.no_duck else DuckReflex(spot.duck if spot else None)
    if reflex and not spot:
        print(paint("\n" + "!" * 66, "1;31"))
        print(paint("!!  SPOT IS DISABLED — dry run, ducks are only printed        !!", "1;31"))
        print(paint("!!  add --spot to actually send the duck command to the robot !!", "1;31"))
        print(paint("!" * 66 + "\n", "1;31"))
    elif reflex:
        print(paint("SPOT ARMED — duck commands WILL be sent to the robot\n", "1;32"))
    try:
        if args.wav:
            run_stream(wav_chunks(args.wav, args.realtime), det, reflex,
                       realtime_note=f" to {Path(args.wav).name}")
        else:
            run_stream(mic_chunks(args.device), det, reflex)
    finally:
        if spot:                                            # clean shutdown on Ctrl-C too
            spot.close()


if __name__ == "__main__":
    main()
