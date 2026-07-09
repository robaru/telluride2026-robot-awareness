# Project status — audio drone detection for the peripersonal robot demo

*2026-07-09* Roberto Barumerli

## What this is

Telluride 2026 project: give Spot (Boston Dynamics robot dog) awareness of its
peripersonal space using **one microphone**. A drone flies near the robot; the audio
pipeline detects it; Spot ducks. Everything is classic signal processing (numpy/scipy),
no ML, designed to survive a live demo in a room full of talking people with the robot's
fan running next to the mic.

**Current state**: the detector works and is validated on two recorded sessions, and the
duck reflex is implemented in the live script (`--spot` enables the detector to send the actual command).

## How the detector works


INIT: mic, 44.1 kHz → 0.1-s window every 0.05 s

Procedure:
```
  → PSD (few STFT frames averaged)
  → interpolate onto a log-frequency grid (700 Hz – 16 kHz, 96 pts/octave)
  → WHITEN: subtract 1/3-octave running median  to control for broadband level
  → subtract CALIBRATED background profile to remove fan lines & room tones
  → "excess" spectrum E(f) = narrowband energy the room didn't have
  → score = mean excess in 3–7 kHz focus on frequency band where speech can't reach
  → score must SUSTAIN THR_ON for 1 s → DRONE  transients can't (stays on until score < THR_OFF)
```

Why each piece is there:
- **Whitening** makes the detector blind to loudness — crowd rumble, mic gain,
  distance all cancel. Only *narrowband* structure (harmonics) survives.
- **Calibration** (10 s of drone-free audio, redone automatically at every mic start)
  measures the room's own narrowband fingerprint — the robot fan's lines
  (~450/880/1650 Hz) — and subtracts it. (10 s validated: any 10-s slice of the crowd
  recording lands the same threshold as 30 s.)
- **3–7 kHz scoring band**: the drone's harmonic comb is strong there; voiced speech
  harmonics die by ~3.5 kHz; above 7 kHz there is only noise floor that dilutes the
  score. Picked by sweeping candidate bands against the crowd-noise session.
- **Threshold is learned, not hand-tuned**: `THR_ON = 2 ×` the loudest score the
  calibration audio *sustains for a full second* (floor 1 dB). Calibrate while people
  talk and the threshold automatically rises to beat that crowd. `THR_OFF = 0.6 × THR_ON`.
- **1-s sustain rule**: single windows are noisy and speech/claps can spike the score,
  but only a real drone *holds* it for a second. This is the false-positive killer.
  It also sets the reaction time: ~1.2 s from takeoff to flag, by design.
- Bonus: a **synthetic harmonic-comb correlation** (log-frequency shift search) gives a
  live fundamental-frequency readout (~320–350 Hz for the big drone) without tracking
  peaks. It is *not* part of the detection decision — speech is also comb-like, so it
  can't discriminate — but it's a nice number to show on screen.

Validated results (fully automatic, calibrate → detect, nothing tuned per file):
**7/7 drone events detected, 0 false positives** over ~15 minutes across two sessions
(quiet room + fan; loud crowd right at the mic — including 192 s of pure crowd noise).
Both sessions learn nearly the same threshold (~1.4 dB).

## How to use it

```bash
conda env create -f environment.yml     # once
conda activate telluride

# demo day, in the demo room:
python 06_realtime_detector.py          # calibrates 10 s at startup (drone OFF, fan ON,
                                        # people talking is GOOD — threshold learns it),
                                        # then detects; add --spot to command the robot
python 06_realtime_detector.py --load-cal  # skip calibration, reuse drone_cal.npz —
                                           # for a restart while the drone is airborne
```

The terminal shows a live status line (score, threshold, arming progress, f0, mic
level, compute time) and prints a timestamped line on every DRONE DETECTED / clear.
It prefers the **"USB PnP Audio Device"** mic automatically; `--list-devices` and
`--device N` override.

Testing without a mic — replay any wav through the identical code path:

```bash
python 06_realtime_detector.py --wav "recordings with big drone/looming and receding.wav"
```

(Replay loads `drone_cal.npz`; `--recal --wav file.wav` calibrates from the file's
first 10 s instead, assumed drone-free.)

If it misbehaves on site:
- **False alarms** → raise `THR_ON_FACTOR` (2.0 → 2.3) or `MIN_ON_S`; recalibrate with
  the crowd talking louder.
- **Misses** → lower `THR_ON_FACTOR` toward 1.7.
- **Detection flickers during flight** → lower `THR_OFF_FACTOR`.
- All knobs sit at the top of `06_realtime_detector.py` (keep the notebook copy in sync).
- The plot to look at when something is weird: the excess spectrogram (notebook step 5).

## Files

| file | what it is |
|---|---|
| `06_realtime_detector.py` | **The live tool.** Mic in, terminal out. Same math as the notebook, plus the **duck reflex**: ducks when the drone's 8–16 kHz level rises ≥ `SLOPE_DUCK` (1 dB/s ≈ charging, time-to-contact < 9 s) or sits ≥ `CLOSE_DB` (5 dB) above its detection-onset level; armed 3 s after detection (so the takeoff ramp can't fake a charge); stays down while the drone hovers close; stands up when it leaves; re-arms per approach. Prints `>>> DUCK <<<` by default, `--spot` sends take/duck/release to Spot (in a thread, failure-safe). `--no-duck` disables. |
| `05_psd_profile_drone_detector.ipynb` | **The documentation.** Step-by-step derivation of the detector with a plot per step, threshold learning, causal replay, and validation on both sessions. Read this to understand the design; re-run top-to-bottom (~3 min) after changing anything. |
| `07_looming_distance_velocity.ipynb` | **Distance/velocity/time-to-contact** from the looming recordings: relative range from the 8–16 kHz level (−6 dB per distance doubling), approach/recede state from its slope, calibration-free τ = 8.686/slope. Includes a causal `LoomingTracker` class ready to wire into the live script, and a suggested duck rule (`detected AND (r < 1.5 m OR τ < 5 s)`). |
| `drone_cal.npz` | Saved calibration (background profile + learned threshold). Refreshed automatically at every mic start; only `--load-cal` runs reuse it. |
| `spot_command.py` | Send move commands to Spot over rosbridge (forward/backward/left/right/stop). |
| `spot_duck.py` | Standalone test of the duck sequence: take Spot's lease, call `/D02/spot/duck`, release. The same sequence is built into the detector's `--spot` mode. |
| `External Team → Spot Command Setup Guide-1.pdf` | Spot-side setup from the robotics team. |
| `recordings with big drone/` | All test audio (44.1 kHz mono wav + Audacity projects). |
| `old/` | First-attempt archive (Wiener/matched-filter approach, crazyflie analysis, m4a clips). Superseded — kept for reference only. |

Recordings (two sessions, same big drone):

| file | content |
|---|---|
| `static.wav` | session 1 (quiet + robot fan): background 0–51 s, drone flying 51.5–81 s |
| `static2.wav` | session 1: drone hovering 0–4.8 s, then background |
| `background noise .wav` | session 2 (loud crowd): **no drone anywhere** — false-positive test |
| `background noise plus drone static.wav` | session 2: crowd, drone hovering ~143–167 s |
| `looming and receding.wav` | session 2: drone approaches/recedes ~5–30 s and ~151–178 s |
| `looming and receding-2.wav` | session 2: same, drone ~74–100 s |

## Spot side (untested end-to-end)

Network: connect to **BrainAirWaves** wifi, Spot's rosbridge at `192.168.167.163:9090`
(`ping` should answer in ~50 ms; `pip install roslibpy`). `spot_duck.py` shows the full
duck sequence: take lease → `/D02/spot/duck` → release.

**Wired in:** with `--spot` the script opens **one persistent rosbridge connection at
startup** (right after calibration — fails fast with a clear message if Spot is
unreachable) and closes it on Ctrl-C; each duck is then just the `/D02/spot/duck`
service call, fired from a daemon thread so the audio loop never blocks. Without
`--spot` a big red banner at startup says Spot is disabled and ducks are print-only. Verified on all six recordings in replay: one duck per
approach, anticipatory (fires ~2–3 s before closest approach on the looming cycles),
zero ducks on 192 s of pure crowd. **Remaining: run it once against the real robot** —
in particular check whether Spot auto-recovers from the duck or needs an explicit
stand command on the "threat passed" event (the hook exists in `run_stream`).

## Known limitations

- **What would fool it:** another rotor-like machine, or a loudspeaker playing sustained
  tones in 3–7 kHz. Both would show up in the excess spectrogram.
- **A drone approaching very slowly from far away** arms late: the score must sustain
  the learned threshold for 1 s. For the demo this is the right trade-off (it fires as
  the drone gets close, which is when Spot should duck).
- **Latency is ~1.2 s by design** (the 1-s sustain rule). Lower `MIN_ON_S` for a
  twitchier demo at the cost of false-positive protection.
- **Calibration is per-room and per-fan-state** — which is why it now runs automatically
  (10 s) at every mic start. The one case to avoid: don't start *without* `--load-cal`
  while the drone is already flying, or the drone gets baked into the background profile.
- The mic runs at 44.1 kHz through CoreAudio conversion (device is 48 kHz native). If
  `pa.open` ever refuses, set `FS = 48000` at the top of the script and recalibrate.
- Distance/looming estimation is **analyzed** in `07_looming_distance_velocity.ipynb`
  (relative range ±40%, velocity ~0.3–0.5 m/s, time-to-contact). The live script's duck
  reflex uses the simple version of that result (level slope + proximity); the full
  distance readout is notebook-only. Note: the *detector score* cannot do distance
  (whitening makes it loudness-blind by design); the range signal is the raw 8–16 kHz
  band level. Doppler is infeasible (rotor RPM wobble is ~15× larger).
