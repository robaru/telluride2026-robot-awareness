# Audio-Based Robot Awareness for Peripersonal Space

A [Telluride 2026](https://tellurideneuromorphic.org/) collaborative project combining robotics and audio signal processing for robot-aware situational awareness.

Audio-based detection of an drone using simple signal processing techniques. The system processes real-time audio feeds to identify drone presence and direction of movement. If that happens, Spot (Boston Dynamic robot dog) is instructed to duck as to simulate collision avoidance.

## Files

- `status.md` — **Start here**: what works, how it works, how to run it
- `06_realtime_detector.py` — Real-time audio detection pipeline (this is for the DEMO! :))
- `05_psd_profile_drone_detector.ipynb` — PSD analysis and detector development
- `07_looming_distance_velocity.ipynb` — analysis for looming/receding drone movement
- `spot_command.py` — Integration with Spot robot platform
- `environment.yml` — Python environment configuration
- `drone_cal.npz` — Calibration data for drone detection

## Setup

```bash
conda env create -f environment.yml
conda activate telluride
```

## Running the detector

Plug USB-C PnP microphone to computer and run:

```bash
python 06_realtime_detector.py           # 10-s calibration at every start (drone OFF,
                                         # people talking is good — the detection
                                         # threshold is learned from it), then detects
python 06_realtime_detector.py --spot    # enable script to send commands to spot
```

(the script is configured to listen to PnP mic, if not available, then the script selects the system default).

## References

- [Telluride 2026 Workshop](https://tellurideneuromorphic.org/)
- [RobNIC26 Topic Area](https://sites.google.com/view/telluride-2026/topic-areas)
- [SYNC26 Topic Area](https://sites.google.com/view/telluride-2026/topic-areas)
