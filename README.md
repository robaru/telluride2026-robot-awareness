# Audio-Based Robot Awareness for Peripersonal Space

A [Telluride 2026](https://tellurideneuromorphic.org/) collaborative project combining robotics and audio signal processing for robot-aware situational awareness.

## Current Focus

Audio-based detection of an drone using simple signal processing techniques. The system processes real-time audio feeds to identify drone presence with the plan to have Spot (Boston Dynamic robot dog) to duck as to simulate collision avoidance.

## Key Files

- `06_realtime_detector.py` — Real-time audio detection pipeline
- `05_psd_profile_drone_detector.ipynb` — PSD analysis and detector development
- `spot_command.py` — Integration with Spot robot platform
- `environment.yml` — Python environment configuration
- `drone_cal.npz` — Calibration data for drone detection

## Setup

```bash
conda env create -f environment.yml
conda activate telluride-audio-robot
```

## References

- [Telluride 2026 Workshop](https://tellurideneuromorphic.org/)
- [RobNIC26 Topic Area](https://sites.google.com/view/telluride-2026/topic-areas)
- [SYNC26 Topic Area](https://sites.google.com/view/telluride-2026/topic-areas)
