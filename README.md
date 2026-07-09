# Audio-Based Robot Awareness for Peripersonal Space

A [Telluride 2026](https://tellurideneuromorphic.org/) collaborative project combining neuromorphic robotics and audio signal processing for robot-aware situational awareness.

## Overview

This project explores how robots can develop awareness of dynamic events in their peripersonal space (the region immediately surrounding a robot) using audio processing. We leverage principles from:

- **RobNIC26**: Neuromorphic Robotics & Integrated Circuitry — applying brain-inspired approaches to robot perception and action
- **SYNC26**: Spatiotemporal Dynamics in Neural Computation — analyzing temporal patterns in audio signals to detect and classify dynamic events

## Current Focus

Audio-based detection and classification of aerial vehicles (drones) using neuromorphic signal processing techniques. The system processes real-time audio feeds to identify drone presence and characteristics.

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
