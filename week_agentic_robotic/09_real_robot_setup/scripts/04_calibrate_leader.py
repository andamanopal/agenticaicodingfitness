#!/usr/bin/env python3
"""Run LeRobot's interactive leader calibration."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lerobot.teleoperators.so_leader import SO101Leader

import robot_config


leader = SO101Leader(robot_config.leader_config())
leader.connect(calibrate=False)
try:
    leader.calibrate()
finally:
    leader.disconnect()

