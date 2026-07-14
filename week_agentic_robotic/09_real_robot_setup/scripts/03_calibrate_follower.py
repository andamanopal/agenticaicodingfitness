#!/usr/bin/env python3
"""Run LeRobot's interactive follower calibration."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lerobot.robots.so_follower import SO101Follower

import robot_config


robot = SO101Follower(robot_config.follower_config())
robot.connect(calibrate=False)
try:
    robot.calibrate()
finally:
    robot.disconnect()

