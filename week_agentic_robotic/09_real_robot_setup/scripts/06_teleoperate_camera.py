#!/usr/bin/env python3
"""Teleoperate with the configured wrist camera and Rerun visualization."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rerun as rr
from lerobot.processor import make_default_processors
from lerobot.robots.so_follower import SO101Follower
from lerobot.scripts.lerobot_teleoperate import teleop_loop
from lerobot.teleoperators.so_leader import SO101Leader
from lerobot.utils.visualization_utils import init_rerun

import robot_config


init_rerun(session_name="teleoperation")
robot = SO101Follower(robot_config.follower_config(with_camera=True))
leader = SO101Leader(robot_config.leader_config())
leader_processor, robot_processor, observation_processor = make_default_processors()
leader.connect()
robot.connect()
try:
    teleop_loop(
        teleop=leader,
        robot=robot,
        fps=robot_config.TELEOP_FPS,
        display_data=True,
        teleop_action_processor=leader_processor,
        robot_action_processor=robot_processor,
        robot_observation_processor=observation_processor,
    )
finally:
    rr.rerun_shutdown()
    leader.disconnect()
    robot.disconnect()

