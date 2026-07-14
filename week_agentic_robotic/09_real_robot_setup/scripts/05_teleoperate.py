#!/usr/bin/env python3
"""Teleoperate the follower from the leader until Ctrl+C."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lerobot.processor import make_default_processors
from lerobot.robots.so_follower import SO101Follower
from lerobot.scripts.lerobot_teleoperate import teleop_loop
from lerobot.teleoperators.so_leader import SO101Leader

import robot_config


robot = SO101Follower(robot_config.follower_config())
leader = SO101Leader(robot_config.leader_config())
leader_processor, robot_processor, observation_processor = make_default_processors()
leader.connect()
robot.connect()
try:
    teleop_loop(
        teleop=leader,
        robot=robot,
        fps=robot_config.TELEOP_FPS,
        teleop_action_processor=leader_processor,
        robot_action_processor=robot_processor,
        robot_observation_processor=observation_processor,
    )
finally:
    leader.disconnect()
    robot.disconnect()

