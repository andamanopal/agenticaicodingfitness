#!/usr/bin/env python3
"""List OpenCV cameras and save their test frames."""

from lerobot.scripts.lerobot_find_cameras import find_and_print_cameras


if __name__ == "__main__":
    find_and_print_cameras("opencv")

