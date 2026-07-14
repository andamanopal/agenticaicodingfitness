"""Known instructor hardware values used by the fallback setup scripts."""

from lerobot.cameras.configs import Cv2Rotation
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101LeaderConfig


FOLLOWER_PORT = "/dev/tty.usbmodem5B3D0482201"
LEADER_PORT = "/dev/tty.usbmodem5B610338721"
FOLLOWER_ID = "my_awesome_follower_arm"
LEADER_ID = "my_awesome_leader_arm"
CAMERA_INDEX = 0
CAMERA_WIDTH = 480
CAMERA_HEIGHT = 640
CAMERA_ROTATION = Cv2Rotation.ROTATE_270
TELEOP_FPS = 60
RECORD_FPS = 30


def follower_config(with_camera=False):
    cameras = {}
    if with_camera:
        cameras["wrist"] = OpenCVCameraConfig(
            index_or_path=CAMERA_INDEX,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            fps=RECORD_FPS,
            rotation=CAMERA_ROTATION,
        )
    return SO101FollowerConfig(
        port=FOLLOWER_PORT,
        id=FOLLOWER_ID,
        cameras=cameras,
    )


def leader_config():
    return SO101LeaderConfig(port=LEADER_PORT, id=LEADER_ID)

