"""Read-only discovery helpers for physical SO-ARM101 setup."""

import json
import os
import threading
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


CALIBRATION_ROOT = Path(
    os.environ.get(
        "HF_LEROBOT_CALIBRATION",
        "~/.cache/huggingface/lerobot/calibration",
    )
).expanduser()
JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
MOTOR_RESOLUTION = 4095
BODY_LIMIT_MARGIN_DEGREES = 2.0
GRIPPER_LIMIT_MARGIN_PERCENT = 1.0
WORKSHOP_LIMIT_OVERRIDES = {
    "wrist_roll": (-90.0, 90.0),
}


class HardwareSetupManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.last_discovery = {
            "scanned": False,
            "lerobot": lerobot_status(),
            "ports": [],
            "cameras": [],
            "calibrations": calibration_records(),
            "camera_error": None,
        }

    def discover(self, include_cameras=True):
        result = {
            "scanned": True,
            "lerobot": lerobot_status(),
            "ports": serial_ports(),
            "cameras": [],
            "calibrations": calibration_records(),
            "camera_error": None,
        }
        if include_cameras:
            try:
                result["cameras"] = opencv_cameras()
            except Exception as error:
                result["camera_error"] = str(error)
        with self.lock:
            self.last_discovery = result
        return result

    def snapshot(self):
        with self.lock:
            return dict(self.last_discovery)


def lerobot_status():
    try:
        installed_version = version("lerobot")
    except PackageNotFoundError:
        return {"installed": False, "version": None}
    return {"installed": True, "version": installed_version}


def serial_ports():
    try:
        from serial.tools import list_ports

        records = [
            {
                "device": port.device,
                "description": port.description or "Serial device",
                "serial_number": port.serial_number,
                "vid": port.vid,
                "pid": port.pid,
            }
            for port in list_ports.comports()
        ]
    except ImportError:
        candidates = set()
        for pattern in ("tty.usb*", "cu.usb*", "ttyACM*", "ttyUSB*"):
            candidates.update(Path("/dev").glob(pattern))
        records = [
            {
                "device": str(path),
                "description": "Serial device",
                "serial_number": None,
                "vid": None,
                "pid": None,
            }
            for path in candidates
        ]
    return sorted(records, key=lambda record: record["device"])


def opencv_cameras():
    try:
        from lerobot.scripts.lerobot_find_cameras import find_all_opencv_cameras
    except ImportError as error:
        raise RuntimeError(
            "LeRobot camera support is not installed in this environment"
        ) from error
    return [_json_safe(record) for record in find_all_opencv_cameras()]


def calibration_records():
    records = []
    patterns = (
        ("follower", "robots", "so*_follower"),
        ("leader", "teleoperators", "so*_leader"),
    )
    for role, category, device_pattern in patterns:
        for path in sorted(
            CALIBRATION_ROOT.glob(f"{category}/{device_pattern}/*.json")
        ):
            try:
                calibration = json.loads(path.read_text())
                limits = normalized_joint_limits(calibration)
            except (OSError, json.JSONDecodeError, ValueError):
                continue
            records.append({
                "role": role,
                "type": path.parent.name,
                "id": path.stem,
                "path": str(path),
                "limits": limits,
            })
    return records


def normalized_joint_limits(calibration):
    limits = {}
    for joint in JOINT_NAMES:
        record = calibration.get(joint)
        if record is None:
            raise ValueError(f"calibration is missing {joint}")
        raw_min = int(_value(record, "range_min"))
        raw_max = int(_value(record, "range_max"))
        if raw_max <= raw_min:
            raise ValueError(f"invalid calibration range for {joint}")
        if joint == "gripper":
            minimum = GRIPPER_LIMIT_MARGIN_PERCENT
            maximum = 100.0 - GRIPPER_LIMIT_MARGIN_PERCENT
            unit = "percent"
        else:
            half_range = (raw_max - raw_min) * 180.0 / MOTOR_RESOLUTION
            minimum = -half_range + BODY_LIMIT_MARGIN_DEGREES
            maximum = half_range - BODY_LIMIT_MARGIN_DEGREES
            unit = "degrees"
            if joint in WORKSHOP_LIMIT_OVERRIDES:
                override_min, override_max = WORKSHOP_LIMIT_OVERRIDES[joint]
                minimum = max(minimum, override_min)
                maximum = min(maximum, override_max)
        limits[joint] = {
            "min": round(minimum, 2),
            "max": round(maximum, 2),
            "unit": unit,
            "raw_min": raw_min,
            "raw_max": raw_max,
        }
    return limits


def _value(record, name):
    return record[name] if isinstance(record, dict) else getattr(record, name)


def _json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)
