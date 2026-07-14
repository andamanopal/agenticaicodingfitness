"""Calibrate and persist the physical wrist-camera workspace from the portal."""

import hashlib
import json
import math
import re
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = ROOT / "09_real_robot_setup" / "output" / "workspace_profiles"
MODEL_PATH = ROOT / "models" / "so101" / "scene.xml"
PROFILE_VERSION = 4
INNER_CORNERS = (8, 6)
SQUARE_SIZE_METERS = 0.009
MINIMUM_INTRINSIC_SAMPLES = 10
VIEWPOINTS = ("survey_left", "survey_center", "survey_right")
ARM_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
MAX_HAND_EYE_TRANSLATION_SPREAD_METERS = 0.025
MAX_HAND_EYE_ROTATION_SPREAD_DEGREES = 10.0
MIN_CAMERA_BASELINE_METERS = 0.04
MIN_JOINT_SPREAD_DEGREES = 15.0
MIN_GRIPPER_ROTATION_SPREAD_DEGREES = 15.0
DEFAULT_WORKSPACE = {
    "minimum": [0.10, -0.20, 0.004],
    "maximum": [0.34, 0.20, 0.12],
    "table_minimum": [-0.08, -0.38],
    "table_maximum": [0.60, 0.38],
    "destinations": {
        "left side": [0.18, 0.16],
        "center": [0.24, 0.0],
        "right side": [0.18, -0.16],
    },
    "gripper_open_percent": 70.0,
    "gripper_closed_percent": 8.0,
    "base_frame_registration_confirmed": False,
    "kinematic_model_confirmed": False,
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def profile_digest(profile):
    return hashlib.sha256(
        json.dumps(profile, sort_keys=True).encode("utf-8")
    ).hexdigest()


def opencv():
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "physical camera calibration requires opencv-python from requirements.txt"
        ) from error
    return cv2


def checkerboard_points():
    columns, rows = INNER_CORNERS
    points = np.zeros((columns * rows, 3), dtype=np.float32)
    points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    points[:, :2] *= SQUARE_SIZE_METERS
    return points


def detect_checkerboard(frame):
    """Find corners and orient them from the target's red origin marker."""
    cv2 = opencv()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    found, corners = cv2.findChessboardCornersSB(
        gray,
        INNER_CORNERS,
        flags=cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if not found:
        raise RuntimeError(
            "checkerboard not found; keep all 8 × 6 inner corners visible and refocus"
        )
    red = (
        (frame[..., 0] > 130)
        & (frame[..., 0] > frame[..., 1].astype(float) * 1.35)
        & (frame[..., 0] > frame[..., 2].astype(float) * 1.35)
    )
    component_count, _, statistics, centroids = cv2.connectedComponentsWithStats(
        red.astype(np.uint8),
    )
    marker_centers = [
        centroids[index]
        for index in range(1, component_count)
        if statistics[index, cv2.CC_STAT_AREA] >= 20
    ]
    if not marker_centers:
        raise RuntimeError(
            "red origin marker not found; use the supplied target printed in color"
        )

    columns, rows = INNER_CORNERS
    grid = corners.reshape(rows, columns, 2)
    grid_corners = (
        (0, 0),
        (0, columns - 1),
        (rows - 1, 0),
        (rows - 1, columns - 1),
    )
    distance, origin_row, origin_column = min(
        (
            float(np.linalg.norm(grid[row, column] - marker_center)),
            row,
            column,
        )
        for marker_center in marker_centers
        for row, column in grid_corners
    )
    horizontal_spacing = np.linalg.norm(grid[:, 1:] - grid[:, :-1], axis=2)
    vertical_spacing = np.linalg.norm(grid[1:] - grid[:-1], axis=2)
    square_spacing = float(np.median(np.concatenate([
        horizontal_spacing.reshape(-1),
        vertical_spacing.reshape(-1),
    ])))
    if distance > 2.5 * square_spacing:
        raise RuntimeError(
            "the red marker is not beside a checkerboard corner; remove other red items"
        )
    if origin_row:
        grid = grid[::-1]
    if origin_column:
        grid = grid[:, ::-1]
    return gray, grid.reshape(-1, 1, 2).astype(np.float32)


def rotation_z(angle_degrees):
    angle = math.radians(angle_degrees)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.array([
        [cosine, -sine, 0.0],
        [sine, cosine, 0.0],
        [0.0, 0.0, 1.0],
    ])


def checkerboard_rotation_base(angle_degrees):
    """Map the face-up printed board frame into the robot base frame."""
    return rotation_z(angle_degrees) @ np.diag([1.0, -1.0, -1.0])


def gripper_transform(model, data, positions):
    """Return the model's base-to-gripper transform at measured joint angles."""
    import mujoco

    data.qpos[:] = 0.0
    for joint in ARM_JOINTS:
        joint_id = model.joint(joint).id
        qpos_index = model.jnt_qposadr[joint_id]
        data.qpos[qpos_index] = math.radians(float(positions[joint]))
    mujoco.mj_forward(model, data)
    gripper_site = model.site("gripperframe").id
    transform = np.eye(4)
    transform[:3, :3] = data.site_xmat[gripper_site].reshape(3, 3)
    transform[:3, 3] = data.site_xpos[gripper_site]
    return transform


def average_rigid_transform(transforms):
    transform = np.eye(4)
    transform[:3, 3] = np.mean(
        [item[:3, 3] for item in transforms],
        axis=0,
    )
    left, _, right = np.linalg.svd(
        np.sum([item[:3, :3] for item in transforms], axis=0)
    )
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    transform[:3, :3] = rotation
    return transform


def camera_transforms_from_joint_positions(profile, positions_by_viewpoint):
    """Apply the measured wrist mount to actual LeRobot joint observations."""
    validation = profile.get("kinematic_validation", {})
    if validation.get("status") != "pass" or not validation.get(
        "T_gripper_camera_cv"
    ):
        raise RuntimeError(
            "physical camera mount transform is not validated; repeat survey setup"
        )
    try:
        import mujoco
    except ImportError as error:
        raise RuntimeError("physical camera poses require mujoco") from error
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    transform_gripper_camera = np.asarray(
        validation["T_gripper_camera_cv"],
        dtype=float,
    )
    return {
        name: gripper_transform(model, data, positions) @ transform_gripper_camera
        for name, positions in positions_by_viewpoint.items()
    }


class PhysicalWorkspace:
    """One camera/profile calibration scoped to one LeRobot follower ID."""

    def __init__(self):
        self.lock = threading.Lock()
        self.robot_id = None
        self.camera_signature = None
        self.path = None
        self.profile = None
        self.samples = []

    def activate(self, robot_id, camera_signature):
        safe_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", robot_id).strip("_")
        if not safe_id:
            raise ValueError("robot ID cannot produce an empty profile name")
        PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        path = PROFILE_DIR / f"{safe_id}.json"
        profile = None
        if path.exists():
            try:
                profile = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                profile = None
        with self.lock:
            self.robot_id = robot_id
            self.camera_signature = dict(camera_signature)
            self.path = path
            self.profile = profile
            self.samples = []

    def deactivate(self):
        with self.lock:
            self.robot_id = None
            self.camera_signature = None
            self.path = None
            self.profile = None
            self.samples = []

    def snapshot(self):
        with self.lock:
            profile = deepcopy(self.profile)
            camera_signature = deepcopy(self.camera_signature)
            robot_id = self.robot_id
            sample_count = len(self.samples)
            path = str(self.path) if self.path else None
        intrinsics = (profile or {}).get("intrinsics", {})
        image_shape_matches = bool(
            camera_signature
            and intrinsics.get("width") == camera_signature.get("width")
            and intrinsics.get("height") == camera_signature.get("height")
        )
        compatible = bool(
            profile
            and profile.get("version") == PROFILE_VERSION
            and profile.get("robot_id") == robot_id
            and profile.get("camera_signature") == camera_signature
            and image_shape_matches
        )
        viewpoints = (profile or {}).get("viewpoints", {})
        kinematic_validation = (profile or {}).get("kinematic_validation", {})
        survey_ready = bool(
            compatible
            and (profile or {}).get("intrinsics")
            and (profile or {}).get("travel_pose")
            and all(name in viewpoints for name in VIEWPOINTS)
        )
        ready = bool(
            survey_ready
            and kinematic_validation.get("status") == "pass"
            and kinematic_validation.get("T_gripper_camera_cv")
        )
        return {
            "active": self.robot_id is not None,
            "robot_id": robot_id,
            "profile_path": path,
            "compatible": compatible,
            "intrinsics_ready": bool(compatible and (profile or {}).get("intrinsics")),
            "intrinsic_samples": sample_count,
            "intrinsic_samples_required": MINIMUM_INTRINSIC_SAMPLES,
            "intrinsic_rms_pixels": (profile or {}).get("intrinsics", {}).get("rms_pixels"),
            "travel_pose_ready": bool(compatible and (profile or {}).get("travel_pose")),
            "viewpoints": {
                name: {
                    "ready": bool(compatible and name in viewpoints),
                    "reprojection_rmse_pixels": viewpoints.get(name, {}).get(
                        "reprojection_rmse_pixels"
                    ),
                }
                for name in VIEWPOINTS
            },
            "workspace": deepcopy((profile or {}).get("workspace", DEFAULT_WORKSPACE)),
            "kinematic_validation": deepcopy(kinematic_validation),
            "survey_ready": survey_ready,
            "scan_ready": ready,
            "execution_ready": bool(
                ready
                and (profile or {}).get("workspace", {}).get(
                    "base_frame_registration_confirmed", False
                )
                and (profile or {}).get("workspace", {}).get(
                    "kinematic_model_confirmed", False
                )
                and kinematic_validation.get("status") == "pass"
            ),
        }

    def add_intrinsic_sample(self, frame):
        gray, corners = detect_checkerboard(frame)
        with self.lock:
            self.samples.append({
                "corners": corners.astype(np.float32),
                "image_size": (int(gray.shape[1]), int(gray.shape[0])),
            })
            self.samples = self.samples[-30:]
            count = len(self.samples)
        return {
            "status": "sample_captured",
            "samples": count,
            "required": MINIMUM_INTRINSIC_SAMPLES,
        }

    def solve_intrinsics(self):
        cv2 = opencv()
        with self.lock:
            samples = list(self.samples)
            camera_signature = deepcopy(self.camera_signature)
            robot_id = self.robot_id
        if len(samples) < MINIMUM_INTRINSIC_SAMPLES:
            raise RuntimeError(
                f"capture at least {MINIMUM_INTRINSIC_SAMPLES} checkerboard views first"
            )
        image_sizes = {sample["image_size"] for sample in samples}
        if len(image_sizes) != 1:
            raise RuntimeError("camera resolution changed during intrinsic calibration")
        object_points = [checkerboard_points() for _ in samples]
        image_points = [sample["corners"] for sample in samples]
        rms, matrix, distortion, _, _ = cv2.calibrateCamera(
            object_points,
            image_points,
            samples[0]["image_size"],
            None,
            None,
        )
        if not np.isfinite(matrix).all() or rms > 2.0:
            raise RuntimeError(
                f"camera calibration RMS is {rms:.2f}px; recapture sharper, varied views"
            )
        profile = {
            "version": PROFILE_VERSION,
            "robot_id": robot_id,
            "camera_signature": camera_signature,
            "intrinsics": {
                "width": samples[0]["image_size"][0],
                "height": samples[0]["image_size"][1],
                "K": np.round(matrix, 8).tolist(),
                "distortion": np.round(distortion.reshape(-1), 8).tolist(),
                "rms_pixels": round(float(rms), 4),
                "samples": len(samples),
                "checkerboard_inner_corners": list(INNER_CORNERS),
                "square_size_meters": SQUARE_SIZE_METERS,
            },
            "workspace": deepcopy(DEFAULT_WORKSPACE),
            "viewpoints": {},
            "created_at_utc": utc_now(),
            "updated_at_utc": utc_now(),
        }
        with self.lock:
            self.profile = profile
            self._save_locked()
        return {
            "status": "intrinsics_calibrated",
            "rms_pixels": round(float(rms), 4),
            "samples": len(samples),
        }

    def record_travel_pose(self, positions):
        self._require_intrinsics()
        with self.lock:
            self.profile["travel_pose"] = {
                "joint_positions_degrees": {
                    name: round(float(value), 5)
                    for name, value in positions.items()
                },
                "recorded_at_utc": utc_now(),
            }
            self.profile["updated_at_utc"] = utc_now()
            self._save_locked()
        return {"status": "travel_pose_recorded"}

    def record_viewpoint(
        self,
        name,
        frame,
        positions,
        board_origin_x,
        board_origin_y,
        board_origin_z,
        board_yaw_degrees,
    ):
        if name not in VIEWPOINTS:
            raise ValueError(f"viewpoint must be one of {list(VIEWPOINTS)}")
        profile = self._require_intrinsics()
        cv2 = opencv()
        intrinsics = profile["intrinsics"]
        matrix = np.asarray(intrinsics["K"], dtype=float)
        distortion = np.asarray(intrinsics["distortion"], dtype=float)
        gray, corners = detect_checkerboard(frame)
        if (gray.shape[1], gray.shape[0]) != (
            intrinsics["width"],
            intrinsics["height"],
        ):
            raise RuntimeError("live frame resolution does not match calibrated intrinsics")
        object_points = checkerboard_points()
        solved, rotation_vector, translation = cv2.solvePnP(
            object_points,
            corners,
            matrix,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not solved:
            raise RuntimeError("OpenCV could not solve this checkerboard camera pose")
        rotation_camera_board, _ = cv2.Rodrigues(rotation_vector)
        transform_camera_board = np.eye(4)
        transform_camera_board[:3, :3] = rotation_camera_board
        transform_camera_board[:3, 3] = translation.reshape(3)

        transform_base_board = np.eye(4)
        transform_base_board[:3, :3] = checkerboard_rotation_base(
            board_yaw_degrees
        )
        transform_base_board[:3, 3] = [
            board_origin_x,
            board_origin_y,
            board_origin_z,
        ]
        transform_base_camera = transform_base_board @ np.linalg.inv(
            transform_camera_board
        )
        camera_height = float(transform_base_camera[2, 3])
        if not 0.04 <= camera_height <= 0.60:
            raise RuntimeError(
                f"solved camera height is {camera_height:.3f}m; check board origin and orientation"
            )
        projected, _ = cv2.projectPoints(
            object_points,
            rotation_vector,
            translation,
            matrix,
            distortion,
        )
        residuals = projected.reshape(-1, 2) - corners.reshape(-1, 2)
        rmse = float(np.sqrt(np.mean(np.square(residuals))))
        if rmse > 1.5:
            raise RuntimeError(
                f"viewpoint reprojection error is {rmse:.2f}px; use a sharper frame"
            )

        with self.lock:
            self.profile["viewpoints"][name] = {
                "joint_positions_degrees": {
                    joint: round(float(value), 5)
                    for joint, value in positions.items()
                },
                "T_base_camera_cv": np.round(transform_base_camera, 8).tolist(),
                "board_pose_base": {
                    "origin_meters": [
                        float(board_origin_x),
                        float(board_origin_y),
                        float(board_origin_z),
                    ],
                    "yaw_degrees": float(board_yaw_degrees),
                },
                "reprojection_rmse_pixels": round(rmse, 4),
                "recorded_at_utc": utc_now(),
            }
            self.profile["workspace"]["base_frame_registration_confirmed"] = False
            self._update_kinematic_validation_locked()
            self.profile["updated_at_utc"] = utc_now()
            self._save_locked()
        return {
            "status": "viewpoint_recorded",
            "viewpoint": name,
            "reprojection_rmse_pixels": round(rmse, 4),
            "camera_height_meters": round(camera_height, 4),
        }

    def update_workspace(self, workspace):
        profile = self._require_intrinsics()
        minimum = np.asarray(workspace["minimum"], dtype=float)
        maximum = np.asarray(workspace["maximum"], dtype=float)
        table_minimum = np.asarray(workspace["table_minimum"], dtype=float)
        table_maximum = np.asarray(workspace["table_maximum"], dtype=float)
        if (
            minimum.shape != (3,)
            or maximum.shape != (3,)
            or not np.isfinite(minimum).all()
            or not np.isfinite(maximum).all()
            or np.any(maximum <= minimum)
        ):
            raise ValueError("workspace maximum must exceed minimum on all three axes")
        if (
            table_minimum.shape != (2,)
            or table_maximum.shape != (2,)
            or not np.isfinite(table_minimum).all()
            or not np.isfinite(table_maximum).all()
            or np.any(table_maximum <= table_minimum)
        ):
            raise ValueError("table maximum must exceed minimum on both axes")
        if np.any(minimum[:2] < table_minimum) or np.any(maximum[:2] > table_maximum):
            raise ValueError("the reachable workspace must stay inside the measured table")
        destinations = workspace["destinations"]
        if set(destinations) != {"left side", "center", "right side"}:
            raise ValueError("workspace must define left side, center, and right side")
        for name, point in destinations.items():
            point = np.asarray(point, dtype=float)
            if (
                point.shape != (2,)
                or not np.isfinite(point).all()
                or np.any(point < minimum[:2] + 0.01)
                or np.any(point > maximum[:2] - 0.01)
            ):
                raise ValueError(f"{name} destination is outside the workspace")
        if not (
            destinations["left side"][1]
            > destinations["center"][1]
            > destinations["right side"][1]
        ):
            raise ValueError("left, center, and right destinations must follow +Y to -Y")
        open_percent = float(workspace["gripper_open_percent"])
        closed_percent = float(workspace["gripper_closed_percent"])
        if not 1.0 <= closed_percent < open_percent <= 99.0:
            raise ValueError("gripper closed percent must be below open percent")
        merged = deepcopy(profile.get("workspace", DEFAULT_WORKSPACE))
        merged.update(workspace)
        with self.lock:
            self.profile["workspace"] = merged
            self.profile["updated_at_utc"] = utc_now()
            self._save_locked()
        return {"status": "workspace_saved"}

    def scan_profile(self):
        snapshot = self.snapshot()
        if not snapshot["scan_ready"]:
            raise RuntimeError(
                "physical workspace is not ready; calibrate intrinsics and record travel plus three survey poses"
            )
        with self.lock:
            return deepcopy(self.profile)

    def undistort(self, frame):
        profile = self._require_intrinsics()
        cv2 = opencv()
        intrinsics = profile["intrinsics"]
        matrix = np.asarray(intrinsics["K"], dtype=float)
        distortion = np.asarray(intrinsics["distortion"], dtype=float)
        return cv2.undistort(frame, matrix, distortion, None, matrix)

    def clear(self):
        with self.lock:
            if self.path:
                self.path.unlink(missing_ok=True)
            self.profile = None
            self.samples = []

    def _require_intrinsics(self):
        snapshot = self.snapshot()
        if not snapshot["compatible"] or not snapshot["intrinsics_ready"]:
            raise RuntimeError(
                "calibrate this exact camera resolution and orientation first"
            )
        with self.lock:
            return deepcopy(self.profile)

    def _save_locked(self):
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.profile, indent=2) + "\n")
        temporary.replace(self.path)

    def _update_kinematic_validation_locked(self):
        viewpoints = self.profile.get("viewpoints", {})
        if not all(name in viewpoints for name in VIEWPOINTS):
            self.profile["kinematic_validation"] = {
                "status": "incomplete",
                "reason": "record all three survey poses",
            }
            return

        try:
            import mujoco

            model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
            data = mujoco.MjData(model)
            hand_eye_transforms = []
            camera_transforms = []
            gripper_transforms = []
            joint_vectors = []
            for name in VIEWPOINTS:
                record = viewpoints[name]
                positions = record["joint_positions_degrees"]
                transform_base_gripper = gripper_transform(
                    model,
                    data,
                    positions,
                )
                transform_base_camera = np.asarray(
                    record["T_base_camera_cv"],
                    dtype=float,
                )
                hand_eye_transforms.append(
                    np.linalg.inv(transform_base_gripper) @ transform_base_camera
                )
                camera_transforms.append(transform_base_camera)
                gripper_transforms.append(transform_base_gripper)
                joint_vectors.append(np.asarray([
                    float(positions[joint]) for joint in ARM_JOINTS
                ]))

            translation_spread = 0.0
            rotation_spread = 0.0
            camera_baseline = 0.0
            joint_spread = 0.0
            gripper_rotation_spread = 0.0
            for first in range(len(VIEWPOINTS)):
                for second in range(first + 1, len(VIEWPOINTS)):
                    translation_spread = max(
                        translation_spread,
                        float(np.linalg.norm(
                            hand_eye_transforms[first][:3, 3]
                            - hand_eye_transforms[second][:3, 3]
                        )),
                    )
                    relative_rotation = (
                        hand_eye_transforms[first][:3, :3].T
                        @ hand_eye_transforms[second][:3, :3]
                    )
                    cosine = np.clip(
                        (np.trace(relative_rotation) - 1.0) / 2.0,
                        -1.0,
                        1.0,
                    )
                    rotation_spread = max(
                        rotation_spread,
                        math.degrees(math.acos(float(cosine))),
                    )
                    camera_baseline = max(
                        camera_baseline,
                        float(np.linalg.norm(
                            camera_transforms[first][:3, 3]
                            - camera_transforms[second][:3, 3]
                        )),
                    )
                    joint_spread = max(
                        joint_spread,
                        float(np.linalg.norm(
                            joint_vectors[first] - joint_vectors[second]
                        )),
                    )
                    gripper_relative_rotation = (
                        gripper_transforms[first][:3, :3].T
                        @ gripper_transforms[second][:3, :3]
                    )
                    gripper_cosine = np.clip(
                        (np.trace(gripper_relative_rotation) - 1.0) / 2.0,
                        -1.0,
                        1.0,
                    )
                    gripper_rotation_spread = max(
                        gripper_rotation_spread,
                        math.degrees(math.acos(float(gripper_cosine))),
                    )

            mount_distance = max(
                float(np.linalg.norm(transform[:3, 3]))
                for transform in hand_eye_transforms
            )
            checks = {
                "viewpoints_are_distinct": (
                    camera_baseline >= MIN_CAMERA_BASELINE_METERS
                    and joint_spread >= MIN_JOINT_SPREAD_DEGREES
                ),
                "viewpoint_rotation_is_observable": (
                    gripper_rotation_spread
                    >= MIN_GRIPPER_ROTATION_SPREAD_DEGREES
                ),
                "fixed_camera_translation_is_consistent": (
                    translation_spread <= MAX_HAND_EYE_TRANSLATION_SPREAD_METERS
                ),
                "fixed_camera_rotation_is_consistent": (
                    rotation_spread <= MAX_HAND_EYE_ROTATION_SPREAD_DEGREES
                ),
                "camera_mount_offset_is_plausible": mount_distance <= 0.25,
            }
            failed = [name for name, passed in checks.items() if not passed]
            validation = {
                "status": "pass" if not failed else "review",
                "checks": checks,
                "failed_checks": failed,
                "camera_baseline_meters": round(camera_baseline, 4),
                "joint_spread_degrees": round(joint_spread, 2),
                "gripper_rotation_spread_degrees": round(
                    gripper_rotation_spread,
                    2,
                ),
                "hand_eye_translation_spread_meters": round(
                    translation_spread,
                    4,
                ),
                "hand_eye_rotation_spread_degrees": round(rotation_spread, 2),
                "maximum_camera_mount_offset_meters": round(mount_distance, 4),
            }
            if not failed:
                validation["T_gripper_camera_cv"] = np.round(
                    average_rigid_transform(hand_eye_transforms),
                    8,
                ).tolist()
            self.profile["kinematic_validation"] = validation
        except Exception as error:
            self.profile["kinematic_validation"] = {
                "status": "error",
                "reason": str(error),
            }
