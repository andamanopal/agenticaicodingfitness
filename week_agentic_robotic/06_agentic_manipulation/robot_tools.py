"""Deterministic robot tools used by the guided and language-model agents."""

import hashlib
import json
import re
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "so101"
WORKSHOP_ROOM_PATH = MODEL_DIR / "workshop_room.xml"
SEMANTIC_PATH = ROOT / "05_semantic_scene" / "output" / "latest_scene" / "semantic_scene.json"
SCAN_PATH = ROOT / "04_active_perception" / "output" / "latest_scan" / "scan.json"
BACKGROUND_PATH = ROOT / "04_active_perception" / "output" / "empty_scan" / "scan.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# Earlier numbered folders are runnable lessons rather than importable package
# names. Adding this one folder lets the tool layer reuse the verified motion
# controller without hiding the kinematics lesson behind a framework.
sys.path.insert(0, str(ROOT / "04_active_perception"))
from scan_workspace import (  # noqa: E402
    ScanCancelled,
    ScanViewer,
)

sys.path.insert(0, str(ROOT / "05_semantic_scene"))
from reconstruct_scene import (  # noqa: E402
    GEOMETRY_CONTRACT,
    GEOMETRY_MODALITIES,
    RECONSTRUCTION_METHOD,
    SUPPORTED_SCAN_CONTRACTS,
)

sys.path.insert(0, str(ROOT / "03_kinematic_pick_place"))
from ik import solve_ik  # noqa: E402
from pick_place import (  # noqa: E402
    CARRY_POSE,
    GRIPPER_OPEN,
    HOVER_Z,
    IK_SEED,
    Controller,
)


DESTINATIONS = {
    "left side": {"xy": [0.11, 0.19], "verification_z": 0.13},
    "center": {"xy": [0.18, 0.0], "verification_z": 0.10},
    "right side": {"xy": [0.11, -0.19], "verification_z": 0.13},
}
WRIST_IMAGE_MAX_EDGE = 640


class ExecutionCancelled(Exception):
    """Stop a visual rollout when the participant closes its window."""


class ExecutionViewer:
    """Adapt the workshop's dual-view renderer to the motion controller."""

    playback_speed = 6.0

    def __init__(self, model, data):
        self.data = data
        self.status = "Preparing approved rollout"
        self.viewer = ScanViewer(
            model,
            title="SO-101 approved manipulation rollout",
            heading="AGENTIC MANIPULATION",
            escape_help="Esc stops the rollout",
            playback_speed=self.playback_speed,
        )

    def sync(self):
        if self.viewer is None:
            return
        try:
            self.viewer.sync(self.data, self.status)
        except ScanCancelled as error:
            self.close()
            raise ExecutionCancelled("visual rollout cancelled by participant") from error

    def set_status(self, status):
        self.status = status

    def hold(self, seconds):
        if self.viewer is None:
            return
        deadline = time.monotonic() + seconds
        try:
            while time.monotonic() < deadline:
                self.viewer.render(self.data, self.status)
        except ScanCancelled as error:
            self.close()
            raise ExecutionCancelled("visual rollout cancelled by participant") from error

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


def normalized_words(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def scene_fingerprint(model_path):
    digest = hashlib.sha256()
    for path in (
        Path(model_path),
        MODEL_DIR / "so101.xml",
        WORKSHOP_ROOM_PATH,
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


class RobotTools:
    """One frozen scene memory plus one approved simulation execution."""

    def __init__(
        self,
        semantic_path=SEMANTIC_PATH,
        scan_path=SCAN_PATH,
        background_path=BACKGROUND_PATH,
        visualize_execution=False,
        execution_viewer_factory=None,
    ):
        semantic_path = Path(semantic_path)
        if not semantic_path.exists():
            raise FileNotFoundError(
                "semantic scene missing; complete 05_semantic_scene first"
            )
        self.scan_path = Path(scan_path)
        if not self.scan_path.exists():
            raise FileNotFoundError(
                "occupied scan missing; run 04_active_perception/scan_workspace.py, "
                "then rebuild geometry, crops, and semantic labels"
            )
        self.background_path = Path(background_path)
        if not self.background_path.exists():
            raise FileNotFoundError(
                "empty-table calibration missing; run "
                "04_active_perception/scan_workspace.py --scene empty, then "
                "rebuild geometry, crops, and semantic labels"
            )
        self.scene = json.loads(semantic_path.read_text())
        self.scan = json.loads(self.scan_path.read_text())
        self.background_scan = json.loads(self.background_path.read_text())
        for label, record in (
            ("occupied scan", self.scan),
            ("empty-table calibration", self.background_scan),
        ):
            is_rgb_only = (
                record.get("perception_contract") in SUPPORTED_SCAN_CONTRACTS
                and record.get("sensor_modalities") == ["rgb"]
                and not any(
                    "depth" in frame for frame in record.get("frames", [])
                )
            )
            if not is_rgb_only:
                raise ValueError(
                    f"{label} uses a legacy or privileged sensor contract; "
                    "recapture and rebuild scene memory"
                )
        if (
            self.scene.get("perception_contract") != GEOMETRY_CONTRACT
            or self.scene.get("sensor_modalities") != GEOMETRY_MODALITIES
            or self.scene.get("uses_privileged_simulator_data") is not False
            or self.scene.get("geometry_method") != RECONSTRUCTION_METHOD
        ):
            raise ValueError(
                "semantic scene is not from the current learned-depth reconstruction; "
                "recapture and rebuild scene memory"
            )
        if self.scene["scene_id"] != self.scan["scan_id"]:
            raise ValueError(
                "semantic scene is stale; rebuild Milestone 05 from the latest scan"
            )
        if self.scene["background_scan_id"] != self.background_scan["scan_id"]:
            raise ValueError(
                "empty-table calibration is stale; rebuild Milestone 05"
            )
        self.objects = {obj["id"]: obj for obj in self.scene["objects"]}
        self.frames = {frame["viewpoint"]: frame for frame in self.scan["frames"]}
        self.plans = {}
        self.simulations = {}
        self.scan_reviewed = False
        self.grounded_object_ids = set()
        self.model = None
        self.data = None
        self.controller = None
        self.execution_state = "not_started"
        self.executed_plan_id = None
        self.visualize_execution = visualize_execution
        self.execution_viewer_factory = execution_viewer_factory
        self.execution_viewer = None

    def scan_workspace(self):
        """Return the already-frozen scene memory built in Milestones 04–05."""
        quality_passed = (
            self.scan["observation_quality"]["status"] == "basic_check_passed"
        )
        self.scan_reviewed = True
        return {
            "scene_id": self.scene["scene_id"],
            "object_count": len(self.objects),
            "observation_quality": self.scan["observation_quality"],
            "semantic_provider": self.scene["semantic_provider"],
            "status": "ready" if quality_passed else "review_required",
        }

    def find_objects(self, query):
        if not self.scan_reviewed:
            return {
                "query": query,
                "status": "error",
                "reason": "call scan_workspace before grounding an object",
                "matches": [],
            }
        query_words = normalized_words(query)
        matches = []

        for obj in self.objects.values():
            names = [obj["label"], *obj["aliases"]]
            best_score = 0
            matched_name = None
            for name in names:
                name_words = normalized_words(name)
                score = len(query_words & name_words)
                if query.lower() == name.lower():
                    score += 3
                if score > best_score:
                    best_score = score
                    matched_name = name
            if best_score:
                matches.append({
                    "id": obj["id"],
                    "label": obj["label"],
                    "aliases": obj["aliases"],
                    "matched_name": matched_name,
                    "semantic_confidence": obj["semantic_confidence"],
                    "geometry_confidence": obj["geometry_confidence"],
                    "evidence_views": obj["evidence_views"],
                    "score": best_score,
                })

        matches.sort(key=lambda match: match["score"], reverse=True)
        if not matches:
            status = "not_found"
        elif len(matches) > 1 and matches[0]["score"] == matches[1]["score"]:
            status = "ambiguous"
        else:
            status = "found"
            matches = matches[:1]
            self.grounded_object_ids.add(matches[0]["id"])
        return {"query": query, "status": status, "matches": matches}

    def plan_pick_and_place(self, object_id, destination):
        if not self.scan_reviewed:
            return {"status": "error", "reason": "call scan_workspace first"}
        if object_id not in self.objects:
            return {"status": "error", "reason": f"unknown object {object_id}"}
        if object_id not in self.grounded_object_ids:
            return {
                "status": "error",
                "reason": "ground this object with find_objects before planning",
            }
        if destination not in DESTINATIONS:
            return {
                "status": "error",
                "reason": f"destination must be one of {list(DESTINATIONS)}",
            }

        obj = self.objects[object_id]
        destination_spec = DESTINATIONS[destination]
        grasp_z = max(0.01, min(0.03, obj["position"][2] * 0.65))
        jaw_yaw = obj["yaw_radians"] + np.pi / 2.0
        plan_id = f"{self.scene['scene_id']}:{object_id}:{destination.replace(' ', '_')}"
        plan = {
            "plan_id": plan_id,
            "scene_id": self.scene["scene_id"],
            "object_id": object_id,
            "object_label": obj["label"],
            "pick": {
                "xy": obj["position"][:2],
                "grasp_z": round(grasp_z, 4),
                "jaw_yaw": round(float(jaw_yaw), 4),
                "estimated_width": obj["dimensions"][1],
                "gripper_target": 0.0,
            },
            "place": {
                "region": destination,
                "xy": destination_spec["xy"],
                "place_z": round(grasp_z + 0.003, 4),
                "verification_z": destination_spec["verification_z"],
            },
            "geometry_confidence": obj["geometry_confidence"],
            "semantic_confidence": obj["semantic_confidence"],
        }
        self.plans[plan_id] = plan
        return {"status": "planned", **plan}

    def _waypoint_ik(self, model, scratch, position, jaw_yaw=None):
        target = np.asarray(position, dtype=float)
        pan_seed = IK_SEED.copy()
        pan_seed[0] = np.arctan2(target[1], target[0])
        solutions = [
            solve_ik(model, scratch, target, seed)
            for seed in (IK_SEED, pan_seed, CARRY_POSE)
        ]
        q, residual = min(solutions, key=lambda result: result[1])
        wrist_margin = None

        if jaw_yaw is not None:
            scratch.qpos[:] = 0.0
            scratch.qpos[:5] = q
            mujoco.mj_forward(model, scratch)
            fixed = scratch.geom("fixed_jaw_sph_tip1").xpos[:2]
            moving = scratch.geom("moving_jaw_sph_tip1").xpos[:2]
            jaw_axis = moving - fixed
            current = np.arctan2(jaw_axis[1], jaw_axis[0])
            delta = (jaw_yaw - current + np.pi) % (2 * np.pi) - np.pi
            if delta > np.pi / 2:
                delta -= np.pi
            elif delta < -np.pi / 2:
                delta += np.pi
            wrist_joint_id = model.joint("wrist_roll").id
            wrist_qpos_index = model.jnt_qposadr[wrist_joint_id]
            requested = q[wrist_qpos_index] + delta
            wrist_margin = min(
                requested - model.jnt_range[wrist_joint_id, 0],
                model.jnt_range[wrist_joint_id, 1] - requested,
            )

        return {
            "position": np.round(target, 4).tolist(),
            "residual_meters": round(float(residual), 6),
            "wrist_margin_radians": (
                None if wrist_margin is None else round(float(wrist_margin), 4)
            ),
            "reachable": bool(
                residual <= 0.03 and (wrist_margin is None or wrist_margin >= 0.0)
            ),
        }

    def simulate_plan(self, plan_id):
        if plan_id not in self.plans:
            return {"safe": False, "reason": "unknown plan; create it first"}
        plan = self.plans[plan_id]
        if plan["scene_id"] != self.scene["scene_id"]:
            return {"safe": False, "reason": "plan belongs to stale scene memory"}

        model = mujoco.MjModel.from_xml_path(str(MODEL_DIR / "scene.xml"))
        scratch = mujoco.MjData(model)
        pick_xy = plan["pick"]["xy"]
        place_xy = plan["place"]["xy"]
        jaw_yaw = plan["pick"]["jaw_yaw"]
        waypoints = [
            self._waypoint_ik(model, scratch, [*pick_xy, HOVER_Z], jaw_yaw),
            self._waypoint_ik(model, scratch, [*pick_xy, plan["pick"]["grasp_z"]], jaw_yaw),
            self._waypoint_ik(model, scratch, [*place_xy, HOVER_Z]),
            self._waypoint_ik(model, scratch, [*place_xy, plan["place"]["place_z"]]),
            self._waypoint_ik(
                model,
                scratch,
                [*place_xy, plan["place"]["verification_z"]],
            ),
        ]

        target = self.objects[plan["object_id"]]
        target_radius = max(target["dimensions"][:2]) / 2.0
        clearances = []
        for other in self.objects.values():
            if other["id"] == target["id"]:
                continue
            other_radius = max(other["dimensions"][:2]) / 2.0
            distance = float(np.linalg.norm(
                np.asarray(place_xy) - np.asarray(other["position"][:2])
            ))
            required = target_radius + other_radius + 0.01
            clearances.append({
                "other_id": other["id"],
                "distance_meters": round(distance, 4),
                "required_meters": round(required, 4),
                "clear": distance >= required,
            })

        checks = {
            "all_waypoints_reachable": all(item["reachable"] for item in waypoints),
            "object_fits_gripper": plan["pick"]["estimated_width"] <= 0.06,
            "destination_clear": all(item["clear"] for item in clearances),
            "observation_quality_passed": (
                self.scan["observation_quality"]["status"] == "basic_check_passed"
            ),
            "semantic_known": target["label"].lower() != "unknown",
            "semantic_confident": plan["semantic_confidence"] >= 0.6,
            "geometry_confident": plan["geometry_confidence"] >= 0.6,
            "geometry_has_two_views": len(target["evidence_views"]) >= 2,
            "geometry_not_clipped": not any(
                flag in {"height_ceiling", "workspace_side"}
                for flag in target.get("boundary_flags", [])
            ),
            "depth_alignment_passed": (
                "table_alignment_review" not in target.get("boundary_flags", [])
            ),
            "execution_world_supported": (
                self.scan.get("layout") == "canonical"
                and self.scan.get("scene") == "scene_office.xml"
                and self.scan.get("scene_model_sha256")
                == scene_fingerprint(MODEL_DIR / "scene_office.xml")
            ),
        }
        result = {
            "plan_id": plan_id,
            "safe": all(checks.values()),
            "planning_world": "kinematic preflight plus reconstructed clearances",
            "uses_hidden_object_truth": False,
            "checks": checks,
            "waypoints": waypoints,
            "clearances": clearances,
            "not_checked": [
                "swept-volume collisions",
                "grasp contact physics",
                "trajectory rollout",
            ],
        }
        self.simulations[plan_id] = result
        return result

    def execute_plan(self, plan_id, approval):
        if approval != "APPROVE":
            return {"status": "not_executed", "reason": "human approval not provided"}
        if plan_id not in self.simulations or not self.simulations[plan_id]["safe"]:
            return {
                "status": "not_executed",
                "reason": "plan has not passed the kinematic preflight",
            }
        if self.execution_state != "not_started":
            return {"status": "not_executed", "reason": "rescan before another execution"}

        plan = self.plans[plan_id]
        self.execution_state = "in_progress"
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_DIR / "scene_office.xml"))
        self.data = mujoco.MjData(self.model)
        if self.execution_viewer_factory is not None:
            self.execution_viewer = self.execution_viewer_factory(
                self.model,
                self.data,
            )
        elif self.visualize_execution:
            self.execution_viewer = ExecutionViewer(self.model, self.data)
        self.controller = Controller(
            self.model,
            self.data,
            viewer=self.execution_viewer,
        )
        self.controller.step_for(0.5)
        pick_xy = plan["pick"]["xy"]
        place_xy = plan["place"]["xy"]
        jaw_yaw = plan["pick"]["jaw_yaw"]
        target = self.objects[plan["object_id"]]

        # No separate destination pre-observation: reaching a goal-photo pose
        # swung the arm and flipped the wrist camera. Go straight to the pick;
        # success is verified from the source (the object left its spot).
        if self.execution_viewer:
            self.execution_viewer.set_status("Approaching the reconstructed object")
        self.controller.tuck()
        self.controller.pan_to(pick_xy)
        self.controller.ramp_ctrl(gripper_target=GRIPPER_OPEN)
        if not self.controller.move_to([*pick_xy, HOVER_Z]):
            return self._execution_failed(plan_id, "pick hover failed")
        if not self.controller.align_jaws(jaw_yaw):
            return self._execution_failed(plan_id, "grasp yaw failed")
        if not self.controller.move_to([*pick_xy, plan["pick"]["grasp_z"]]):
            return self._execution_failed(plan_id, "pick lower failed")
        if not self.controller.align_jaws(jaw_yaw):
            return self._execution_failed(plan_id, "grasp yaw failed")

        if self.execution_viewer:
            self.execution_viewer.set_status("Closing the gripper")
        self.controller.ramp_ctrl(
            gripper_target=plan["pick"]["gripper_target"], speed=0.6
        )
        self.controller.step_for(0.5)
        if self.execution_viewer:
            self.execution_viewer.set_status("Lifting the object")
        if not self.controller.move_to([*pick_xy, HOVER_Z]):
            return self._execution_failed(plan_id, "lift waypoint failed")
        if not self.controller.align_jaws(jaw_yaw):
            return self._execution_failed(plan_id, "lift yaw failed")

        if self.execution_viewer:
            self.execution_viewer.set_status("Carrying toward the destination")
        self.controller.tuck()
        self.controller.pan_to(place_xy)
        if not self.controller.move_to([*place_xy, HOVER_Z]):
            return self._execution_failed(plan_id, "place hover failed")
        if not self.controller.move_to([*place_xy, plan["place"]["place_z"]]):
            return self._execution_failed(plan_id, "place lower failed")
        if self.execution_viewer:
            self.execution_viewer.set_status("Releasing at the destination")
        self.controller.ramp_ctrl(gripper_target=0.5)
        self.controller.step_for(0.3)
        self.controller.ramp_ctrl(gripper_target=GRIPPER_OPEN)
        self.controller.step_for(0.5)
        # Several small Cartesian targets approximate a vertical retreat. A
        # single large joint interpolation sweeps sideways and can push a thin
        # object immediately after release.
        retreat_heights = np.linspace(plan["place"]["place_z"], HOVER_Z, 5)[1:]
        for height in retreat_heights:
            if not self.controller.move_to([*place_xy, height]):
                return self._execution_failed(plan_id, "release retreat failed")

        self.controller.tuck()

        self.execution_state = "completed"
        self.executed_plan_id = plan_id
        if self.execution_viewer:
            self.execution_viewer.set_status("MOTION COMPLETE — preparing verification")
            self.execution_viewer.hold(1.0)
            self.execution_viewer.close()
            self.controller.viewer = None
        return {
            "status": "executed_unverified",
            "plan_id": plan_id,
            "next_tool": "verify_plan",
        }

    def _execution_failed(self, plan_id, reason):
        """Lock the stale scene and hold the measured joint positions."""
        self.execution_state = "failed"
        if self.model is not None and self.data is not None:
            for name in (
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
                "gripper",
            ):
                actuator_id = self.model.actuator(name).id
                joint_id = self.model.joint(name).id
                qpos_index = self.model.jnt_qposadr[joint_id]
                self.data.ctrl[actuator_id] = self.data.qpos[qpos_index]
            self.controller.step_for(0.1)
        if self.execution_viewer:
            self.execution_viewer.set_status(f"EXECUTION STOPPED — {reason}")
            self.execution_viewer.hold(1.0)
            self.execution_viewer.close()
            self.controller.viewer = None
        return {
            "status": "execution_failed_requires_rescan",
            "plan_id": plan_id,
            "reason": reason,
            "safe_response": "hold current measured joint positions",
        }

    def _render_wrist(self):
        width, height = self._wrist_image_size()
        renderer = mujoco.Renderer(self.model, height=height, width=width)
        try:
            renderer.update_scene(self.data, camera="wrist_cam")
            return renderer.render().copy()
        finally:
            renderer.close()

    def _wrist_image_size(self):
        camera_id = self.model.camera("wrist_cam").id
        source_width, source_height = self.model.cam_resolution[camera_id]
        if source_width <= 0 or source_height <= 0:
            raise ValueError("wrist_cam must declare a positive resolution")

        scale = WRIST_IMAGE_MAX_EDGE / max(source_width, source_height)
        width = max(1, round(source_width * scale))
        height = max(1, round(source_height * scale))
        return width, height

    def _current_camera_frame(self):
        camera_id = self.model.camera("wrist_cam").id
        width, height = self._wrist_image_size()
        base_id = self.model.body("base").id
        rotation_world_camera = self.data.cam_xmat[camera_id].reshape(3, 3)
        rotation_world_base = self.data.xmat[base_id].reshape(3, 3)
        rotation = rotation_world_base.T @ rotation_world_camera
        rotation = rotation @ np.diag([1.0, -1.0, -1.0])
        position = rotation_world_base.T @ (
            self.data.cam_xpos[camera_id] - self.data.xpos[base_id]
        )
        fovy = np.deg2rad(self.model.cam_fovy[camera_id])
        focal = height / (2.0 * np.tan(fovy / 2.0))
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = position
        return {
            "intrinsics": {
                "width": width,
                "height": height,
                "K": [
                    [focal, 0.0, width / 2.0],
                    [0.0, focal, height / 2.0],
                    [0.0, 0.0, 1.0],
                ],
            },
            "T_base_camera_cv": transform.tolist(),
        }

    def _capture_verification_view(self, viewpoint):
        """Return to one stored survey pose and capture a post-action image."""
        frame = self.frames[viewpoint]
        arm_target = np.asarray([
            frame["joint_positions"][name]
            for name in (
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
            )
        ])
        self.controller.tuck()
        self.controller.ramp_ctrl(
            arm_target=arm_target,
            gripper_target=frame["joint_positions"]["gripper"],
        )
        return self._render_wrist()

    @staticmethod
    def _foreground(rgb, background):
        change = np.abs(rgb.astype(np.int16) - background.astype(np.int16)).mean(axis=2)
        image = Image.fromarray((change > 24.0).astype(np.uint8) * 255)
        image = image.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))
        return np.asarray(image) > 0

    @staticmethod
    def _project_points(points, frame):
        transform = np.asarray(frame["T_base_camera_cv"], dtype=float)
        rotation = transform[:3, :3]
        position = transform[:3, 3]
        camera = (np.asarray(points) - position) @ rotation
        intrinsics = np.asarray(frame["intrinsics"]["K"], dtype=float)
        u = intrinsics[0, 0] * camera[:, 0] / camera[:, 2] + intrinsics[0, 2]
        v = intrinsics[1, 1] * camera[:, 1] / camera[:, 2] + intrinsics[1, 2]
        return u, v, camera[:, 2]

    def _object_roi(self, obj, target_xy, frame, padding=12):
        offset = np.asarray(target_xy) - np.asarray(obj["position"][:2])
        footprint = np.asarray(obj["footprint_corners"]) + offset
        half_height = obj["dimensions"][2] / 2.0
        low_z = max(0.004, obj["position"][2] - half_height)
        high_z = obj["position"][2] + half_height
        points = np.asarray([
            [corner[0], corner[1], z]
            for corner in footprint
            for z in (low_z, high_z)
        ])
        u, v, depth = self._project_points(points, frame)
        if np.any(depth <= 0.01):
            return None

        width = frame["intrinsics"]["width"]
        height = frame["intrinsics"]["height"]
        left = max(0, int(np.floor(u.min())) - padding)
        top = max(0, int(np.floor(v.min())) - padding)
        right = min(width, int(np.ceil(u.max())) + padding)
        bottom = min(height, int(np.ceil(v.max())) + padding)
        if left >= right or top >= bottom:
            return None
        return [left, top, right, bottom]

    @staticmethod
    def _box_ratio(mask, box):
        if box is None:
            return 0.0
        left, top, right, bottom = box
        return float(mask[top:bottom, left:right].mean())

    @staticmethod
    def _color_signature(rgb, mask, box):
        if box is None:
            return None
        left, top, right, bottom = box
        pixels = rgb[top:bottom, left:right][mask[top:bottom, left:right]]
        pixels = pixels.astype(float)
        pixels = pixels[pixels.sum(axis=1) > 20.0]
        if len(pixels) < 12:
            return None

        # Chromaticity keeps color ratios while reducing sensitivity to a new
        # camera angle making the same surface brighter or darker.
        pixels = pixels / pixels.sum(axis=1, keepdims=True)
        signature = []
        for channel in range(3):
            counts, _ = np.histogram(pixels[:, channel], bins=8, range=(0.0, 1.0))
            signature.extend(counts)
        signature = np.asarray(signature, dtype=float)
        norm = np.linalg.norm(signature)
        return None if norm == 0 else signature / norm

    def _source_verification_reference(self, plan):
        background_frames = {
            frame["viewpoint"]: frame for frame in self.background_scan["frames"]
        }
        target = self.objects[plan["object_id"]]
        candidates = []

        for viewpoint in target["evidence_views"]:
            if viewpoint not in self.frames or viewpoint not in background_frames:
                continue
            frame = self.frames[viewpoint]
            source_box = self._object_roi(target, target["position"][:2], frame)
            if source_box is None:
                continue

            with Image.open(self.scan_path.parent / frame["rgb"]) as image:
                before = np.asarray(image.convert("RGB"))
            background_frame = background_frames[viewpoint]
            with Image.open(
                self.background_path.parent / background_frame["rgb"]
            ) as image:
                background = np.asarray(image.convert("RGB"))
            before_mask = self._foreground(before, background)
            source_ratio = self._box_ratio(before_mask, source_box)
            candidates.append((
                source_ratio,
                viewpoint,
                before,
                background,
                source_box,
            ))

        if not candidates:
            raise RuntimeError("no stored survey view contains the source object")
        return max(candidates, key=lambda candidate: candidate[0])[1:]

    def verify_plan(self, plan_id):
        if plan_id != self.executed_plan_id:
            return {"verified": False, "reason": "this plan has not been executed"}
        plan = self.plans[plan_id]
        (
            viewpoint,
            source_before_rgb,
            background,
            source_box,
        ) = self._source_verification_reference(plan)
        source_after_rgb = self._capture_verification_view(viewpoint)
        source_before_mask = self._foreground(source_before_rgb, background)
        source_after_mask = self._foreground(source_after_rgb, background)
        source_change_mask = self._foreground(source_before_rgb, source_after_rgb)
        source_evidence_mask = source_change_mask & source_before_mask

        source_before = self._box_ratio(source_before_mask, source_box)
        source_after = self._box_ratio(source_after_mask, source_box)
        source_change = self._box_ratio(source_evidence_mask, source_box)

        # Verify from the source only: the requested object visibly left its
        # spot. The destination pre-observation was removed because reaching a
        # separate goal-photo pose flipped the wrist camera and swung the arm.
        checks = {
            "source_visible_before": source_before >= 0.05,
            "source_disappeared": source_after <= max(0.04, source_before * 0.45),
            "source_changed": source_change >= 0.05,
        }
        verified = all(checks.values())

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        panel_height, panel_width = source_before_rgb.shape[:2]
        label_height = 34
        row_height = label_height + panel_height
        annotated = Image.new(
            "RGB",
            (panel_width * 2, row_height),
            "#11161c",
        )
        draw = ImageDraw.Draw(annotated)
        panels = [
            (
                source_before_rgb,
                0,
                label_height,
                "SOURCE BEFORE — stored survey",
                source_box,
                "#55c7ff",
            ),
            (
                source_after_rgb,
                panel_width,
                label_height,
                "SOURCE AFTER — same survey pose",
                source_box,
                "#55c7ff",
            ),
        ]
        for image, x, y, label, box, outline in panels:
            annotated.paste(Image.fromarray(image), (x, y))
            draw.text((x + 12, y - 24), label, fill="#f1f5f9")
            left, top, right, bottom = box
            draw.rectangle(
                [left + x, top + y, right + x, bottom + y],
                outline=outline,
                width=3,
            )
        evidence_path = OUTPUT_DIR / "verification.png"
        annotated.save(evidence_path)

        return {
            "status": "verified" if verified else "ambiguous",
            "verified": verified,
            "method": "source-change detection at the stored survey pose",
            "uses_hidden_object_truth": False,
            "reference": "stored survey plus matched live after wrist view",
            "source_viewpoint": viewpoint,
            "checks": checks,
            "source_foreground_before": round(source_before, 4),
            "source_foreground_after": round(source_after, 4),
            "source_change_ratio": round(source_change, 4),
            "evidence_image": str(evidence_path),
        }
