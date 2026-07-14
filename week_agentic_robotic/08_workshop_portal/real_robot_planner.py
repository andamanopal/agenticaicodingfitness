"""Build a bounded, reviewable joint-space plan for the physical SO-101."""

import hashlib
import json
import re
import sys
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "so101" / "scene.xml"
sys.path.insert(0, str(ROOT / "03_kinematic_pick_place"))
from ik import solve_ik  # noqa: E402
from pick_place import IK_SEED  # noqa: E402
from physical_workspace import profile_digest


def normalized_words(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class RealRobotPlanner:
    """Ground one object and generate one physical plan without executing it."""

    def __init__(
        self,
        semantic_path,
        scan_path,
        workspace_profile,
        joint_limits,
        current_positions,
    ):
        semantic_path = Path(semantic_path)
        self.scene = json.loads(semantic_path.read_text())
        self.scan = json.loads(Path(scan_path).read_text())
        if self.scan.get("source") != "real_so101":
            raise RuntimeError("physical planning requires a scene captured by the real SO-101")
        if self.scene["scene_id"] != self.scan["scan_id"]:
            raise RuntimeError("semantic scene is stale; reconstruct the physical scan again")
        diagnostics_path = semantic_path.parent / "depth" / "diagnostics.json"
        if not diagnostics_path.exists():
            raise RuntimeError("physical reconstruction diagnostics are missing")
        diagnostics = json.loads(diagnostics_path.read_text())
        if (
            diagnostics.get("scene_id") != self.scene["scene_id"]
            or diagnostics.get("background_scan_id")
            != self.scene.get("background_scan_id")
        ):
            raise RuntimeError("physical reconstruction diagnostics are stale")
        unresolved_stages = [
            name
            for name, stage in diagnostics.get("stages", {}).items()
            if stage.get("status") != "complete"
        ]
        if unresolved_stages:
            raise RuntimeError(
                "physical reconstruction needs review before planning: "
                + ", ".join(name.replace("_", " ") for name in unresolved_stages)
            )
        self.reconstruction_stages = diagnostics.get("stages", {})
        self.profile = workspace_profile
        self.profile_matches_scan = (
            self.scan.get("workspace_profile_sha256")
            == profile_digest(workspace_profile)
        )
        if not self.profile_matches_scan:
            raise RuntimeError(
                "physical camera or workspace calibration changed after this scan; "
                "capture the empty table and objects again"
            )
        self.workspace = workspace_profile["workspace"]
        if not self.workspace.get("base_frame_registration_confirmed"):
            raise RuntimeError(
                "confirm the checkerboard and tabletop measurements use the "
                "same robot base frame as the SO-101 model first"
            )
        if not self.workspace.get("kinematic_model_confirmed"):
            raise RuntimeError(
                "confirm the physical joint directions against the SO-101 kinematic model first"
            )
        self.kinematic_validation = workspace_profile.get(
            "kinematic_validation",
            {},
        )
        if self.kinematic_validation.get("status") != "pass":
            raise RuntimeError(
                "the three measured survey poses do not yet validate the physical "
                "joint coordinates against the SO-101 kinematic model"
            )
        self.joint_limits = joint_limits
        self.current_positions = current_positions
        self.objects = {item["id"]: item for item in self.scene["objects"]}
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.scratch = mujoco.MjData(self.model)
        self.plan = None

    def create_plan(self, target, destination):
        if destination not in self.workspace["destinations"]:
            raise ValueError(
                f"destination must be one of {list(self.workspace['destinations'])}"
            )
        obj = self._ground(target)
        pick_xy = np.asarray(obj["position"][:2], dtype=float)
        place_xy = np.asarray(self.workspace["destinations"][destination], dtype=float)
        grasp_z = max(0.01, min(0.035, float(obj["position"][2]) * 0.65))
        hover_z = min(
            float(self.workspace["maximum"][2]) - 0.005,
            max(0.10, grasp_z + 0.06),
        )
        jaw_yaw = float(obj["yaw_radians"]) + np.pi / 2.0
        open_gripper = float(self.workspace["gripper_open_percent"])
        closed_gripper = float(self.workspace["gripper_closed_percent"])
        travel = dict(
            self.profile["travel_pose"]["joint_positions_degrees"]
        )
        travel["gripper"] = open_gripper

        current_arm = np.radians([
            self.current_positions[name]
            for name in (
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
            )
        ])
        reports = []
        steps = []

        def add_joint_step(name, arm_radians, gripper):
            targets = {
                joint: round(float(np.degrees(arm_radians[index])), 5)
                for index, joint in enumerate((
                    "shoulder_pan",
                    "shoulder_lift",
                    "elbow_flex",
                    "wrist_flex",
                    "wrist_roll",
                ))
            }
            targets["gripper"] = round(float(gripper), 5)
            steps.append({"name": name, "targets": targets})

        def cartesian_path(name, xy, start_z, end_z, seed, gripper, yaw=None):
            count = max(2, int(np.ceil(abs(end_z - start_z) / 0.015)) + 1)
            current = seed
            for index, z in enumerate(np.linspace(start_z, end_z, count)[1:], start=1):
                current, report = self._solve([xy[0], xy[1], z], current, yaw)
                report["phase"] = name
                reports.append(report)
                add_joint_step(f"{name} {index}/{count - 1}", current, gripper)
            return current

        steps.append({"name": "open gripper", "targets": {
            **self.current_positions,
            "gripper": open_gripper,
        }})
        steps.append({"name": "safe travel", "targets": travel})
        travel_pick = dict(travel)
        travel_pick["shoulder_pan"] = round(
            float(np.degrees(np.arctan2(pick_xy[1], pick_xy[0]))),
            5,
        )
        steps.append({"name": "turn above pick", "targets": travel_pick})

        seed, report = self._solve([*pick_xy, hover_z], current_arm, jaw_yaw)
        report["phase"] = "pick hover"
        reports.append(report)
        add_joint_step("pick hover", seed, open_gripper)
        seed = cartesian_path(
            "lower to object",
            pick_xy,
            hover_z,
            grasp_z,
            seed,
            open_gripper,
            jaw_yaw,
        )
        add_joint_step("close gripper", seed, closed_gripper)
        seed = cartesian_path(
            "lift object",
            pick_xy,
            grasp_z,
            hover_z,
            seed,
            closed_gripper,
            jaw_yaw,
        )

        travel_with_object = dict(travel)
        travel_with_object["gripper"] = closed_gripper
        steps.append({"name": "safe carry", "targets": travel_with_object})
        travel_place = dict(travel_with_object)
        travel_place["shoulder_pan"] = round(
            float(np.degrees(np.arctan2(place_xy[1], place_xy[0]))),
            5,
        )
        steps.append({"name": "turn above destination", "targets": travel_place})
        seed, report = self._solve([*place_xy, hover_z], seed, None)
        report["phase"] = "place hover"
        reports.append(report)
        add_joint_step("place hover", seed, closed_gripper)
        seed = cartesian_path(
            "lower to table",
            place_xy,
            hover_z,
            grasp_z + 0.004,
            seed,
            closed_gripper,
        )
        add_joint_step("release object", seed, open_gripper)
        seed = cartesian_path(
            "retreat",
            place_xy,
            grasp_z + 0.004,
            hover_z,
            seed,
            open_gripper,
        )
        steps.append({"name": "safe travel", "targets": travel})

        target_radius = max(obj["dimensions"][:2]) / 2.0
        clearances = []
        approach_clearances = []
        for other in self.objects.values():
            if other["id"] == obj["id"]:
                continue
            pick_distance = float(np.linalg.norm(
                pick_xy - np.asarray(other["position"][:2], dtype=float)
            ))
            pick_required = (
                target_radius
                + max(other["dimensions"][:2]) / 2.0
                + 0.02
            )
            approach_clearances.append({
                "object_id": other["id"],
                "distance_meters": round(pick_distance, 4),
                "required_meters": round(pick_required, 4),
                "clear": pick_distance >= pick_required,
            })
            distance = float(np.linalg.norm(
                place_xy - np.asarray(other["position"][:2], dtype=float)
            ))
            required = target_radius + max(other["dimensions"][:2]) / 2.0 + 0.015
            clearances.append({
                "object_id": other["id"],
                "distance_meters": round(distance, 4),
                "required_meters": round(required, 4),
                "clear": distance >= required,
            })

        (
            collision_free,
            collision_records,
            minimum_carried_object_bottom,
        ) = self._trajectory_collision_check(steps, grasp_z)
        checks = {
            "physical_scene": self.scan.get("source") == "real_so101",
            "camera_profile_matches_scan": self.profile_matches_scan,
            "all_waypoints_reachable": all(report["reachable"] for report in reports),
            "all_joint_targets_within_calibration": self._steps_within_limits(steps),
            "all_joint_targets_within_robot_model": self._steps_within_model_limits(steps),
            "sampled_robot_table_and_self_collision_free": collision_free,
            "carried_object_stays_above_table": (
                minimum_carried_object_bottom >= -0.003
            ),
            "object_fits_gripper": float(obj["dimensions"][1]) <= 0.06,
            "pick_approach_clear": all(item["clear"] for item in approach_clearances),
            "destination_clear": all(item["clear"] for item in clearances),
            "observation_quality_passed": (
                self.scan["observation_quality"]["status"] == "basic_check_passed"
            ),
            "semantic_known": obj["label"].lower() != "unknown",
            "semantic_confident": float(obj["semantic_confidence"]) >= 0.6,
            "geometry_confident": float(obj["geometry_confidence"]) >= 0.6,
            "geometry_has_two_views": len(obj["evidence_views"]) >= 2,
            "geometry_not_clipped": not obj.get("boundary_flags"),
            "reconstruction_diagnostics_passed": all(
                stage.get("status") == "complete"
                for stage in self.reconstruction_stages.values()
            ),
            "base_frame_registration_confirmed": bool(
                self.workspace.get("base_frame_registration_confirmed")
            ),
            "kinematic_model_confirmed": bool(
                self.workspace.get("kinematic_model_confirmed")
            ),
            "measured_kinematic_consistency_passed": (
                self.kinematic_validation.get("status") == "pass"
            ),
        }
        plan_fingerprint = hashlib.sha256(json.dumps({
            "scene_id": self.scene["scene_id"],
            "object_id": obj["id"],
            "destination": destination,
            "motion_steps": steps,
        }, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        plan_id = f"physical:{plan_fingerprint}"
        self.plan = {
            "status": "planned",
            "plan_id": plan_id,
            "scene_id": self.scene["scene_id"],
            "object_id": obj["id"],
            "object_label": obj["label"],
            "destination": destination,
            "pick_xyz": [round(float(value), 4) for value in [*pick_xy, grasp_z]],
            "place_xyz": [round(float(value), 4) for value in [*place_xy, grasp_z + 0.004]],
            "planned_from_positions": {
                name: round(float(value), 5)
                for name, value in self.current_positions.items()
            },
            "safe": all(checks.values()),
            "checks": checks,
            "approach_clearances": approach_clearances,
            "clearances": clearances,
            "collision_records": collision_records,
            "minimum_carried_object_bottom_meters": round(
                minimum_carried_object_bottom,
                4,
            ),
            "ik_waypoints": reports,
            "motion_steps": steps,
            "requires_physical_approval": True,
            "not_checked": [
                "camera occlusion and object coverage beyond brightness/contrast",
                "residual camera motion blur after the settle hold",
                "collision with unmodeled cables, clamps, or room objects",
                "collision between the carried object and unmodeled surroundings",
                "grasp contact force",
                "object slip during carry",
            ],
        }
        return self.plan

    def _ground(self, query):
        query_words = normalized_words(query)
        matches = []
        for obj in self.objects.values():
            best = 0
            for name in [obj["label"], *obj.get("aliases", [])]:
                score = len(query_words & normalized_words(name))
                if query.lower() == name.lower():
                    score += 3
                best = max(best, score)
            if best:
                matches.append((best, obj))
        matches.sort(key=lambda item: item[0], reverse=True)
        if not matches:
            raise RuntimeError(f"{query!r} does not match an object in physical scene memory")
        if len(matches) > 1 and matches[0][0] == matches[1][0]:
            raise RuntimeError(f"{query!r} matches more than one physical object")
        return matches[0][1]

    def _solve(self, target, seed, jaw_yaw):
        target = np.asarray(target, dtype=float)
        pan_seed = IK_SEED.copy()
        pan_seed[0] = np.arctan2(target[1], target[0])
        candidates = [
            solve_ik(self.model, self.scratch, target, initial)
            for initial in (seed, IK_SEED, pan_seed)
        ]
        q, residual = min(
            candidates,
            key=lambda result: result[1] + 0.002 * np.linalg.norm(result[0] - seed),
        )
        wrist_error = 0.0
        if jaw_yaw is not None:
            self.scratch.qpos[:] = 0.0
            self.scratch.qpos[:5] = q
            mujoco.mj_forward(self.model, self.scratch)
            fixed = self.scratch.geom("fixed_jaw_sph_tip1").xpos[:2]
            moving = self.scratch.geom("moving_jaw_sph_tip1").xpos[:2]
            current_yaw = np.arctan2(*(moving - fixed)[::-1])
            delta = (jaw_yaw - current_yaw + np.pi) % (2 * np.pi) - np.pi
            if delta > np.pi / 2:
                delta -= np.pi
            elif delta < -np.pi / 2:
                delta += np.pi
            requested = q[4] + delta
            lower, upper = self.model.jnt_range[self.model.joint("wrist_roll").id]
            q[4] = np.clip(requested, lower, upper)
            wrist_error = abs(float(q[4] - requested))
        reachable = bool(residual <= 0.025 and wrist_error <= 0.05)
        return q, {
            "target_xyz": np.round(target, 4).tolist(),
            "residual_meters": round(float(residual), 6),
            "wrist_limit_error_radians": round(wrist_error, 6),
            "reachable": reachable,
        }

    def _steps_within_limits(self, steps):
        for step in steps:
            for joint, target in step["targets"].items():
                limit = self.joint_limits[joint]
                if not limit["min"] <= float(target) <= limit["max"]:
                    return False
        return True

    def _steps_within_model_limits(self, steps):
        for step in steps:
            for joint in (
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
            ):
                joint_id = self.model.joint(joint).id
                lower, upper = self.model.jnt_range[joint_id]
                value = np.radians(float(step["targets"][joint]))
                if not lower <= value <= upper:
                    return False
        return True

    def _trajectory_collision_check(self, steps, grasp_z):
        collisions = []
        previous = None
        holding_object = False
        minimum_object_bottom = float("inf")
        gripper_site = self.model.site("gripperframe").id
        for step in steps:
            target = step["targets"]
            if previous is None:
                samples = [target]
            else:
                body_delta = max(
                    abs(float(target[name]) - float(previous[name]))
                    for name in (
                        "shoulder_pan",
                        "shoulder_lift",
                        "elbow_flex",
                        "wrist_flex",
                        "wrist_roll",
                    )
                )
                gripper_delta = abs(
                    float(target["gripper"]) - float(previous["gripper"])
                )
                count = max(
                    1,
                    int(np.ceil(body_delta / 2.0)),
                    int(np.ceil(gripper_delta / 5.0)),
                )
                samples = [{
                    name: float(previous[name])
                    + (float(target[name]) - float(previous[name])) * index / count
                    for name in target
                } for index in range(1, count + 1)]

            for sample in samples:
                self.scratch.qpos[:] = 0.0
                for index, name in enumerate((
                    "shoulder_pan",
                    "shoulder_lift",
                    "elbow_flex",
                    "wrist_flex",
                    "wrist_roll",
                )):
                    self.scratch.qpos[index] = np.radians(float(sample[name]))
                gripper_joint = self.model.joint("gripper")
                lower, upper = self.model.jnt_range[gripper_joint.id]
                self.scratch.qpos[5] = lower + (
                    np.clip(float(sample["gripper"]), 0.0, 100.0) / 100.0
                ) * (upper - lower)
                mujoco.mj_forward(self.model, self.scratch)
                if holding_object:
                    object_bottom = float(
                        self.scratch.site_xpos[gripper_site][2] - grasp_z
                    )
                    minimum_object_bottom = min(
                        minimum_object_bottom,
                        object_bottom,
                    )
                for contact in self.scratch.contact[:self.scratch.ncon]:
                    if contact.dist >= 0.0:
                        continue
                    first = mujoco.mj_id2name(
                        self.model,
                        mujoco.mjtObj.mjOBJ_GEOM,
                        int(contact.geom1),
                    )
                    second = mujoco.mj_id2name(
                        self.model,
                        mujoco.mjtObj.mjOBJ_GEOM,
                        int(contact.geom2),
                    )
                    collisions.append({
                        "step": step["name"],
                        "geoms": [first, second],
                        "penetration_meters": round(float(-contact.dist), 6),
                    })
                    if len(collisions) >= 12:
                        if not np.isfinite(minimum_object_bottom):
                            minimum_object_bottom = -1.0
                        return False, collisions, minimum_object_bottom
            previous = target
            if step["name"] == "close gripper":
                holding_object = True
            elif step["name"] == "release object":
                holding_object = False
        if not np.isfinite(minimum_object_bottom):
            minimum_object_bottom = -1.0
        return not collisions, collisions, minimum_object_bottom
