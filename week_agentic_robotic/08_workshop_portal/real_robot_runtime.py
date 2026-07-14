"""Safety-gated LeRobot runtime for a physical SO-101 follower."""

import json
import math
import queue
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from hardware_setup import normalized_joint_limits
from physical_workspace import PhysicalWorkspace, VIEWPOINTS
from robot_runtime import JOINT_NAMES, jpeg_bytes


RECORDINGS_DIR = (
    Path(__file__).resolve().parents[1]
    / "09_real_robot_setup"
    / "output"
    / "teleop_recordings"
)


BODY_JOG_SPEED_DEGREES = 20.0
GRIPPER_JOG_SPEED_PERCENT = 40.0
MANUAL_CONTROL_LEASE_SECONDS = 0.30
AUTONOMOUS_BODY_SPEED_DEGREES = 8.0
AUTONOMOUS_GRIPPER_SPEED_PERCENT = 18.0
BODY_TRACKING_TOLERANCE_DEGREES = 1.5
GRIPPER_TRACKING_TOLERANCE_PERCENT = 8.0
CONTROL_LOOP_HZ = 50.0
MAX_CONTROL_STEP_SECONDS = 0.05
# During teleop, keep the leader->follower loop tight. Read follower state and
# the wrist camera at a low rate so serial reads and JPEG encoding (which holds
# the GIL) do not add latency between the leader moving and the follower moving.
TELEOP_STATE_READ_HZ = 15.0
TELEOP_CAMERA_HZ = 12.0


def lerobot_classes():
    try:
        from lerobot.cameras.opencv import OpenCVCameraConfig
        from lerobot.robots.so101_follower.config_so101_follower import (
            SO101FollowerConfig,
        )
        from lerobot.robots.so101_follower.so101_follower import SO101Follower
    except ImportError:
        try:
            from lerobot.cameras.opencv import OpenCVCameraConfig
            from lerobot.robots.so_follower import (
                SO101Follower,
                SO101FollowerConfig,
            )
        except ImportError as error:
            raise RuntimeError(
                "LeRobot with Feetech support is not installed. "
                "Install this repository's requirements.txt in its .venv."
            ) from error
    return SO101Follower, SO101FollowerConfig, OpenCVCameraConfig


def leader_classes():
    try:
        from lerobot.teleoperators.so_leader import (
            SO101Leader,
            SO101LeaderConfig,
        )
    except ImportError:
        try:
            from lerobot.teleoperators.so101_leader import (
                SO101Leader,
                SO101LeaderConfig,
            )
        except ImportError as error:
            raise RuntimeError(
                "LeRobot SO-101 leader teleoperator is not available in this install."
            ) from error
    return SO101Leader, SO101LeaderConfig


class RealRobotRuntime:
    """Own all physical I/O on one thread and gate every form of motion."""

    def __init__(self, frame_hub, simulation_runtime):
        self.frame_hub = frame_hub
        self.simulation_runtime = simulation_runtime
        self.workspace = PhysicalWorkspace()
        self.robot = None
        self.thread = None
        self.running = threading.Event()
        self.emergency_stop = threading.Event()
        self.commands = queue.Queue()
        self.lock = threading.Lock()
        self.held = set()
        self.manual_lease_deadline = 0.0
        self.targets = {}
        self.positions = {}
        self.position_history = deque(maxlen=60)
        self.latest_frame = None
        self.joint_limits = {}
        self.limit_hits = {}
        self.connected = False
        self.armed = False
        self.scan_approved = False
        self.approved_plan_id = None
        self.autonomous_active = False
        self.leader = None
        self.leader_config = None
        self.teleop_active = False
        self.recording = False
        self.recorded_frames = []
        self.last_recording = None
        self.status = "MuJoCo active"
        self.error = None
        self.config = None

    def connect(self, port, robot_id, camera_index, width, height, fps, rotation):
        if self.connected:
            raise RuntimeError("the real robot is already connected")
        if self.thread is not None:
            raise RuntimeError("a real-robot connection is already starting")

        SO101Follower, SO101FollowerConfig, OpenCVCameraConfig = lerobot_classes()
        camera_rotation = {
            90: -90,
            270: 90,
        }.get(rotation, rotation)
        camera = OpenCVCameraConfig(
            index_or_path=camera_index,
            width=width,
            height=height,
            fps=fps,
            rotation=camera_rotation,
        )
        config = SO101FollowerConfig(
            port=port,
            id=robot_id,
            cameras={"wrist": camera},
            max_relative_target=4.0,
            disable_torque_on_disconnect=True,
            use_degrees=True,
        )
        robot = SO101Follower(config)
        if not robot.calibration:
            raise RuntimeError(
                f"no LeRobot calibration found for {robot_id!r}; run the guided "
                "LeRobot follower calibration with the same ID first"
            )
        joint_limits = normalized_joint_limits(robot.calibration)

        self.status = "Connecting to physical SO-101"
        try:
            robot.bus.connect()

            # Match every servo's hold target to its exact raw position before
            # LeRobot briefly releases torque to configure the motor bus.
            raw_positions = robot.bus.sync_read(
                "Present_Position",
                normalize=False,
                num_retry=3,
            )
            for motor, raw_position in raw_positions.items():
                robot.bus.write(
                    "Goal_Position",
                    motor,
                    raw_position,
                    normalize=False,
                    num_retry=3,
                )

            if not robot.is_calibrated:
                raise RuntimeError(
                    "the saved LeRobot calibration does not match the connected "
                    "follower; run follower calibration with this robot ID first"
                )

            robot.configure()
            for wrist_camera in robot.cameras.values():
                wrist_camera.connect()
            observation = robot.get_observation()
        except Exception as error:
            cleanup_errors = []
            for wrist_camera in robot.cameras.values():
                if wrist_camera.is_connected:
                    try:
                        wrist_camera.disconnect()
                    except Exception as cleanup_error:
                        cleanup_errors.append(f"camera cleanup failed: {cleanup_error}")
            if robot.bus.is_connected:
                try:
                    robot.bus.disconnect(disable_torque=True)
                except Exception as cleanup_error:
                    cleanup_errors.append(f"motor torque cleanup failed: {cleanup_error}")
            detail = f"could not connect to the physical SO-101: {error}"
            if cleanup_errors:
                detail += "; " + "; ".join(cleanup_errors)
            raise RuntimeError(detail) from error
        positions = self._positions(observation)
        if len(positions) != len(JOINT_NAMES):
            robot.disconnect()
            raise RuntimeError("LeRobot did not return all six SO-101 joint positions")

        camera_signature = {
            "index": int(camera_index),
            "width": int(width),
            "height": int(height),
            "rotation_ccw_degrees": int(rotation),
        }
        self.workspace.activate(robot_id, camera_signature)
        self.robot = robot
        self.targets = positions
        self.positions = positions
        self.position_history.clear()
        self.position_history.append((time.monotonic(), dict(positions)))
        self.latest_frame = np.asarray(observation["wrist"]).copy()
        self.joint_limits = joint_limits
        self.limit_hits = {}
        self.connected = True
        self.armed = False
        self.scan_approved = False
        self.approved_plan_id = None
        self.autonomous_active = False
        self.error = None
        self.config = {
            "port": port,
            "robot_id": robot_id,
            "camera_index": camera_index,
            "width": width,
            "height": height,
            "fps": fps,
            "rotation": rotation,
        }
        self.simulation_runtime.start_mirror(positions)
        self.emergency_stop.clear()
        self.commands = queue.Queue()
        self.manual_lease_deadline = 0.0
        self.running.set()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        self.status = "Real SO-101 connected · controls locked · torque holding pose"

    def disconnect(self):
        if self.teleop_active or self.leader is not None:
            self.stop_teleop()
        self.emergency_stop.set()
        self.disarm()
        self.running.clear()
        thread = self.thread
        if thread is not None:
            thread.join(timeout=3)
        robot = self.robot
        disconnect_errors = []
        if robot is not None:
            for wrist_camera in robot.cameras.values():
                if wrist_camera.is_connected:
                    try:
                        wrist_camera.disconnect()
                    except Exception as error:
                        disconnect_errors.append(f"camera cleanup failed: {error}")
            if robot.bus.is_connected:
                try:
                    robot.bus.disconnect(disable_torque=True)
                except Exception as error:
                    disconnect_errors.append(
                        f"motor torque cleanup failed: {error}"
                    )
        if thread is not None and thread.is_alive():
            thread.join(timeout=3)
        thread_stuck = bool(thread is not None and thread.is_alive())
        self.thread = thread if thread_stuck else None
        self.robot = robot if thread_stuck else None
        self.connected = False
        self.positions = {}
        self.position_history.clear()
        self.targets = {}
        self.manual_lease_deadline = 0.0
        self.latest_frame = None
        self.joint_limits = {}
        self.limit_hits = {}
        self.scan_approved = False
        self.approved_plan_id = None
        self.autonomous_active = False
        self.workspace.deactivate()
        self.simulation_runtime.stop_mirror()
        if thread_stuck:
            self.error = "physical I/O thread did not stop after disconnect"
            self.status = f"Real robot error · {self.error}"
            raise RuntimeError(self.error)
        if disconnect_errors:
            self.error = "robot disconnected with an error: " + "; ".join(
                disconnect_errors
            )
            self.status = f"Real robot error · {self.error}"
            raise RuntimeError(self.error)
        self.config = None
        self.error = None
        self.status = "MuJoCo active"

    def arm(self):
        self._require_operational()
        if self.autonomous_active:
            raise RuntimeError("scripted motion is active")
        with self.lock:
            if len(self.targets) != len(JOINT_NAMES):
                raise RuntimeError("joint state is not ready")
            if len(self.joint_limits) != len(JOINT_NAMES):
                raise RuntimeError("calibrated joint limits are not ready")
            self.held.clear()
            self.manual_lease_deadline = 0.0
            self.scan_approved = False
            self.approved_plan_id = None
            self.armed = True
        self.status = "Real SO-101 connected · manual control armed"

    def disarm(self):
        with self.lock:
            self.held.clear()
            self.manual_lease_deadline = 0.0
            self.armed = False
            self.scan_approved = False
            self.approved_plan_id = None
        if self.connected and not self.autonomous_active:
            self.status = "Real SO-101 connected · controls locked · torque holding pose"

    def start_teleop(self, leader_port, leader_id):
        """Connect the leader arm; the follower mirrors it in the control loop."""
        self._require_operational()
        if self.autonomous_active:
            raise RuntimeError("wait for scripted motion to stop before teleoperating")
        if self.teleop_active:
            raise RuntimeError("teleoperation is already active")
        SO101Leader, SO101LeaderConfig = leader_classes()
        leader = SO101Leader(
            SO101LeaderConfig(port=leader_port, id=leader_id, use_degrees=True)
        )
        leader.connect()
        if not leader.is_connected:
            raise RuntimeError("the leader arm did not connect")
        if not leader.is_calibrated:
            leader.disconnect()
            raise RuntimeError(
                f"no LeRobot calibration found for leader {leader_id!r}; "
                "calibrate the leader with that ID first"
            )
        with self.lock:
            self.held.clear()
            self.manual_lease_deadline = 0.0
            self.armed = False
            self.scan_approved = False
            self.approved_plan_id = None
            self.leader = leader
            self.leader_config = {"port": leader_port, "robot_id": leader_id}
            self.teleop_active = True
            self.recording = False
            self.recorded_frames = []
            self.targets = dict(self.positions)
        # Match the standalone teleop scripts: let the follower track the leader
        # directly. Dropping the per-command slew cap removes both the trailing
        # lag and an extra present-position serial read on every send_action.
        # The joint-limit clamp in _teleop_action still bounds every target.
        self.robot.config.max_relative_target = None
        self.status = "Leader teleoperation active · follower mirrors the leader"

    def stop_teleop(self):
        with self.lock:
            leader = self.leader
            was_recording = self.recording
            self.teleop_active = False
            self.recording = False
            self.leader = None
        saved = self._save_recording() if was_recording else None
        if leader is not None:
            try:
                leader.disconnect()
            except Exception:
                pass
        # Restore the per-command slew cap for autonomous (scan/plan) motion.
        if self.robot is not None:
            self.robot.config.max_relative_target = 4.0
        with self.lock:
            if self.connected:
                self.targets = dict(self.positions)
        if self.connected and not self.autonomous_active:
            self.status = "Real SO-101 connected · controls locked · torque holding pose"
        return saved

    def start_recording(self):
        with self.lock:
            if not self.teleop_active:
                raise RuntimeError("start teleoperation before recording")
            self.recorded_frames = []
            self.recording = True
        self.status = "Leader teleoperation · recording trajectory"
        return {"recording": True}

    def stop_recording(self):
        with self.lock:
            if not self.recording:
                raise RuntimeError("no recording is in progress")
            self.recording = False
        result = self._save_recording()
        if self.teleop_active:
            self.status = "Leader teleoperation active · follower mirrors the leader"
        return result

    def _save_recording(self):
        with self.lock:
            frames = list(self.recorded_frames)
            self.recorded_frames = []
        if len(frames) < 2:
            return {"saved": False, "reason": "recording was too short"}
        start_time = frames[0][0]
        payload = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "joint_names": list(JOINT_NAMES),
            "frames": [
                {
                    "t": round(timestamp - start_time, 4),
                    "positions": {
                        joint: round(float(value), 4)
                        for joint, value in positions.items()
                    },
                }
                for timestamp, positions in frames
            ],
        }
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        name = datetime.now(timezone.utc).strftime("teleop_%Y%m%dT%H%M%S")
        (RECORDINGS_DIR / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n")
        self.last_recording = name
        return {
            "saved": True,
            "name": name,
            "frames": len(frames),
            "duration_seconds": round(frames[-1][0] - start_time, 2),
        }

    def list_recordings(self):
        if not RECORDINGS_DIR.exists():
            return []
        recordings = []
        for path in sorted(RECORDINGS_DIR.glob("teleop_*.json"), reverse=True):
            try:
                data = json.loads(path.read_text())
                frames = data.get("frames", [])
            except (OSError, json.JSONDecodeError):
                continue
            recordings.append({
                "name": path.stem,
                "frames": len(frames),
                "duration_seconds": frames[-1]["t"] if frames else 0.0,
            })
        return recordings

    def delete_recording(self, name):
        recordings_dir = RECORDINGS_DIR.resolve()
        path = (recordings_dir / f"{name}.json").resolve()
        if path.parent != recordings_dir or not path.name.startswith("teleop_"):
            raise RuntimeError("invalid recording name")
        if not path.exists():
            raise RuntimeError("recording not found")
        path.unlink()
        if self.last_recording == name:
            self.last_recording = None
        return {"deleted": name}

    def replay(self, name):
        self._require_operational()
        if self.teleop_active:
            raise RuntimeError("stop teleoperation before replaying")
        if self.autonomous_active:
            raise RuntimeError("wait for the current motion to stop")
        path = RECORDINGS_DIR / f"{name}.json"
        if not path.exists():
            raise RuntimeError("recording not found")
        frames = json.loads(path.read_text()).get("frames", [])
        if len(frames) < 2:
            raise RuntimeError("recording is too short to replay")
        return self._submit_motion("replay", frames)

    def lock_manual_control(self):
        """Revoke manual control without consuming a scan or plan approval."""
        with self.lock:
            self.held.clear()
            self.manual_lease_deadline = 0.0
            self.armed = False
        if (
            self.connected
            and not self.autonomous_active
            and not self.scan_approved
            and self.approved_plan_id is None
        ):
            self.status = "Real SO-101 connected · controls locked · torque holding pose"

    def approve_scan(self):
        self._require_operational()
        if not self.workspace.snapshot()["scan_ready"]:
            raise RuntimeError("finish physical camera and survey-pose calibration first")
        with self.lock:
            self.held.clear()
            self.manual_lease_deadline = 0.0
            self.armed = False
            self.approved_plan_id = None
            self.scan_approved = True
        self.status = "One physical three-view scan approved"

    def approve_plan(self, plan_id):
        self._require_operational()
        if not plan_id:
            raise ValueError("plan ID is required")
        if not self.workspace.snapshot()["execution_ready"]:
            raise RuntimeError("physical kinematic-model confirmation is incomplete")
        with self.lock:
            self.held.clear()
            self.manual_lease_deadline = 0.0
            self.armed = False
            self.scan_approved = False
            self.approved_plan_id = plan_id
        self.status = "One physical manipulation plan approved"

    def set_control(self, joint, direction, active):
        if joint not in JOINT_NAMES or direction not in (-1, 1):
            raise ValueError("unknown manual control")
        with self.lock:
            control = (joint, direction)
            if active and not self.armed:
                raise ValueError("arm manual control before moving the robot")
            if active:
                self.held.add(control)
                self.manual_lease_deadline = (
                    time.monotonic() + MANUAL_CONTROL_LEASE_SECONDS
                )
            else:
                self.held.discard(control)
                if not self.held:
                    self.manual_lease_deadline = 0.0

    def refresh_manual_control_lease(self):
        with self.lock:
            if self.connected and self.armed and self.held:
                self.manual_lease_deadline = (
                    time.monotonic() + MANUAL_CONTROL_LEASE_SECONDS
                )

    def release_all(self):
        with self.lock:
            self.held.clear()
            self.manual_lease_deadline = 0.0

    def calibration_frame(self):
        self._require_operational()
        with self.lock:
            if self.latest_frame is None:
                raise RuntimeError("the wrist camera has not produced a frame yet")
            return self.latest_frame.copy(), dict(self.positions)

    def add_intrinsic_sample(self):
        frame, _ = self.calibration_frame()
        return self.workspace.add_intrinsic_sample(frame)

    def solve_intrinsics(self):
        return self.workspace.solve_intrinsics()

    def record_travel_pose(self):
        self.release_all()
        self._require_stable_pose()
        _, positions = self.calibration_frame()
        return self.workspace.record_travel_pose(positions)

    def record_viewpoint(self, name, **board_pose):
        self.release_all()
        self._require_stable_pose()
        frame, positions = self.calibration_frame()
        return self.workspace.record_viewpoint(
            name,
            frame,
            positions,
            **board_pose,
        )

    def save_workspace(self, workspace):
        return self.workspace.update_workspace(workspace)

    def run_scan(self, on_capture=None):
        with self.lock:
            if not self.scan_approved:
                raise RuntimeError(
                    "physical scan motion is locked; approve one scan in the safety bar"
                )
            self.scan_approved = False
        profile = self.workspace.scan_profile()
        travel = profile["travel_pose"]["joint_positions_degrees"]
        steps = [{"name": "safe travel", "targets": travel}]
        for name in VIEWPOINTS:
            steps.extend([
                {
                    "name": name,
                    "targets": profile["viewpoints"][name][
                        "joint_positions_degrees"
                    ],
                    "settle_seconds": 0.7,
                    "capture": name,
                },
                {"name": "safe travel", "targets": travel},
            ])
        return self._submit_motion("scan", steps, on_capture=on_capture)

    def run_plan(self, plan_id, steps):
        with self.lock:
            if self.approved_plan_id != plan_id:
                raise RuntimeError(
                    "physical motion is locked; approve this exact plan in the portal"
                )
            self.approved_plan_id = None
        return self._submit_motion("manipulation", steps, plan_id=plan_id)

    def snapshot(self):
        workspace = self.workspace.snapshot()
        with self.lock:
            return {
                "mode": "real" if self.connected else "simulation",
                "connected": self.connected,
                "armed": self.armed,
                "scan_approved": self.scan_approved,
                "approved_plan_id": self.approved_plan_id,
                "autonomous_active": self.autonomous_active,
                "teleop_active": self.teleop_active,
                "recording": self.recording,
                "recorded_frames": len(self.recorded_frames),
                "last_recording": self.last_recording,
                "leader_config": self.leader_config,
                "status": self.status,
                "error": self.error,
                "config": self.config,
                "positions": dict(self.positions),
                "joint_limits": dict(self.joint_limits),
                "limit_hits": dict(self.limit_hits),
                "workspace": workspace,
                "capabilities": {
                    "wrist_camera": self.connected,
                    "manual_jog": self.connected,
                    "teleop": self.connected,
                    "autonomous_scan": self.connected and workspace["scan_ready"],
                    "autonomous_execute": self.connected and workspace["execution_ready"],
                },
            }

    def _submit_motion(self, kind, steps, plan_id=None, on_capture=None):
        self._require_operational()
        task = {
            "kind": kind,
            "steps": steps,
            "plan_id": plan_id,
            "on_capture": on_capture,
            "done": threading.Event(),
            "result": None,
            "error": None,
        }
        self.commands.put(task)
        if not task["done"].wait(timeout=240):
            self.emergency_stop.set()
            raise RuntimeError("physical motion timed out and was stopped")
        if task["error"]:
            raise RuntimeError(task["error"])
        return task["result"]

    def _run(self):
        control_period = 1.0 / CONTROL_LOOP_HZ
        publish_period = 1.0 / self.config["fps"]
        last_time = time.monotonic()
        last_publish = 0.0
        last_position_read = 0.0
        active_task = None
        try:
            while self.running.is_set():
                try:
                    active_task = self.commands.get_nowait()
                except queue.Empty:
                    active_task = None
                if active_task is not None:
                    self._execute_task(active_task)
                    active_task = None
                    last_time = time.monotonic()
                    continue

                started = time.monotonic()
                elapsed = min(started - last_time, MAX_CONTROL_STEP_SECONDS)
                last_time = started

                # Command the servos every control tick, but read the wrist
                # camera and JPEG-encode it on the slower publish cadence. The
                # arm no longer advances one goal per camera frame, so manual
                # control is smooth instead of stepping at the camera frame rate.
                with self.lock:
                    teleop = self.teleop_active
                    leader = self.leader
                    recording = self.recording

                # Teleop fast path: a tight leader->follower loop. Read follower
                # state and the camera at a low rate so serial reads and JPEG
                # encoding (which holds the GIL) never sit between the leader
                # moving and the follower moving. Record the clamped target so
                # replay never trips the software-limit check.
                if teleop and leader is not None:
                    teleop_action = self._teleop_action(leader)
                    if teleop_action is not None:
                        self.robot.send_action(teleop_action)
                        if recording:
                            with self.lock:
                                self.recorded_frames.append((started, dict(self.targets)))
                    if started - last_position_read >= 1.0 / TELEOP_STATE_READ_HZ:
                        self._read_positions()
                        last_position_read = started
                    if started - last_publish >= 1.0 / TELEOP_CAMERA_HZ:
                        self._publish_frame()
                        last_publish = started
                    time.sleep(max(0.0, control_period - (time.monotonic() - started)))
                    continue

                self._read_positions()
                last_position_read = started
                if started - last_publish >= publish_period:
                    self._publish_frame()
                    last_publish = started

                action = None
                with self.lock:
                    if not self.armed:
                        self.targets = dict(self.positions)
                    if (
                        self.held
                        and started > self.manual_lease_deadline
                    ):
                        self.held.clear()
                        self.manual_lease_deadline = 0.0
                        self.status = (
                            "Real SO-101 connected · manual control stopped · "
                            "control signal lost"
                        )
                    held = tuple(self.held)
                    if self.armed and held:
                        for joint, direction in held:
                            speed = (
                                GRIPPER_JOG_SPEED_PERCENT
                                if joint == "gripper"
                                else BODY_JOG_SPEED_DEGREES
                            )
                            proposed = self.targets[joint] + direction * speed * elapsed
                            bounded, at_limit = self._bounded_target(
                                joint,
                                self.targets[joint],
                                proposed,
                            )
                            self.targets[joint] = bounded
                            if at_limit:
                                self.limit_hits[joint] = direction
                            else:
                                self.limit_hits.pop(joint, None)
                        action = self._action(self.targets)
                if action is not None:
                    self.robot.send_action(action)

                time.sleep(max(0.0, control_period - (time.monotonic() - started)))
        except Exception as error:
            self.error = str(error)
            self.status = f"Real robot error · {error}"
            self.emergency_stop.set()
            self.disarm()
            self.running.clear()
            if active_task is not None:
                active_task["error"] = str(error)
                active_task["done"].set()
        finally:
            while True:
                try:
                    pending = self.commands.get_nowait()
                except queue.Empty:
                    break
                pending["error"] = "physical robot stopped before the motion began"
                pending["done"].set()

    def _teleop_action(self, leader):
        try:
            leader_action = leader.get_action()
        except Exception as error:
            self.status = f"Teleoperation stopped · leader read failed · {error}"
            self.stop_teleop()
            return None
        target = {}
        for joint in JOINT_NAMES:
            key = f"{joint}.pos"
            if key not in leader_action:
                return None
            limits = self.joint_limits[joint]
            target[joint] = min(
                limits["max"],
                max(limits["min"], float(leader_action[key])),
            )
        with self.lock:
            self.targets.update(target)
        return self._action(target)

    def _execute_task(self, task):
        captures = {}
        with self.lock:
            self.held.clear()
            self.armed = False
            self.autonomous_active = True
        try:
            if task["kind"] == "replay":
                self._replay_frames(task["steps"])
                task["result"] = {
                    "status": "completed",
                    "kind": "replay",
                    "plan_id": None,
                    "captures": {},
                }
                self.status = "Replay complete · controls locked · torque holding pose"
                return
            for index, step in enumerate(task["steps"], start=1):
                if self.emergency_stop.is_set() or not self.running.is_set():
                    raise RuntimeError("physical motion stopped")
                self.status = (
                    f"Physical {task['kind']} · {step['name']} "
                    f"({index}/{len(task['steps'])})"
                )
                self._move_to(step["targets"])
                self._settle(step.get("settle_seconds", 0.25))
                capture_name = step.get("capture")
                if capture_name:
                    frame, positions = self.calibration_frame()
                    captures[capture_name] = {
                        "rgb": self.workspace.undistort(frame),
                        "joint_positions_degrees": positions,
                    }
                    if task["on_capture"] is not None:
                        task["on_capture"](capture_name, captures[capture_name])
            task["result"] = {
                "status": "completed",
                "kind": task["kind"],
                "plan_id": task["plan_id"],
                "captures": captures,
            }
            self.status = (
                f"Physical {task['kind']} complete · controls locked · "
                "torque holding pose"
            )
        except Exception as error:
            task["error"] = str(error)
            self.status = f"Physical motion stopped · {error}"
        finally:
            with self.lock:
                self.autonomous_active = False
                self.targets = dict(self.positions)
            task["done"].set()

    def _clamp_positions(self, positions):
        return {
            joint: min(
                self.joint_limits[joint]["max"],
                max(self.joint_limits[joint]["min"], float(positions[joint])),
            )
            for joint in JOINT_NAMES
            if joint in positions
        }

    def _replay_frames(self, frames):
        # Track each recorded frame directly, like the standalone replay script:
        # drop the per-command slew cap so fast recorded motion is not smeared
        # (restored afterward). Clamp every target — older recordings stored
        # measured positions that can sit a fraction past the software limit.
        self.robot.config.max_relative_target = None
        try:
            self.status = "Replay · moving to the start pose"
            self._move_to(self._clamp_positions(frames[0]["positions"]), body_tolerance=6.0)
            self._settle(0.2)
            previous_t = frames[0]["t"]
            total = len(frames)
            for index, frame in enumerate(frames[1:], start=2):
                if self.emergency_stop.is_set() or not self.running.is_set():
                    raise RuntimeError("replay stopped")
                self.status = f"Physical replay · frame {index}/{total}"
                started = time.monotonic()
                self.robot.send_action(self._action(self._clamp_positions(frame["positions"])))
                self._read_positions()
                delay = (frame["t"] - previous_t) - (time.monotonic() - started)
                time.sleep(max(0.0, min(delay, 0.1)))
                previous_t = frame["t"]
        finally:
            self.robot.config.max_relative_target = 4.0

    def _move_to(self, requested_targets, body_tolerance=BODY_TRACKING_TOLERANCE_DEGREES):
        missing = [name for name in JOINT_NAMES if name not in requested_targets]
        if missing:
            raise ValueError(f"motion target is missing {missing[0]}")
        target = {}
        for joint in JOINT_NAMES:
            value = float(requested_targets[joint])
            minimum = self.joint_limits[joint]["min"]
            maximum = self.joint_limits[joint]["max"]
            if not minimum <= value <= maximum:
                raise RuntimeError(
                    f"{joint} target {value:.2f} exceeds calibrated limits "
                    f"[{minimum:.2f}, {maximum:.2f}]"
                )
            target[joint] = value
        with self.lock:
            start = dict(self.positions)
        durations = []
        for joint in JOINT_NAMES:
            speed = (
                AUTONOMOUS_GRIPPER_SPEED_PERCENT
                if joint == "gripper"
                else AUTONOMOUS_BODY_SPEED_DEGREES
            )
            durations.append(abs(target[joint] - start[joint]) / speed)
        duration = max(0.2, max(durations))
        frequency = max(10, min(int(self.config["fps"]), 30))
        count = max(1, math.ceil(duration * frequency))
        for index in range(count):
            if self.emergency_stop.is_set() or not self.running.is_set():
                raise RuntimeError("physical motion stopped")
            fraction = (index + 1) / count
            command = {
                joint: start[joint] + (target[joint] - start[joint]) * fraction
                for joint in JOINT_NAMES
            }
            started = time.monotonic()
            self.robot.send_action(self._action(command))
            self._observe()
            time.sleep(max(0.0, 1.0 / frequency - (time.monotonic() - started)))
        self._settle(0.2)
        with self.lock:
            actual = dict(self.positions)
        errors = {
            joint: abs(actual[joint] - target[joint])
            for joint in JOINT_NAMES
        }
        failed = [
            joint
            for joint, error in errors.items()
            if error
            > (
                GRIPPER_TRACKING_TOLERANCE_PERCENT
                if joint == "gripper"
                else body_tolerance
            )
        ]
        if failed:
            joint = failed[0]
            raise RuntimeError(
                f"{joint} missed its target by {errors[joint]:.2f}; motion stopped"
            )

    def _settle(self, seconds):
        deadline = time.monotonic() + max(0.0, seconds)
        while time.monotonic() < deadline:
            if self.emergency_stop.is_set() or not self.running.is_set():
                raise RuntimeError("physical motion stopped")
            self._observe()

    def _observe(self):
        observation = self.robot.get_observation()
        positions = self._positions(observation)
        frame = np.asarray(observation["wrist"]).copy()
        height, width = frame.shape[:2]
        with self.lock:
            self.positions = positions
            self.latest_frame = frame
            self.position_history.append((time.monotonic(), dict(positions)))
        self.simulation_runtime.update_mirror(positions)
        self.frame_hub.publish_encoded(
            "wrist",
            jpeg_bytes(frame, 88),
            "real-so101",
            self.status,
            (width, height),
        )
        return observation

    def _read_positions(self):
        """Read only the servo positions — no camera — for the control tick."""
        raw = self.robot.bus.sync_read("Present_Position")
        positions = {
            joint: float(raw[joint]) for joint in JOINT_NAMES if joint in raw
        }
        with self.lock:
            self.positions = positions
            self.position_history.append((time.monotonic(), dict(positions)))
        self.simulation_runtime.update_mirror(positions)
        return positions

    def _publish_frame(self):
        """Grab and publish the wrist frame on the slower video cadence."""
        camera = self.robot.cameras.get("wrist")
        if camera is None:
            return
        frame = np.asarray(camera.async_read()).copy()
        height, width = frame.shape[:2]
        with self.lock:
            self.latest_frame = frame
        self.frame_hub.publish_encoded(
            "wrist",
            jpeg_bytes(frame, 88),
            "real-so101",
            self.status,
            (width, height),
        )

    @staticmethod
    def _positions(observation):
        return {
            joint: float(observation[f"{joint}.pos"])
            for joint in JOINT_NAMES
            if f"{joint}.pos" in observation
        }

    @staticmethod
    def _action(positions):
        return {f"{joint}.pos": value for joint, value in positions.items()}

    def _bounded_target(self, joint, current, proposed):
        limits = self.joint_limits[joint]
        minimum = limits["min"]
        maximum = limits["max"]
        if current < minimum:
            bounded = min(maximum, max(current, proposed))
            return bounded, proposed <= current or bounded != proposed
        if current > maximum:
            bounded = max(minimum, min(current, proposed))
            return bounded, proposed >= current or bounded != proposed
        bounded = min(maximum, max(minimum, proposed))
        return bounded, bounded != proposed

    def _require_stable_pose(self):
        now = time.monotonic()
        with self.lock:
            records = [
                (timestamp, positions)
                for timestamp, positions in self.position_history
                if now - timestamp <= 0.6
            ]
            history_span = (
                records[-1][0] - records[0][0]
                if len(records) >= 2
                else 0.0
            )
        if len(records) < 2 or history_span < 0.25:
            raise RuntimeError("wait for the physical arm to settle, then record again")
        samples = [positions for _, positions in records]
        for joint in JOINT_NAMES:
            values = [sample[joint] for sample in samples]
            tolerance = 2.0 if joint == "gripper" else 0.4
            if max(values) - min(values) > tolerance:
                raise RuntimeError(
                    f"{joint} is still moving; release the control and wait before recording"
                )

    def _require_operational(self):
        if not self.connected or self.robot is None:
            raise RuntimeError("connect the real robot first")
        if not self.running.is_set() or self.emergency_stop.is_set():
            raise RuntimeError(
                "physical I/O is stopped; disconnect, inspect the arm, and reconnect"
            )
