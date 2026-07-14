"""One thread owns the portal's interactive MuJoCo simulation."""

import io
import queue
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "so101" / "scene_office.xml"
JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
TARGET_SPEED = 0.8
STREAM_FPS = 60.0
# The digital-twin mirror renders while the physical arm runs. Keep it low so
# its JPEG encoding does not hog the GIL and add latency to real-time teleop.
MIRROR_FPS = 20.0
MAIN_STREAM_SIZE = (1920, 1080)
WRIST_STREAM_SIZE = (1080, 1920)


def jpeg_bytes(rgb, quality=82):
    output = io.BytesIO()
    Image.fromarray(rgb).save(output, format="JPEG", quality=quality)
    return output.getvalue()


class FrameHub:
    """Compress only the newest rendered pair and expose it to HTTP clients."""

    def __init__(self):
        self.lock = threading.Lock()
        self.pending_condition = threading.Condition()
        self.pending = None
        self.running = threading.Event()
        self.encoder_thread = None
        self.jpeg_pool = None
        self.main = None
        self.wrist = None
        self.sequences = {"main": 0, "wrist": 0}
        self.source = "manual"
        self.status = "Simulation ready"
        self.main_resolution = list(MAIN_STREAM_SIZE)
        self.wrist_resolution = list(WRIST_STREAM_SIZE)
        self.encoded_times = deque(maxlen=60)

    def start(self):
        if self.encoder_thread is not None:
            return
        self.jpeg_pool = ThreadPoolExecutor(max_workers=2)
        self.running.set()
        self.encoder_thread = threading.Thread(
            target=self._encode_latest,
            daemon=True,
        )
        self.encoder_thread.start()

    def stop(self):
        self.running.clear()
        with self.pending_condition:
            self.pending_condition.notify_all()
        if self.encoder_thread is not None:
            self.encoder_thread.join(timeout=3)
            self.encoder_thread = None
        if self.jpeg_pool is not None:
            self.jpeg_pool.shutdown(wait=True, cancel_futures=True)
            self.jpeg_pool = None

    def publish(self, main, wrist, source, status):
        with self.lock:
            self.source = source
            self.status = status
        with self.pending_condition:
            self.pending = (main, wrist)
            self.pending_condition.notify()

    def _encode_latest(self):
        while self.running.is_set():
            with self.pending_condition:
                self.pending_condition.wait_for(
                    lambda: self.pending is not None or not self.running.is_set()
                )
                if not self.running.is_set():
                    return
                main, wrist = self.pending
                self.pending = None

            main_future = self.jpeg_pool.submit(jpeg_bytes, main, 72)
            wrist_future = self.jpeg_pool.submit(jpeg_bytes, wrist, 72)
            main_jpeg = main_future.result()
            wrist_jpeg = wrist_future.result()
            with self.lock:
                self.main = main_jpeg
                self.wrist = wrist_jpeg
                self.sequences["main"] += 1
                self.sequences["wrist"] += 1
                self.main_resolution = list(MAIN_STREAM_SIZE)
                self.wrist_resolution = list(WRIST_STREAM_SIZE)
                self.encoded_times.append(time.monotonic())

    def publish_encoded(self, camera, frame, source, status, resolution):
        if camera not in {"main", "wrist"}:
            raise ValueError("unknown camera")
        with self.lock:
            setattr(self, camera, frame)
            self.sequences[camera] += 1
            self.source = source
            self.status = status
            setattr(self, f"{camera}_resolution", list(resolution))
            self.encoded_times.append(time.monotonic())

    def frame(self, camera):
        with self.lock:
            return self.wrist if camera == "wrist" else self.main

    def frame_with_sequence(self, camera):
        with self.lock:
            frame = self.wrist if camera == "wrist" else self.main
            return frame, self.sequences[camera]

    def metadata(self):
        with self.lock:
            fps = 0.0
            if len(self.encoded_times) >= 2:
                duration = self.encoded_times[-1] - self.encoded_times[0]
                if duration > 0:
                    fps = (len(self.encoded_times) - 1) / duration
            return {
                "source": self.source,
                "status": self.status,
                "fps": round(fps, 1),
                "main_resolution": self.main_resolution,
                "wrist_resolution": self.wrist_resolution,
            }


def configure_offscreen_buffer(model):
    """Make room for the wrist camera's declared 1080×1920 frame."""
    model.vis.global_.offwidth = max(
        model.vis.global_.offwidth,
        WRIST_STREAM_SIZE[1],
    )
    model.vis.global_.offheight = max(
        model.vis.global_.offheight,
        WRIST_STREAM_SIZE[1],
    )


class RobotRuntime:
    """Advance controls, physics, and rendering without sharing MjData."""

    def __init__(self, frame_hub):
        self.frame_hub = frame_hub
        self.commands = queue.Queue()
        self.held = set()
        self.lock = threading.Lock()
        self.running = threading.Event()
        self.paused = threading.Event()
        self.mirror = threading.Event()
        self.mirror_positions = None
        self.thread = None
        self.targets = np.zeros(len(JOINT_NAMES))
        self.control_range = None

    def start(self):
        if self.thread is not None:
            return
        self.running.set()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running.clear()
        if self.thread is not None:
            self.thread.join(timeout=3)
            self.thread = None

    def set_control(self, joint, direction, active):
        if joint not in JOINT_NAMES or direction not in (-1, 1):
            raise ValueError("unknown manual control")
        if self.paused.is_set():
            raise ValueError("manual control is paused during scripted motion")
        with self.lock:
            control = (JOINT_NAMES.index(joint), direction)
            if active:
                self.held.add(control)
            else:
                self.held.discard(control)

    def release_all(self):
        with self.lock:
            self.held.clear()

    def start_mirror(self, positions=None):
        """Echo the physical robot: drive sim joints from measured angles."""
        self.release_all()
        with self.lock:
            if positions is not None:
                self.mirror_positions = dict(positions)
        self.mirror.set()

    def stop_mirror(self):
        self.mirror.clear()
        with self.lock:
            self.mirror_positions = None

    def update_mirror(self, positions):
        with self.lock:
            self.mirror_positions = dict(positions)

    def home(self):
        if self.paused.is_set():
            raise ValueError("manual control is paused during scripted motion")
        self.commands.put("home")

    def reset(self):
        if self.paused.is_set():
            raise ValueError("manual control is paused during scripted motion")
        self.commands.put("reset")

    def _run(self):
        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        configure_offscreen_buffer(model)
        data = mujoco.MjData(model)
        self.control_range = model.actuator_ctrlrange.copy()
        self.targets = np.clip(
            np.zeros(model.nu),
            self.control_range[:, 0],
            self.control_range[:, 1],
        )
        arm_qpos_index = {
            name: int(model.jnt_qposadr[model.joint(name).id])
            for name in JOINT_NAMES
        }
        gripper_range = model.jnt_range[model.joint("gripper").id].copy()
        main_renderer = mujoco.Renderer(
            model,
            height=MAIN_STREAM_SIZE[1],
            width=MAIN_STREAM_SIZE[0],
        )
        wrist_renderer = mujoco.Renderer(
            model,
            height=WRIST_STREAM_SIZE[1],
            width=WRIST_STREAM_SIZE[0],
        )
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = [0.16, 0.0, 0.12]
        camera.distance = 0.78
        camera.azimuth = 132
        camera.elevation = -24
        last_time = time.monotonic()
        last_render = 0.0
        physics_time = 0.0
        was_mirroring = False

        try:
            while self.running.is_set():
                now = time.monotonic()
                elapsed = min(now - last_time, 0.04)
                last_time = now
                while True:
                    try:
                        command = self.commands.get_nowait()
                    except queue.Empty:
                        break
                    if command == "home":
                        self.targets[:] = np.clip(
                            0.0,
                            self.control_range[:, 0],
                            self.control_range[:, 1],
                        )
                        self.release_all()
                    elif command == "reset":
                        mujoco.mj_resetData(model, data)
                        self.targets[:] = np.clip(
                            0.0,
                            self.control_range[:, 0],
                            self.control_range[:, 1],
                        )
                        self.release_all()

                if self.mirror.is_set():
                    if now - last_render >= 1.0 / MIRROR_FPS:
                        with self.lock:
                            positions = (
                                dict(self.mirror_positions)
                                if self.mirror_positions is not None
                                else None
                            )
                        if positions is not None:
                            for name in JOINT_NAMES:
                                if name not in positions:
                                    continue
                                index = arm_qpos_index[name]
                                if name == "gripper":
                                    fraction = float(
                                        np.clip(positions[name] / 100.0, 0.0, 1.0)
                                    )
                                    data.qpos[index] = gripper_range[0] + fraction * (
                                        gripper_range[1] - gripper_range[0]
                                    )
                                else:
                                    data.qpos[index] = np.radians(
                                        float(positions[name])
                                    )
                            mujoco.mj_forward(model, data)
                        main_renderer.update_scene(data, camera=camera)
                        main = main_renderer.render().copy()
                        self.frame_hub.publish_encoded(
                            "main",
                            jpeg_bytes(main, 72),
                            "real-mirror",
                            "Live digital twin · mirroring physical SO-101",
                            MAIN_STREAM_SIZE,
                        )
                        last_render = now
                    was_mirroring = True
                    physics_time = 0.0
                elif not self.paused.is_set():
                    if was_mirroring:
                        # Adopt the mirrored pose as the hold target so the arm
                        # does not snap back when the physical robot disconnects.
                        for actuator, name in enumerate(JOINT_NAMES):
                            self.targets[actuator] = data.qpos[arm_qpos_index[name]]
                        np.clip(
                            self.targets,
                            self.control_range[:, 0],
                            self.control_range[:, 1],
                            out=self.targets,
                        )
                        data.qvel[:] = 0.0
                        was_mirroring = False
                    direction = np.zeros(model.nu)
                    with self.lock:
                        held = tuple(self.held)
                    for actuator, sign in held:
                        direction[actuator] += sign
                    physics_time += elapsed
                    while physics_time >= model.opt.timestep:
                        self.targets += (
                            direction * TARGET_SPEED * model.opt.timestep
                        )
                        np.clip(
                            self.targets,
                            self.control_range[:, 0],
                            self.control_range[:, 1],
                            out=self.targets,
                        )
                        data.ctrl[:] = self.targets
                        mujoco.mj_step(model, data)
                        physics_time -= model.opt.timestep

                    if now - last_render >= 1.0 / STREAM_FPS:
                        main_renderer.update_scene(data, camera=camera)
                        main = main_renderer.render().copy()
                        wrist_renderer.update_scene(data, camera="wrist_cam")
                        wrist = wrist_renderer.render().copy()
                        self.frame_hub.publish(
                            main,
                            wrist,
                            "manual",
                            "Manual control · live simulation",
                        )
                        last_render = now
                else:
                    physics_time = 0.0
                time.sleep(0.004)
        finally:
            main_renderer.close()
            wrist_renderer.close()


class PortalViewer:
    """Render an existing scripted MuJoCo motion into the portal streams."""

    def __init__(
        self,
        model,
        data,
        frame_hub,
        source,
        playback_speed=1.0,
    ):
        self.model = model
        self.data = data
        self.frame_hub = frame_hub
        self.source = source
        self.playback_speed = playback_speed
        self.status = "Preparing"
        configure_offscreen_buffer(model)
        self.main_renderer = mujoco.Renderer(
            model,
            height=MAIN_STREAM_SIZE[1],
            width=MAIN_STREAM_SIZE[0],
        )
        self.wrist_renderer = mujoco.Renderer(
            model,
            height=WRIST_STREAM_SIZE[1],
            width=WRIST_STREAM_SIZE[0],
        )
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = [0.16, 0.0, 0.12]
        self.camera.distance = 0.78
        self.camera.azimuth = 132
        self.camera.elevation = -24
        self.start_wall_time = time.monotonic()
        self.start_sim_time = data.time
        self.last_render_time = 0.0
        self.closed = False

    def set_status(self, status):
        self.status = status

    def sync(self, data=None, status=None):
        if data is not None:
            self.data = data
        if status is not None:
            self.status = status
        target_time = self.start_wall_time + (
            (self.data.time - self.start_sim_time) / self.playback_speed
        )
        remaining = target_time - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        if time.monotonic() - self.last_render_time >= 1.0 / STREAM_FPS:
            self.render(self.data, self.status)

    def render(self, data=None, status=None):
        if self.closed:
            return
        if data is not None:
            self.data = data
        if status is not None:
            self.status = status
        self.main_renderer.update_scene(self.data, camera=self.camera)
        main = self.main_renderer.render().copy()
        self.wrist_renderer.update_scene(self.data, camera="wrist_cam")
        wrist = self.wrist_renderer.render().copy()
        self.frame_hub.publish(main, wrist, self.source, self.status)
        self.last_render_time = time.monotonic()

    def show_capture(self, data, viewpoint, captured, total):
        pause_started = time.monotonic()
        deadline = pause_started + 0.65
        status = f"Captured {viewpoint} · {captured}/{total}"
        while time.monotonic() < deadline:
            self.render(data, status)
            time.sleep(1.0 / STREAM_FPS)
        self.start_wall_time += time.monotonic() - pause_started

    def show_complete(self, data, output_dir, frame_count):
        self.render(data, f"Scan complete · {frame_count} views saved")
        time.sleep(0.7)

    def hold(self, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.render(self.data, self.status)
            time.sleep(1.0 / STREAM_FPS)

    def close(self):
        if self.closed:
            return
        self.main_renderer.close()
        self.wrist_renderer.close()
        self.closed = True
