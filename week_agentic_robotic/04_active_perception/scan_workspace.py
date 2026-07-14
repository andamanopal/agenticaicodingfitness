"""Capture a settled, RGB-only three-view scan from the SO-101 wrist camera.

Usage:
    python 04_active_perception/scan_workspace.py
    python 04_active_perception/scan_workspace.py --headless
    python 04_active_perception/scan_workspace.py --scene primitives
    python 04_active_perception/scan_workspace.py --output my_scan
"""

import argparse
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import glfw
import mujoco
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "so101"
WORKSHOP_ROOM_PATH = MODEL_DIR / "workshop_room.xml"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output" / "latest_scan"
SCENES = {
    "office": "scene_office.xml",
    "empty": "scene_office_empty.xml",
    "primitives": "scene_objects.xml",
}

MAX_IMAGE_EDGE = 640
CAMERA_NAME = "wrist_cam"
PERCEPTION_CONTRACT = "calibrated_multi_view_rgb_scan_v2"
SURVEY_WRIST_ROLL = -0.5
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Positive base-frame Y is the robot's left when it faces the workspace. The
# wrist roll stays fixed so parallax comes from camera motion, not image flips.
# These poses were checked for joint limits, object contact, and image coverage.
SURVEY_POSES = {
    "survey_left": np.array([
        -0.4052, -1.1803, 0.5646, 1.4466, SURVEY_WRIST_ROLL,
    ]),
    "survey_center": np.array([
        -0.3, -1.5762, 0.4832, 1.6388, SURVEY_WRIST_ROLL,
    ]),
    "survey_right": np.array([
        0.3967, -0.9019, 0.3855, 1.4158, SURVEY_WRIST_ROLL,
    ]),
}

# A folded pose keeps the camera and gripper high while moving between views.
TRAVEL_POSE = np.array([0.0, -1.543, 0.469, 1.658, SURVEY_WRIST_ROLL])

OFFICE_OBJECTS = {
    "marker": {"joint": "marker_joint", "z": 0.009},
    "eraser": {"joint": "eraser_joint", "z": 0.011},
    "small_box": {"joint": "small_box_joint", "z": 0.025},
}


class ScanCancelled(Exception):
    """Stop a visual scan without leaving a partial scene record."""


class ScanViewer:
    """Show the full robot and live wrist camera during a scripted scan."""

    def __init__(
        self,
        model,
        *,
        title="SO-101 active perception scan",
        heading="ACTIVE PERCEPTION",
        escape_help="Esc closes without saving",
        playback_speed=1.0,
    ):
        if playback_speed <= 0:
            raise ValueError("playback_speed must be positive")
        if not glfw.init():
            raise RuntimeError("GLFW could not initialize; rerun with --headless")

        self.model = model
        self.window = None
        self.context = None
        try:
            self.window = glfw.create_window(
                1200,
                800,
                title,
                None,
                None,
            )
            if self.window is None:
                raise RuntimeError(
                    "GLFW could not create the scan window; rerun with --headless"
                )

            def close_on_escape(window, key, scancode, action, mods):
                del scancode, mods
                if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
                    glfw.set_window_should_close(window, True)

            glfw.set_key_callback(self.window, close_on_escape)
            glfw.make_context_current(self.window)
            glfw.swap_interval(1)
            self.main_camera = mujoco.MjvCamera()
            self.wrist_camera = mujoco.MjvCamera()
            self.option = mujoco.MjvOption()
            self.main_scene = mujoco.MjvScene(model, maxgeom=10_000)
            self.wrist_scene = mujoco.MjvScene(model, maxgeom=10_000)
            self.context = mujoco.MjrContext(
                model,
                mujoco.mjtFontScale.mjFONTSCALE_150,
            )
            mujoco.mjv_defaultCamera(self.main_camera)
            mujoco.mjv_defaultCamera(self.wrist_camera)
            mujoco.mjv_defaultOption(self.option)
            mujoco.mjv_defaultFreeCamera(model, self.main_camera)
            self.wrist_camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.wrist_camera.fixedcamid = model.camera(CAMERA_NAME).id
            self.start_wall_time = None
            self.start_sim_time = None
            self.last_render_time = 0.0
            self.capture_flash_until = 0.0
            self.heading = heading
            self.escape_help = escape_help
            self.playback_speed = playback_speed
        except Exception:
            self.close()
            raise

    def sync(self, data, status):
        """Pace simulated motion in real time and refresh at display speed."""
        if self.start_wall_time is None:
            self.start_wall_time = time.monotonic()
            self.start_sim_time = data.time

        target_wall_time = self.start_wall_time + (
            (data.time - self.start_sim_time) / self.playback_speed
        )
        remaining = target_wall_time - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)

        now = time.monotonic()
        if now - self.last_render_time >= 1.0 / 60.0:
            self.render(data, status)
            self.last_render_time = time.monotonic()
        if glfw.window_should_close(self.window):
            raise ScanCancelled("scan cancelled; no partial scan.json was written")

    def show_capture(self, data, viewpoint, captured, total):
        pause_started = time.monotonic()
        self.capture_flash_until = time.monotonic() + 0.65
        status = f"CAPTURED {viewpoint}  ({captured}/{total})"
        while time.monotonic() < self.capture_flash_until:
            self.render(data, status)
        self.start_wall_time += time.monotonic() - pause_started

    def show_complete(self, data, output_dir, frame_count):
        deadline = time.monotonic() + 1.0
        status = f"SCAN COMPLETE\nSaved {frame_count} views to {output_dir.name}"
        while time.monotonic() < deadline:
            self.render(data, status)

    def render(self, data, status):
        glfw.make_context_current(self.window)
        glfw.poll_events()
        if glfw.window_should_close(self.window):
            raise ScanCancelled("scan cancelled; no partial scan.json was written")
        width, height = glfw.get_framebuffer_size(self.window)
        if width <= 0 or height <= 0:
            return

        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjv_updateScene(
            self.model,
            data,
            self.option,
            None,
            self.main_camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.main_scene,
        )
        mujoco.mjr_render(viewport, self.main_scene, self.context)

        camera_id = self.model.camera(CAMERA_NAME).id
        camera_width, camera_height = self.model.cam_resolution[camera_id]
        camera_aspect = camera_width / camera_height
        inset_width = int(width * 0.32)
        inset_height = int(inset_width / camera_aspect)
        max_inset_height = int(height * 0.40)
        if inset_height > max_inset_height:
            inset_height = max_inset_height
            inset_width = int(inset_height * camera_aspect)
        margin = max(8, int(width * 0.012))
        inset = mujoco.MjrRect(
            width - inset_width - margin,
            height - inset_height - margin,
            inset_width,
            inset_height,
        )
        flashing = time.monotonic() < self.capture_flash_until
        border_color = (0.33, 0.84, 0.66) if flashing else (0.33, 0.78, 1.0)
        border = 4 if flashing else 3
        mujoco.mjr_rectangle(
            mujoco.MjrRect(
                inset.left - border,
                inset.bottom - border,
                inset.width + 2 * border,
                inset.height + 2 * border,
            ),
            *border_color,
            1.0,
        )
        mujoco.mjv_updateScene(
            self.model,
            data,
            self.option,
            None,
            self.wrist_camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.wrist_scene,
        )
        mujoco.mjr_render(inset, self.wrist_scene, self.context)

        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            viewport,
            self.heading,
            f"{status}\n\n{self.escape_help}",
            self.context,
        )
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            inset,
            "WRIST CAMERA",
            "CAPTURE" if flashing else "LIVE",
            self.context,
        )
        glfw.swap_buffers(self.window)

    def close(self):
        if self.context is not None:
            self.context.free()
            self.context = None
        if self.window is not None:
            glfw.destroy_window(self.window)
            self.window = None
        glfw.terminate()


def scene_fingerprint(model_path):
    """Fingerprint the scene plus its shared robot and room models."""
    digest = hashlib.sha256()
    for path in (
        Path(model_path),
        MODEL_DIR / "so101.xml",
        WORKSHOP_ROOM_PATH,
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def randomize_office_layout(model, data, seed):
    """Place the three office objects in separated, randomized desk lanes."""
    random = np.random.default_rng(seed)
    lanes = random.permutation([-0.10, 0.0, 0.10])
    layout = {}

    for (name, spec), lane_y in zip(OFFICE_OBJECTS.items(), lanes):
        x = random.uniform(0.21, 0.28)
        y = lane_y + random.uniform(-0.012, 0.012)
        yaw = random.uniform(-0.35, 0.35)
        qpos_index = model.jnt_qposadr[model.joint(spec["joint"]).id]
        data.qpos[qpos_index:qpos_index + 3] = [x, y, spec["z"]]
        data.qpos[qpos_index + 3:qpos_index + 7] = [
            np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)
        ]
        layout[name] = {"position": [x, y, spec["z"]], "yaw": yaw}

    mujoco.mj_forward(model, data)
    return layout


def move_joints(model, data, target, speed=1.0, after_step=None):
    """Move the five arm servos to one target with a linear joint ramp."""
    start = data.ctrl[:5].copy()
    duration = max(0.5, float(np.abs(target - start).max()) / speed)
    steps = max(1, int(duration / model.opt.timestep))

    for step in range(steps):
        fraction = (step + 1) / steps
        data.ctrl[:5] = start + (target - start) * fraction
        mujoco.mj_step(model, data)
        if after_step is not None:
            after_step()


def settle(model, data, seconds=0.4, after_step=None):
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
        if after_step is not None:
            after_step()


def camera_intrinsics(model, camera_id, image_width, image_height):
    fovy = np.deg2rad(model.cam_fovy[camera_id])
    focal = image_height / (2.0 * np.tan(fovy / 2.0))
    return np.array([
        [focal, 0.0, image_width / 2.0],
        [0.0, focal, image_height / 2.0],
        [0.0, 0.0, 1.0],
    ])


def image_quality(rgb):
    gray = rgb.mean(axis=2) / 255.0
    brightness = float(gray.mean())
    contrast = float(gray.std())
    status = "pass" if 0.12 < brightness < 0.9 and contrast > 0.08 else "review"
    return {
        "status": status,
        "checks": ["brightness", "contrast"],
        "not_checked": ["object coverage", "motion blur", "occlusion"],
        "brightness": round(brightness, 4),
        "contrast": round(contrast, 4),
    }


def capture_frame(
    model, data, renderer, output_dir, viewpoint, image_width, image_height
):
    camera_id = model.camera(CAMERA_NAME).id
    renderer.update_scene(data, camera=CAMERA_NAME)
    rgb = renderer.render()

    image_name = f"{viewpoint}.png"
    Image.fromarray(rgb).save(output_dir / image_name)

    # MuJoCo cameras look along local -Z with +Y up. This axis flip stores the
    # conventional CV frame: +Z forward, +X right, +Y down.
    rotation_world_camera = data.cam_xmat[camera_id].reshape(3, 3)
    base_id = model.body("base").id
    rotation_world_base = data.xmat[base_id].reshape(3, 3)
    rotation_base_camera = rotation_world_base.T @ rotation_world_camera
    position_base_camera = rotation_world_base.T @ (
        data.cam_xpos[camera_id] - data.xpos[base_id]
    )
    rotation_base_cv = rotation_base_camera @ np.diag([1.0, -1.0, -1.0])
    transform = np.eye(4)
    transform[:3, :3] = rotation_base_cv
    transform[:3, 3] = position_base_camera

    joints = {}
    for name in JOINT_NAMES:
        qpos_index = model.jnt_qposadr[model.joint(name).id]
        joints[name] = round(float(data.qpos[qpos_index]), 6)

    return {
        "viewpoint": viewpoint,
        "rgb": image_name,
        "intrinsics": {
            "width": image_width,
            "height": image_height,
            "K": np.round(
                camera_intrinsics(model, camera_id, image_width, image_height), 6
            ).tolist(),
        },
        "T_base_camera_cv": np.round(transform, 8).tolist(),
        "joint_positions": joints,
        "sim_time_seconds": round(float(data.time), 6),
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "quality": image_quality(rgb),
    }


def save_contact_sheet(output_dir, frames, image_width, image_height):
    label_height = 34
    sheet = Image.new(
        "RGB",
        (image_width * len(frames), image_height + label_height),
        "#11161c",
    )
    draw = ImageDraw.Draw(sheet)

    for index, frame in enumerate(frames):
        image = Image.open(output_dir / frame["rgb"])
        x = index * image_width
        sheet.paste(image, (x, label_height))
        draw.text((x + 12, 10), frame["viewpoint"], fill="#f1f5f9")

    sheet.save(output_dir / "contact_sheet.png")


def scan_workspace(
    viewpoints=None,
    output_dir=DEFAULT_OUTPUT,
    scene="office",
    seed=None,
    visualize=False,
    viewer_factory=None,
):
    """Run the survey and return the JSON-safe scan record."""
    viewpoints = viewpoints or list(SURVEY_POSES)
    unknown = [name for name in viewpoints if name not in SURVEY_POSES]
    if unknown:
        raise ValueError(f"unknown viewpoint {unknown[0]!r}; choose from {list(SURVEY_POSES)}")
    if scene not in SCENES:
        raise ValueError(f"unknown scene {scene!r}; choose from {list(SCENES)}")

    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir.with_name(f".{output_dir.name}.in_progress")
    backup_dir = output_dir.with_name(f".{output_dir.name}.previous")
    shutil.rmtree(staging_dir, ignore_errors=True)
    staging_dir.mkdir(parents=True)
    model_path = MODEL_DIR / SCENES[scene]
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    camera_id = model.camera(CAMERA_NAME).id
    camera_width, camera_height = map(int, model.cam_resolution[camera_id])
    image_scale = MAX_IMAGE_EDGE / max(camera_width, camera_height)
    image_width = max(1, round(camera_width * image_scale))
    image_height = max(1, round(camera_height * image_scale))
    if seed is not None:
        if scene != "office":
            raise ValueError("--seed is supported only for the office scene")
        randomize_office_layout(model, data, seed)

    if visualize and viewer_factory is not None:
        viewer = viewer_factory(model, data)
    else:
        viewer = ScanViewer(model) if visualize else None
    try:
        renderer = mujoco.Renderer(model, height=image_height, width=image_width)
        frames = []
        status = "Settling the scene"

        def update_viewer():
            if viewer is not None:
                viewer.sync(data, status)

        try:
            settle(model, data, seconds=0.5, after_step=update_viewer)
            status = "Moving to the safe travel pose"
            move_joints(model, data, TRAVEL_POSE, after_step=update_viewer)

            for index, viewpoint in enumerate(viewpoints, start=1):
                print(f"capturing {viewpoint}...")
                status = f"Moving to {viewpoint}  ({index}/{len(viewpoints)})"
                move_joints(
                    model,
                    data,
                    SURVEY_POSES[viewpoint],
                    after_step=update_viewer,
                )
                status = f"Settling at {viewpoint}"
                settle(model, data, after_step=update_viewer)
                status = f"Capturing {viewpoint}"
                if viewer is not None:
                    viewer.render(data, status)
                frames.append(capture_frame(
                    model,
                    data,
                    renderer,
                    staging_dir,
                    viewpoint,
                    image_width,
                    image_height,
                ))
                if viewer is not None:
                    viewer.show_capture(data, viewpoint, index, len(viewpoints))
                status = "Returning to the safe travel pose"
                move_joints(model, data, TRAVEL_POSE, after_step=update_viewer)
        finally:
            renderer.close()

        save_contact_sheet(staging_dir, frames, image_width, image_height)
        frames_for_review = [
            frame["viewpoint"]
            for frame in frames
            if frame["quality"]["status"] != "pass"
        ]
        quality_checks = ["brightness", "contrast"]
        quality_note = None
        if scene == "empty":
            frames_for_review = [
                frame["viewpoint"]
                for frame in frames
                if not 0.12 < frame["quality"]["brightness"] < 0.9
            ]
            quality_checks = ["brightness"]
            quality_note = "Low contrast is expected for an empty calibration table."
        scan = {
            "scan_id": datetime.now(timezone.utc).strftime("scan_%Y%m%dT%H%M%S%fZ"),
            "scene": model_path.name,
            "layout": "randomized" if seed is not None else "canonical",
            "scene_model_sha256": scene_fingerprint(model_path),
            "camera": CAMERA_NAME,
            "sensor_modalities": ["rgb"],
            "perception_contract": PERCEPTION_CONTRACT,
            "uses_privileged_simulator_data": False,
            "camera_pose_source": "robot_kinematics_and_fixed_camera_mount",
            "coordinate_convention": "robot base; camera +X right, +Y down, +Z forward",
            "scene_lock": {
                "status": "assumed_from_scan_start",
                "rule": "Do not move objects from the first view until the planned action finishes.",
            },
            "transform_equation": "p_base = T_base_camera_cv @ p_camera_cv",
            "observation_quality": {
                "status": "basic_check_passed" if not frames_for_review else "review",
                "checks": quality_checks,
                "frames_for_review": frames_for_review,
                "note": quality_note,
            },
            "frames": frames,
        }
        (staging_dir / "scan.json").write_text(json.dumps(scan, indent=2) + "\n")

        shutil.rmtree(backup_dir, ignore_errors=True)
        if output_dir.exists():
            output_dir.replace(backup_dir)
        try:
            staging_dir.replace(output_dir)
        except Exception:
            if backup_dir.exists() and not output_dir.exists():
                backup_dir.replace(output_dir)
            raise
        shutil.rmtree(backup_dir, ignore_errors=True)

        print(f"saved {len(frames)} views -> {output_dir}")
        print(f"basic image check: {scan['observation_quality']['status']}")
        if viewer is not None:
            viewer.show_complete(data, output_dir, len(frames))
        return scan
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
        if viewer is not None:
            viewer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=SCENES, default="office")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="capture without opening the animated MuJoCo scan window",
    )
    args = parser.parse_args()
    output = args.output
    if output is None:
        default_names = {
            "office": "latest_scan",
            "empty": "empty_scan",
            "primitives": "primitives_scan",
        }
        output = DEFAULT_OUTPUT.parent / default_names[args.scene]
        if args.seed is not None:
            output = DEFAULT_OUTPUT.parent / f"seed_{args.seed:03d}_scan"
    try:
        scan_workspace(
            output_dir=output,
            scene=args.scene,
            seed=args.seed,
            visualize=not args.headless,
        )
    except ScanCancelled as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
