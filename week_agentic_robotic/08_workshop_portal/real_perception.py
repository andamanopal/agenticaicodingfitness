"""Write a physical SO-101 scan in the same RGB contract as MuJoCo."""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw

from physical_workspace import (
    VIEWPOINTS,
    camera_transforms_from_joint_positions,
    profile_digest,
)


PERCEPTION_CONTRACT = "calibrated_multi_view_rgb_scan_v2"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


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


def save_contact_sheet(output_dir, frames):
    images = [Image.open(output_dir / frame["rgb"]).convert("RGB") for frame in frames]
    try:
        width, height = images[0].size
        label_height = 34
        sheet = Image.new("RGB", (width * len(images), height + label_height), "#11161c")
        draw = ImageDraw.Draw(sheet)
        for index, (frame, image) in enumerate(zip(frames, images)):
            x = index * width
            sheet.paste(image, (x, label_height))
            draw.text((x + 12, 10), frame["viewpoint"], fill="#f1f5f9")
        sheet.save(output_dir / "contact_sheet.png")
    finally:
        for image in images:
            image.close()


def capture_real_scan(runtime, output_dir, scene):
    """Move through taught poses once and atomically publish three RGB frames."""
    if scene not in {"empty", "office"}:
        raise ValueError("physical scan scene must be empty or office")
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.with_name(f".{output_dir.name}.in_progress")
    backup = output_dir.with_name(f".{output_dir.name}.previous")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    profile = runtime.workspace.scan_profile()

    try:
        def publish_capture(name, capture):
            Image.fromarray(capture["rgb"]).save(staging / f"{name}.png")

        result = runtime.run_scan(on_capture=publish_capture)
        captures = result["captures"]
        camera_transforms = camera_transforms_from_joint_positions(
            profile,
            {
                name: capture["joint_positions_degrees"]
                for name, capture in captures.items()
            },
        )
        frames = []
        intrinsics = profile["intrinsics"]
        for name in VIEWPOINTS:
            if name not in captures:
                raise RuntimeError(f"physical scan did not return {name}")
            capture = captures[name]
            rgb = capture["rgb"]
            taught = profile["viewpoints"][name]["joint_positions_degrees"]
            pose_errors = {
                joint: abs(
                    float(capture["joint_positions_degrees"][joint])
                    - float(taught[joint])
                )
                for joint in taught
                if joint != "gripper"
            }
            maximum_pose_error = max(pose_errors.values())
            if maximum_pose_error > 1.5:
                raise RuntimeError(
                    f"{name} missed its taught camera pose by "
                    f"{maximum_pose_error:.2f} degrees; inspect the arm and retry"
                )
            image_name = f"{name}.png"
            Image.fromarray(rgb).save(staging / image_name)
            frames.append({
                "viewpoint": name,
                "rgb": image_name,
                "intrinsics": {
                    "width": intrinsics["width"],
                    "height": intrinsics["height"],
                    "K": intrinsics["K"],
                    "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
                    "image_state": "undistorted",
                },
                "T_base_camera_cv": camera_transforms[name].round(8).tolist(),
                "taught_T_base_camera_cv": profile["viewpoints"][name][
                    "T_base_camera_cv"
                ],
                "joint_positions": capture["joint_positions_degrees"],
                "joint_units": "degrees; gripper percent",
                "maximum_taught_pose_error_degrees": round(maximum_pose_error, 4),
                "captured_at_utc": utc_now(),
                "quality": image_quality(rgb),
            })
        save_contact_sheet(staging, frames)

        review = [
            frame["viewpoint"]
            for frame in frames
            if frame["quality"]["status"] != "pass"
        ]
        checks = ["brightness", "contrast", "taught_pose_repeatability"]
        note = None
        if scene == "empty":
            review = [
                frame["viewpoint"]
                for frame in frames
                if not 0.12 < frame["quality"]["brightness"] < 0.9
            ]
            checks = ["brightness", "taught_pose_repeatability"]
            note = "Low contrast is expected for an empty calibration table."

        profile_sha256 = profile_digest(profile)
        workspace = profile["workspace"]
        scan = {
            "scan_id": datetime.now(timezone.utc).strftime("scan_%Y%m%dT%H%M%S%fZ"),
            "scene": "physical_empty_table" if scene == "empty" else "physical_office_table",
            "source": "real_so101",
            "robot_id": profile["robot_id"],
            "layout": "physical",
            "workspace_profile_sha256": profile_sha256,
            "camera": f"real-so101:{profile['robot_id']}:wrist",
            "sensor_modalities": ["rgb"],
            "perception_contract": PERCEPTION_CONTRACT,
            "uses_privileged_simulator_data": False,
            "camera_pose_source": (
                "validated_fixed_gripper_camera_transform_plus_actual_joint_fk"
            ),
            "coordinate_convention": "robot base; camera +X right, +Y down, +Z forward",
            "workspace_bounds": {
                "minimum": workspace["minimum"],
                "maximum": workspace["maximum"],
            },
            "table_bounds": {
                "minimum": workspace["table_minimum"],
                "maximum": workspace["table_maximum"],
            },
            "scene_lock": {
                "status": "required",
                "rule": "Do not move the camera mount, robot base, table, or objects until the action finishes.",
            },
            "transform_equation": "p_base = T_base_camera_cv @ p_camera_cv",
            "observation_quality": {
                "status": "basic_check_passed" if not review else "review",
                "checks": checks,
                "frames_for_review": review,
                "note": note,
            },
            "frames": frames,
        }
        (staging / "scan.json").write_text(json.dumps(scan, indent=2) + "\n")

        shutil.rmtree(backup, ignore_errors=True)
        if output_dir.exists():
            output_dir.replace(backup)
        try:
            staging.replace(output_dir)
        except Exception:
            if backup.exists() and not output_dir.exists():
                backup.replace(output_dir)
            raise
        shutil.rmtree(backup, ignore_errors=True)
        return scan
    finally:
        shutil.rmtree(staging, ignore_errors=True)
