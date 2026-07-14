"""Check that this laptop can load, step, and render the SO-101.

Usage:
    python 00_getting_started/check_setup.py
"""

import sys
from pathlib import Path

try:
    import glfw
    import mujoco
    import torch
    import transformers
    from PIL import Image
except ModuleNotFoundError as error:
    raise SystemExit(
        f"missing {error.name!r}; run: python -m pip install -r requirements.txt"
    ) from error


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "so101" / "scene_office.xml"
EXPECTED_ACTUATORS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def main():
    print(f"Python {sys.version_info.major}.{sys.version_info.minor}")
    print(f"GLFW {glfw.__version__}")
    print(f"MuJoCo {mujoco.__version__}")
    print(f"Pillow {Image.__version__}")
    print(f"PyTorch {torch.__version__}")
    print(f"Transformers {transformers.__version__}")
    if torch.cuda.is_available():
        depth_device = "CUDA"
    elif torch.backends.mps.is_available():
        depth_device = "Apple Metal"
    else:
        depth_device = "CPU"
    print(f"Learned-depth device: {depth_device}")

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    actuator_names = [model.actuator(index).name for index in range(model.nu)]
    if actuator_names != EXPECTED_ACTUATORS:
        raise SystemExit(f"unexpected actuators: {actuator_names}")

    for _ in range(20):
        mujoco.mj_step(model, data)

    camera_id = model.camera("wrist_cam").id
    camera_width, camera_height = model.cam_resolution[camera_id]
    scale = 160 / max(camera_width, camera_height)
    render_width = round(camera_width * scale)
    render_height = round(camera_height * scale)
    renderer = mujoco.Renderer(
        model,
        height=render_height,
        width=render_width,
    )
    try:
        renderer.update_scene(data, camera="wrist_cam")
        image = renderer.render()
    finally:
        renderer.close()

    print(f"✓ loaded {MODEL_PATH.name}")
    print(f"✓ found all {model.nu} actuators and wrist_cam")
    print(f"✓ advanced physics to {data.time:.3f} seconds")
    print(f"✓ rendered one wrist image: {image.shape[1]} × {image.shape[0]}")
    print("\nSetup is ready. Next: python 01_sim_manual_control/view.py")


if __name__ == "__main__":
    main()
