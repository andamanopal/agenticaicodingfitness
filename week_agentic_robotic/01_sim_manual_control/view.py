"""Open the SO-101 scene in MuJoCo's built-in interactive viewer.

Usage:
    python view.py            # scene with 3 pickable objects
    python view.py --office   # marker, eraser, and small box
    python view.py --empty    # arm only, no objects

In the viewer, open the "Control" panel on the right to drive each joint
with sliders. Double-click a body and Ctrl+right-drag to apply forces.
"""

import sys
from pathlib import Path

import mujoco
import mujoco.viewer

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "so101"


def main():
    if "--empty" in sys.argv:
        scene = "scene.xml"
    elif "--office" in sys.argv:
        scene = "scene_office.xml"
    else:
        scene = "scene_objects.xml"
    model = mujoco.MjModel.from_xml_path(str(MODEL_DIR / scene))
    data = mujoco.MjData(model)
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
