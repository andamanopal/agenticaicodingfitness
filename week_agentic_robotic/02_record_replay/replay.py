"""Replay a recorded trajectory in the viewer.

Usage (macOS needs mjpython):
    mjpython replay.py trajectories/pick_red_cube.json
    mjpython replay.py trajectories/pick_red_cube.json --loop

The scene is reset to the exact state captured at the start of the
recording, then the saved control targets are fed back step by step.
MuJoCo physics is deterministic, so the motion reproduces exactly.
"""

import json
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "so101"


def reset_to_initial(model, data, traj):
    """Restore the state values captured at recording start.

    qacc_warmstart matters: the constraint solver warm-starts from it, and
    contact dynamics amplify even microscopic differences over time. No
    mj_forward here — it would overwrite the warm-start; mj_step recomputes
    everything anyway.
    """
    mujoco.mj_resetData(model, data)
    data.qpos[:] = traj["init_qpos"]
    data.qvel[:] = traj["init_qvel"]
    data.ctrl[:] = traj["init_ctrl"]
    data.qacc_warmstart[:] = traj["init_warmstart"]


def main():
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not paths:
        sys.exit("usage: mjpython replay.py <trajectory.json> [--loop]")
    loop = "--loop" in sys.argv

    traj = json.loads(Path(paths[0]).read_text())
    model = mujoco.MjModel.from_xml_path(str(MODEL_DIR / traj["scene"]))
    data = mujoco.MjData(model)
    reset_to_initial(model, data, traj)

    frames = traj["frames"]
    print(f"replaying {len(frames)} frames ({len(frames) * traj['timestep']:.1f}s)"
          + (" on loop" if loop else ""))

    i = 0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            if i < len(frames):
                data.ctrl[:] = frames[i]
                i += 1
            elif loop:
                reset_to_initial(model, data, traj)
                data.ctrl[:] = frames[0]
                i = 1
            mujoco.mj_step(model, data)
            viewer.sync()
            leftover = model.opt.timestep - (time.time() - step_start)
            if leftover > 0:
                time.sleep(leftover)


if __name__ == "__main__":
    main()
