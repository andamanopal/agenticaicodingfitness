"""Record a continuously controlled trajectory as a replayable skill.

Usage:
    python 02_record_replay/record.py pick_red_cube

Hold the same Q/A through Y/H controls as Milestone 01, plus:
    SPACE   start recording / stop recording, save, and exit
    P       reset the scene before recording starts
    0       return all joints to home

The trajectory stores the initial positions, velocities, controls, solver
warm-start vector, and the control target applied at every physics step.
"""

import json
import sys
from pathlib import Path

import glfw
import mujoco


ROOT = Path(__file__).resolve().parents[1]
CONTROL_DIR = ROOT / "01_sim_manual_control"
sys.path.insert(0, str(CONTROL_DIR))

from teleop import KeyboardController, run_keyboard_viewer  # noqa: E402


MODEL_DIR = ROOT / "models" / "so101"
TRAJ_DIR = Path(__file__).resolve().parent / "trajectories"
SCENE = "scene_objects.xml"


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python 02_record_replay/record.py <trajectory_name>")
    name = sys.argv[1]
    out_path = TRAJ_DIR / f"{name}.json"

    model = mujoco.MjModel.from_xml_path(str(MODEL_DIR / SCENE))
    data = mujoco.MjData(model)
    controller = KeyboardController(model.actuator_ctrlrange)

    recording = {"scene": SCENE, "timestep": model.opt.timestep}
    frames = []
    is_recording = False

    def on_key_press(key):
        nonlocal is_recording
        if key == glfw.KEY_SPACE:
            if not is_recording:
                recording["init_qpos"] = data.qpos.tolist()
                recording["init_qvel"] = data.qvel.tolist()
                recording["init_ctrl"] = data.ctrl.tolist()
                recording["init_warmstart"] = data.qacc_warmstart.tolist()
                is_recording = True
                print("● recording... press SPACE to stop and save")
            else:
                return True
        elif key == glfw.KEY_P and not is_recording:
            mujoco.mj_resetData(model, data)
            mujoco.mj_forward(model, data)
            controller.home()
            print("scene reset")
        return False

    def after_step():
        if is_recording:
            frames.append(data.ctrl.tolist())

    def status_text():
        if is_recording:
            return "RECORDING — press SPACE to save"
        return "Ready — SPACE starts | P resets"

    print(__doc__)
    run_keyboard_viewer(
        model,
        data,
        controller,
        title="SO-101 trajectory recorder",
        on_key_press=on_key_press,
        after_step=after_step,
        status_text=status_text,
    )

    if not frames:
        sys.exit("nothing recorded — press SPACE to start recording next time")

    recording["frames"] = frames
    TRAJ_DIR.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(recording))
    duration = len(frames) * recording["timestep"]
    print(f"saved {len(frames)} frames ({duration:.1f}s) -> {out_path}")


if __name__ == "__main__":
    main()
