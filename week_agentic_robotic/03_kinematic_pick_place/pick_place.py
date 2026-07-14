"""Kinematics-based pick-and-place: no learning, no recordings.

Reads object positions straight from the simulator (stand-in for
perception), solves IK for each waypoint, and ramps the position servos
through the sequence: tuck -> turn -> hover -> descend -> grip -> lift ->
tuck -> turn -> lower -> release. Travel between azimuths happens in the
folded carry posture so the swing arc clears every object by construction.

Usage:
    mjpython pick_place.py red_cube            # watch one pick (macOS)
    mjpython pick_place.py all                 # all three objects
    python  pick_place.py all --headless       # no viewer, prints results
"""

import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from ik import solve_ik

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "so101"

GRIPPER_OPEN = 1.2
GRIPPER_CLOSED = 0.0

# Elbow-bent pose near the workspace center: good IK seed, but its gripper is
# at floor level, so it is never used as a physical waypoint. The all-zeros
# home pose is fully extended — a singularity where gradient-based IK stalls.
IK_SEED = np.array([0.0, 0.0, 0.47, 1.18, 1.58])

# Folded travel posture ("tuck and turn"): every hand geom stays above
# z=0.16 and inside r=0.16, so rotating the pan joint in this posture sweeps
# a small arc high above the objects (tallest 0.06) and inside their ring
# (r=0.25). Pan (index 0) is set per move.
CARRY_POSE = np.array([0.0, -1.543, 0.469, 1.658, 1.737])
HOVER_Z = 0.10
DROP_ZONE = np.array([0.12, 0.18])

# grasp height = object center; place offsets keep objects from stacking on
# the pad. "close" is the gripper target while holding: boxes and cylinders
# tolerate a hard squeeze (force-limited servo just stalls), but a sphere
# squirts out watermelon-seed style — close only a few mm past contact
# (jaw gap ~3.5cm at 0.25 rad vs the 4cm ball).
OBJECTS = {
    "red_cube": {"grasp_z": 0.018, "close": 0.0, "place_xy": DROP_ZONE + [-0.02, -0.02]},
    "green_cylinder": {"grasp_z": 0.032, "close": 0.0, "place_xy": DROP_ZONE + [0.0, 0.02]},
    "blue_sphere": {"grasp_z": 0.022, "close": 0.25, "place_xy": DROP_ZONE + [0.02, -0.01]},
}


class Controller:
    """Ramps position-servo targets through IK waypoints on a live sim."""

    def __init__(self, model, data, viewer=None):
        self.model = model
        self.data = data
        self.viewer = viewer
        self.scratch = mujoco.MjData(model)  # IK workspace, never stepped

    def step_for(self, seconds):
        steps = int(seconds / self.model.opt.timestep)
        for _ in range(steps):
            t0 = time.time()
            mujoco.mj_step(self.model, self.data)
            if self.viewer:
                self.viewer.sync()
                playback_speed = getattr(self.viewer, "playback_speed", 1.0)
                leftover = (
                    self.model.opt.timestep / playback_speed
                    - (time.time() - t0)
                )
                if leftover > 0:
                    time.sleep(leftover)

    def ramp_ctrl(self, arm_target=None, gripper_target=None, speed=1.2):
        """Linearly interpolate ctrl to the target, then settle briefly."""
        start = self.data.ctrl.copy()
        end = start.copy()
        if arm_target is not None:
            end[:5] = arm_target
        if gripper_target is not None:
            end[5] = gripper_target
        duration = max(0.4, float(np.abs(end - start).max()) / speed)
        steps = int(duration / self.model.opt.timestep)
        for i in range(steps):
            t0 = time.time()
            self.data.ctrl[:] = start + (end - start) * (i + 1) / steps
            mujoco.mj_step(self.model, self.data)
            if self.viewer:
                self.viewer.sync()
                playback_speed = getattr(self.viewer, "playback_speed", 1.0)
                leftover = (
                    self.model.opt.timestep / playback_speed
                    - (time.time() - t0)
                )
                if leftover > 0:
                    time.sleep(leftover)
        self.step_for(0.3)

    def move_to(self, pos):
        target = np.asarray(pos, dtype=float)
        pan_aligned = IK_SEED.copy()
        pan_aligned[0] = np.arctan2(target[1], target[0])
        seeds = [self.data.qpos[:5].copy(), IK_SEED, pan_aligned]
        q, err = min((solve_ik(self.model, self.scratch, target, q_init=s)
                      for s in seeds), key=lambda r: r[1])
        if err > 0.03:
            print(f"  stopping plan: IK residual {err * 1000:.1f}mm at {pos}")
            return False
        self.ramp_ctrl(arm_target=q)
        return True

    def align_jaws(self, yaw):
        """Rotate the wrist so the jaw closing axis matches a table yaw."""
        fixed = self.data.geom("fixed_jaw_sph_tip1").xpos[:2]
        moving = self.data.geom("moving_jaw_sph_tip1").xpos[:2]
        jaw_axis = moving - fixed
        current_yaw = np.arctan2(jaw_axis[1], jaw_axis[0])
        delta = (yaw - current_yaw + np.pi) % (2 * np.pi) - np.pi

        # A parallel-jaw grasp is unchanged after a 180-degree rotation.
        if delta > np.pi / 2:
            delta -= np.pi
        elif delta < -np.pi / 2:
            delta += np.pi

        wrist_joint_id = self.model.joint("wrist_roll").id
        wrist_qpos_index = self.model.jnt_qposadr[wrist_joint_id]
        wrist_actuator_id = self.model.actuator("wrist_roll").id
        arm = self.data.ctrl[:5].copy()
        requested = self.data.qpos[wrist_qpos_index] + delta
        lower, upper = self.model.jnt_range[wrist_joint_id]
        arm[wrist_actuator_id] = np.clip(requested, lower, upper)
        if abs(arm[wrist_actuator_id] - requested) > 0.05:
            print("  stopping plan: grasp yaw exceeds the wrist-roll limit")
            return False
        self.ramp_ctrl(arm_target=arm)

        fixed = self.data.geom("fixed_jaw_sph_tip1").xpos[:2]
        moving = self.data.geom("moving_jaw_sph_tip1").xpos[:2]
        jaw_axis = moving - fixed
        achieved_yaw = np.arctan2(jaw_axis[1], jaw_axis[0])
        error = (yaw - achieved_yaw + np.pi / 2) % np.pi - np.pi / 2
        if abs(error) > 0.08:
            print(f"  stopping plan: grasp yaw error is {abs(error):.3f} radians")
            return False
        return True

    def tuck(self):
        """Fold into the carry posture without changing azimuth."""
        carry = CARRY_POSE.copy()
        carry[0] = self.data.ctrl[0]
        self.ramp_ctrl(arm_target=carry)

    def pan_to(self, xy):
        """Rotate only the pan joint; in the carry posture this arc is
        provably clear of the objects."""
        arm = self.data.ctrl[:5].copy()
        arm[0] = np.arctan2(xy[1], xy[0])
        self.ramp_ctrl(arm_target=arm)

    def pick_and_place(self, name):
        spec = OBJECTS[name]
        obj_xy = self.data.body(name).xpos[:2].copy()
        grasp_z, place_xy = spec["grasp_z"], spec["place_xy"]
        print(f"picking {name} at ({obj_xy[0]:.3f}, {obj_xy[1]:.3f})")

        self.tuck()
        self.pan_to(obj_xy)
        self.ramp_ctrl(gripper_target=GRIPPER_OPEN)
        if not self.move_to([*obj_xy, HOVER_Z]):
            return False
        if not self.move_to([*obj_xy, grasp_z]):
            return False
        self.ramp_ctrl(gripper_target=spec["close"], speed=0.6)
        self.step_for(0.5)
        if not self.move_to([*obj_xy, HOVER_Z]):
            return False
        self.tuck()
        self.pan_to(place_xy)
        if not self.move_to([*place_xy, HOVER_Z]):
            return False
        if not self.move_to([*place_xy, grasp_z + 0.003]):
            return False
        self.ramp_ctrl(gripper_target=0.5)  # crack the jaws, let it settle
        self.step_for(0.3)
        self.ramp_ctrl(gripper_target=GRIPPER_OPEN)
        if not self.move_to([*place_xy, HOVER_Z]):
            return False

        placed = self.data.body(name).xpos
        ok = (np.abs(placed[:2] - DROP_ZONE) < 0.06).all() and placed[2] < 0.08
        print(f"  -> {name} at ({placed[0]:.3f}, {placed[1]:.3f}, {placed[2]:.3f})"
              f"  {'ON PAD ✓' if ok else 'MISSED ✗'}")
        return ok

    def home(self):
        self.tuck()
        self.ramp_ctrl(arm_target=np.zeros(5), gripper_target=GRIPPER_CLOSED)


def run(names, headless):
    model = mujoco.MjModel.from_xml_path(str(MODEL_DIR / "scene_objects.xml"))
    data = mujoco.MjData(model)

    def sequence(viewer=None):
        ctl = Controller(model, data, viewer)
        ctl.step_for(0.5)  # let objects settle
        results = []
        for name in names:
            ok = ctl.pick_and_place(name)
            results.append(ok)
            if not ok:
                print("  stopping sequence after the failed move")
                break
        ctl.home()
        return results

    if headless:
        results = sequence()
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            results = sequence(viewer)
            while viewer.is_running():  # keep the window open to inspect
                viewer.sync()
                time.sleep(0.05)

    print(f"\n{sum(results)}/{len(results)} placed on the drop zone")
    return all(results)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    target = args[0] if args else "all"
    names = list(OBJECTS) if target == "all" else [target]
    unknown = [n for n in names if n not in OBJECTS]
    if unknown:
        sys.exit(f"unknown object {unknown[0]}; choose from {list(OBJECTS)} or 'all'")
    ok = run(names, headless="--headless" in sys.argv)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
