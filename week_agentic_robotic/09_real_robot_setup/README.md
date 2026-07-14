# 09 · Real SO-ARM101 Setup

This folder keeps the instructor's proven LeRobot setup scripts beside the
workshop project. They remain useful as transparent CLI fallbacks while the
same capabilities move into the portal.

The portal should expose hardware in this order:

1. discover serial ports and cameras;
2. assign follower, leader, and wrist camera;
3. verify saved calibration IDs and joint ranges;
4. calibrate only when a calibration is missing or intentionally replaced;
5. seed the raw current positions as hold targets, configure the bus, and connect
   with software motion locked;
6. read joint and camera observations;
7. explicitly arm bounded manual motion;
8. teach measured camera poses and tabletop bounds;
9. approve one scan or one exact manipulation plan at a time.

Place the arm in a compact supported pose for connection. LeRobot briefly
releases torque while configuring the motors, then torque holds the seeded pose.
This is different from permission to move: manual jogging, scanning, and plan
execution remain separately gated in software.

The scripts in `scripts/` are copied from the instructor's earlier
`my-SO-ARM101` project. Run them from this folder only when the portal setup
flow is unavailable.

LeRobot stores calibration by robot/teleoperator ID under
`~/.cache/huggingface/lerobot/calibration`. Reuse the same IDs whenever the
same physical arms are connected.

The instructor's existing environment currently uses LeRobot `0.4.4`. Keep
that version for the first hardware integration pass; upgrade to the current
stable release only after discovery, calibration loading, teleoperation, and
disconnect behavior pass again together.

## Install inside this workshop

From the workshop repository root, create the repository-owned environment and
install the complete stack:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The root `requirements.txt` includes the simulation, portal, and LeRobot stack
with Feetech support. No code or virtual environment from the earlier reference
project is required.

## What the portal persists

`09_real_robot_setup/output/workspace_profiles/<robot-id>.json` contains the
camera signature, OpenCV intrinsics, one safe travel posture, three taught joint
postures with measured camera-to-base transforms, workspace/table bounds,
placement points, gripper settings, and a measured multi-pose kinematic
consistency result. The profile also records the operator's explicit
confirmation that every tabletop coordinate uses the model's base datum. Real
scan and reconstruction artifacts
are kept separately under `09_real_robot_setup/output/pipeline/`; they never
overwrite the MuJoCo lessons.

The physical profile is invalid if the follower ID, camera resolution, or image
orientation changes. Version 4 stores camera rotation as counter-clockwise output
rotation and rejects older profiles. If the robot base, table, camera mount, or board
measurement changes, delete the profile from **Workspace setup** and recalibrate.
The supplied checkerboard is recorded printed-side up: printed +X follows its
arrow, printed +Y runs down the page, and printed +Z points into the tabletop.
The portal converts that board frame into robot +X-forward, +Y-left, +Z-up;
do not add another 90° or 180° correction to the measured board yaw.
Physical manipulation requires both manual joint-sign confirmation and a passing
three-pose hand-eye consistency check, plus the base-frame registration
confirmation. Recording three nearly identical poses, an inconsistent zero/sign
convention, or unregistered tabletop measurements does not unlock execution.

The portal deliberately does not perform LeRobot's initial interactive motor
calibration. If the saved follower calibration is missing or does not match the
connected bus, run `lerobot-calibrate` once with the intended ID, then return to
the portal. All later discovery, connection, camera calibration, survey-pose
teaching, scene construction, planning, approval, and bounded commands are in
the dashboard.
