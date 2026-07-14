# Week: Agentic Robotics — SO-ARM101

This repository grows from a simulated SO-ARM101 into a tool-using robot
agent. The code is organized as numbered lessons: run one visible idea, inspect
its output, and then carry that output into the next lesson.

MuJoCo remains the first teaching and acceptance runtime. The portal also has a
safety-gated LeRobot path for the physical follower: wrist video, joint state,
manual jogging, measured three-view RGB capture, learned-depth reconstruction,
reviewable planning, and one-shot approved commands. Physical manipulation stays
locked until camera, survey-pose, workspace, joint-direction, and preflight gates
all pass. Tabletop coordinates also require an explicit base-frame registration.
The survey poses are checked against forward kinematics; a checkbox alone cannot
unlock physical manipulation.

## Workshop journey

| Step | Participant milestone | Status |
|---|---|---|
| 00 | `00_getting_started/` — install and preflight | ✅ |
| 01 | `01_sim_manual_control/` — view and move every joint | ✅ |
| 02 | `02_record_replay/` — record and replay controls | ✅ |
| 03 | `03_kinematic_pick_place/` — understand IK and oracle manipulation | ✅ |
| 04 | `04_active_perception/` — scan with the wrist camera | ✅ |
| 05 | `05_semantic_scene/` — reconstruct geometry and name objects | ✅ |
| 06 | `06_agentic_manipulation/` — expose safe robot tools to an LLM | ✅ |
| 07 | `07_guided_full_pipeline/` — follow the complete loop together | ✅ |
| 08 | `08_workshop_portal/` — operate simulation or the gated physical path from one dashboard | ✅ |
| 09 | `09_real_robot_setup/` — physical profiles and LeRobot setup reference | ✅ |

## Setup

```bash
python3.11 -m venv .venv          # macOS / Linux
source .venv/bin/activate          # macOS / Linux
python -m pip install -r requirements.txt
python 00_getting_started/check_setup.py
```

The root requirements file installs the complete simulation, learned-depth,
portal, and LeRobot hardware stack. The project does not use the earlier
`my-SO-ARM101` project or its virtual environment. The first reconstruction
downloads the small indoor Depth Anything V2 checkpoint into the local Hugging
Face cache.

On Windows, create the environment with `py -3.11 -m venv .venv`, activate with
`.venv\Scripts\activate`, and then use `python` for the remaining commands.

> **macOS:** `replay.py` and non-headless `pick_place.py` use MuJoCo's passive
> viewer and must run with `mjpython`. Teleoperation and recording use the
> workshop's dedicated GLFW window and run with normal `python` on every
> platform. On Linux and Windows, use `python` in place of `mjpython` below.

Hosted semantics can use OpenRouter, OpenAI, Anthropic, or a custom
OpenAI-compatible endpoint. In the portal, enter the provider, model, and key
in **Model settings**; no `.env` file is required. Leave **Remember key** off on
shared workshop laptops. CLI-only lessons still read an OpenRouter key from the
current shell:

```bash
export OPENROUTER_API_KEY="..."
```

On Windows PowerShell, use `$env:OPENROUTER_API_KEY="..."` instead.

The portal defaults to `openrouter/free`, but that route can select different
models over time. It still requires an OpenRouter account and key. Confirm tool
calling before the workshop; if the routed model cannot read images, the portal
asks the participant for one explicit label per reconstructed object instead of
inventing semantic names.

## 01 — View and manual control

```bash
python 01_sim_manual_control/view.py
python 01_sim_manual_control/view.py --office
python 01_sim_manual_control/teleop.py
```

The full viewer's Control panel exposes all six actuators. The dedicated
teleoperation window uses `q/a w/s e/d r/f t/g y/h`; hold a key for continuous
motion and release it to stop changing the target. Press `0` to return every
target to zero. This window does not install MuJoCo's conflicting letter
shortcuts. A synchronized picture-in-picture in the upper-right shows the
robot's live wrist-mounted camera view. Its portrait orientation is intentional:
the simulated sensor is mounted 90° counter-clockwise while preserving the full
camera frame.

## 02 — Record and replay

```bash
python 02_record_replay/record.py my_skill
mjpython 02_record_replay/replay.py 02_record_replay/trajectories/my_skill.json --loop
```

In the recorder, press SPACE to begin, move the arm with the teleoperation
keys, and press SPACE again to save and exit. Recording saves the initial
positions, velocities, control targets, solver warm-start values, and one
control vector per simulation step. The live wrist-camera inset remains visible
while recording. Replay restores the saved values before applying the controls
again.

## 03 — Kinematic pick and place

```bash
mjpython 03_kinematic_pick_place/pick_place.py all
python 03_kinematic_pick_place/pick_place.py all --headless
```

This lesson intentionally reads object positions from MuJoCo. It is the oracle
baseline: if this layer fails, perception is not the problem. `ik.py` uses
damped-least-squares IK and `pick_place.py` executes an explicit tuck, pan,
hover, descend, grasp, carry, place, and release sequence.

## 04 — Active perception

```bash
python 04_active_perception/scan_workspace.py
```

The arm visits left, center, and right survey poses and captures settled RGB
images from its simulated wrist-camera model. The participant command opens an
animated MuJoCo window: the main view shows the arm moving, the portrait inset
shows the live wrist camera, and a green flash marks each saved frame. For an
automated or remote run, add `--headless`. Open:

```text
04_active_perception/output/latest_scan/contact_sheet.png
04_active_perception/output/latest_scan/scan.json
```

The JSON contains RGB paths, camera intrinsics, camera poses in the robot base
frame, named robot joints, timestamps, and image-quality measurements. The
equation `p_base = T_base_camera_cv @ p_camera_cv` makes the transform direction
explicit. The wrist roll stays fixed across all three captures, so coverage
comes from camera translation and arm parallax rather than rotating the image.
The scan never calls MuJoCo's depth renderer or stores hidden object poses. It
uses the same RGB, camera-calibration, and eye-in-hand pose contract required by
the physical wrist camera.

## Two scenes, two teaching purposes

- `scene_objects.xml` uses colored primitives and a pad for the oracle
  kinematics baseline.
- `scene_office.xml` uses a marker, eraser, and small cardboard box with no
  colored destination pad. This is the default perception scene.

Both scenes include `workshop_room.xml`: a compact standing desk, visible
C-clamp beneath the SO-ARM101 base at the near desk edge, floor, walls, and soft
overhead light. The tabletop remains the `z = 0` reference plane, so the room
changes presentation without changing the lesson's robot coordinates.

## 05–07 — Geometry, semantics, and the agent loop

```bash
python 04_active_perception/scan_workspace.py --scene empty
python 04_active_perception/scan_workspace.py
python 05_semantic_scene/reconstruct_scene.py
python 05_semantic_scene/make_object_crops.py
python 05_semantic_scene/evaluate_geometry.py
python 05_semantic_scene/label_scene.py \
  --label "object_1=small cardboard box" \
  --label "object_2=eraser" \
  --label "object_3=whiteboard marker" \
  --alias "object_1=box,package" \
  --alias "object_3=marker,pen,writing tool"

python 06_agentic_manipulation/guided_demo.py
python 07_guided_full_pipeline/run_guided_workshop.py --skip-prepare
python 07_guided_full_pipeline/evaluate_randomized_geometry.py
```

The sequence above deliberately uses manual labels for the first deterministic
test. Once it passes, run `python 05_semantic_scene/label_scene.py` without
`--label` values to test hosted OpenRouter semantics. Milestone 06 exposes six
validated robot tools and provides an offline guided loop plus a hosted
tool-calling loop whose `--model` defaults to `openrouter/free`. Replace
`<provider>/<model>:free` with a real OpenRouter model slug when selecting a
specific model. Milestone 07 uses hosted labels by default; add
`--manual-labels` for the deterministic offline path.

## 08 — Unified workshop portal

```bash
python 08_workshop_portal/app.py
```

Open `http://127.0.0.1:8000`. The portal reuses the existing milestone code and
artifacts; it does not replace the CLI lessons. It adds continuous browser-held
manual controls, synchronized simulation and portrait wrist-camera streams, and
a state-aware OpenRouter chat agent. A natural-language request can inspect
memory, run missing scan/reconstruction/labeling tools, ground an object, run
the existing safety preflight, and animate a verified MuJoCo execution.
Scripted motion temporarily owns the streams, then returns them to the
manual-control runtime.
The **Real SO-101** connection uses its own physical artifact directory and
measured camera transforms; it does not treat MuJoCo poses as sensor truth. Every
three-view scan needs a one-scan approval, and every safe manipulation pauses on
an expandable exact joint sequence before a one-shot physical approval. See
`08_workshop_portal/README.md` for the staged test procedure and safety scope.
At connection time, the portal copies the exact raw encoder positions into the
servo goal registers before LeRobot enables holding torque. Software motion stays
locked until a separate manual-jog, scan, or exact-plan approval. Keep the arm in
a compact supported pose because motor configuration briefly releases torque.

## Model credit

`models/so101/` comes from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
(`robotstudio_so101`, Apache-2.0), based on
[TheRobotStudio SO-ARM100/101](https://github.com/TheRobotStudio/SO-ARM100).
The workshop scenes and scripts are local additions.
