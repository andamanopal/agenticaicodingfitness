# 08 · Workshop Portal

This local dashboard brings the existing numbered lessons into one browser
interface. It does not replace the CLI lessons or create a second robotics
pipeline.

## Start it

From the repository root:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python 08_workshop_portal/app.py
```

Open `http://127.0.0.1:8000`.

Click the model chip in the copilot heading, then choose OpenRouter, OpenAI,
Anthropic, or a custom OpenAI-compatible endpoint. Enter the model ID and API
key directly in the dialog; the portal does not require `.env` files or shell
environment variables. The key stays in the page unless you explicitly choose
to remember it in this browser.

The portal can also be launched explicitly with:

```bash
uvicorn app:app --app-dir 08_workshop_portal --host 127.0.0.1 --port 8000 \
  --no-access-log --timeout-graceful-shutdown 2
```

## What is reused

- Milestone 00 setup checks
- Milestone 03 kinematic demonstration
- Milestone 04 animated empty and occupied scans
- Milestone 05 reconstruction, crops, and semantic labeling
- Milestone 06 grounding, planning, preflight, approved execution, and verification

The portal owns one manual-control simulation thread. Scripted scans and
approved execution temporarily take over the two browser streams, then return
them to manual control.

Both camera panels use persistent MJPEG streams at approximately 24 FPS, so the
browser does not repeatedly download and replace standalone images. Portal
scans and approved rollouts are paced at `1×` wall-clock speed; capture holds
pause that clock so motion resumes naturally after each saved view.

The right side is a state-aware chat interface. Greetings stay conversational;
scene questions must read artifact memory, and action requests must use tool
results for every state claim. The model can call these portal tools:

1. inspect scene memory;
2. capture the empty-table calibration;
3. scan RGB objects from three wrist-camera viewpoints;
4. predict, align, cross-check, and fuse metric depth before creating crops;
5. name objects with the selected vision-capable model;
6. save a complete set of labels supplied by the participant when vision is
   unavailable;
7. ground and safety-check an object move;
8. execute and visually verify a safe MuJoCo plan.

For example, `Move the pen to the right side of the table` can complete missing
perception steps before planning. The depth network and language model never
receive MuJoCo depth or hidden object positions and cannot bypass the existing
preflight checks. Tool activity is shown inside the conversation, while raw
script output remains under **Pipeline details**.

During reconstruction, Scene memory updates through six inspectable stages:
foreground masks, raw learned depth, table alignment, cross-view agreement,
voxel fusion, and planner-object fitting. Expand the per-camera alignment row
to inspect scale, offset, and table RMSE. The image gallery appears as each
intermediate artifact is written, rather than waiting for semantic labels.

If the selected model cannot read images, automatic naming returns the ordered
reconstructed IDs instead of ending the workflow. Reply in the chat with a
complete mapping such as `The first, second, and third objects are a box, an
eraser, and a pen.` The copilot maps that order to scene memory and stores the
labels through the same semantic-scene contract used by vision. It cannot invent
or silently fill a label that was not present in your message.

## Controls

Hover or focus the small **Manual** information box above the streams to reveal
the joint controls. Hold the on-screen buttons or these keys:

- `Q/A` shoulder pan
- `W/S` shoulder lift
- `E/D` elbow flex
- `R/F` wrist flex
- `T/G` wrist roll
- `Y/H` gripper

The browser consumes these keys, so they do not reach MuJoCo's native viewer
shortcuts. Releasing a key, pointer, or browser focus releases all held motion.

## Connect the real SO-101

The root workshop installation already includes LeRobot and Feetech support in
this repository's `.venv`:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run these commands from the workshop repository root. Do not activate the
virtual environment from the earlier `my-SO-ARM101` reference project.

Find the follower port and calibrate it before opening the portal. Keep the same
robot ID when connecting from the portal:

```bash
lerobot-find-port
lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodemXXXXXXXX \
  --robot.id=workshop_so101
```

In the portal, choose **Real SO-101** and follow the setup modal. It lists
serial devices, OpenCV cameras, the installed LeRobot version, saved
follower/leader calibration IDs, and the selected follower's derived software
joint limits. Hardware preferences stay in browser local storage.

The physical-robot modal is a four-step safety gate:

1. clamp the base, clear the sweep, and put the arm in a compact supported pose;
2. scan and assign the follower serial port and wrist camera;
3. match the saved LeRobot calibration ID and inspect software joint limits;
4. review the exact configuration and connect with motion controls locked.

The leader assignment is stored for the later teleoperation bridge but is not
opened by the current follower-only runtime. Missing follower calibration is a
blocking error; the portal never starts an implicit physical calibration.
Before motor configuration enables holding torque, the connection path reads
every raw encoder position and writes that same raw value into the corresponding
goal register with acknowledged writes. LeRobot briefly releases torque while it
configures the bus, so support the arm during connection. Once connected, torque
holds the current pose but no motion command is allowed without a separate gate.

Connect remains a separate deliberate action. The real mode provides:

- the physical wrist-camera stream through LeRobot;
- live follower joint reads;
- explicitly armed, speed-limited manual joint jogging;
- a 300 ms server-side dead-man lease refreshed only while a browser control is held;
- absolute target clamping from the follower calibration ranges;
- LeRobot's per-command `max_relative_target` clipping as a second limit;
- a **Stop + disconnect** control that disarms and disconnects with torque
  disabled.

After connecting, click **Workspace setup**. This second four-step flow creates
a physical calibration profile scoped to the exact follower ID, camera
resolution, and image orientation:

1. print the supplied 8 × 6 inner-corner checkerboard at 100%, capture at least
   ten varied views, and solve camera intrinsics with OpenCV;
2. place the board flat at a measured table coordinate, then teach one safe
   travel posture and left, center, and right wrist-camera postures;
3. enter the reachable workspace, table footprint, placement coordinates, and
   gripper settings; explicitly confirm that every coordinate was measured from
   the same robot base datum, then verify the physical zero pose and five joint
   directions against the MuJoCo SO-101 model;
4. review whether scanning only or both scanning and manipulation are enabled.

Each taught camera posture stores a measured `T_base_camera_cv` from OpenCV
`solvePnP`; the real pipeline does not copy a simulated wrist transform. Empty
and occupied scans must use the same immutable calibration-profile hash and
repeat every taught arm pose within 1.5 degrees.
The red marker canonicalizes all four possible OpenCV checkerboard orderings.
Camera rotation is stored as counter-clockwise output rotation and translated to
LeRobot's opposite 90-degree convention at the device boundary. Physical profile
version 4 rejects profiles written by the older orientation logic; recapture them.
At capture time, the portal combines the validated fixed gripper-to-camera
transform with the actual measured joint positions instead of pretending the
servo landed exactly on its taught values. Reconstruction then warps the empty
image onto the occupied camera pose through the measured table plane before
foreground subtraction. Both registration overlap and foreground fraction are
visible in Scene memory.

The physical base datum is the tabletop point directly below the center of the
fixed robot base. Positive X points away from the table edge into the workspace;
positive Y points to the robot's left. The checkerboard origin, table bounds,
workspace bounds, and destinations must all use that one frame. Intrinsic camera
calibration cannot correct a bad ruler measurement here: a constant 20 mm frame
offset produces roughly a constant 20 mm grasp offset.
The target lies printed-side up. Its printed +Y runs down the page and its local
+Z points into the tabletop; the calibration transform applies that 180-degree
in-plane-frame flip automatically. The entered yaw describes only printed +X.

The three poses also form an objective kinematic gate. For each one, the portal
computes the MuJoCo gripper transform from the measured LeRobot joint angles and
derives the implied fixed gripper-to-camera transform. Manipulation stays locked
unless the poses are sufficiently different and those three implied transforms
agree within 25 mm translation and 10 degrees rotation. The manual joint-sign
confirmation is required as a second check, not accepted as the only evidence.
The survey set must include at least 15 degrees of gripper-frame rotation so a
constant board-origin mistake cannot hide behind three nearly parallel views.

## Test the physical pipeline

Use the chat while **Real SO-101** is active:

1. remove every object and ask `Capture the empty table`;
2. when the copilot stops, click **Approve one scan**; the approval is consumed
   by exactly one slow three-view survey;
3. place light desk objects without moving the robot, table, or camera mount and
   ask `Scan the objects and reconstruct the scene`;
4. approve that one scan, then inspect foreground masks, raw learned depth,
   table-aligned metric depth, cross-view agreement, the fused height map, and
   fitted footprints in Scene memory;
5. name crops with a vision model, or supply one complete ordered label list in
   chat;
6. ask `Move the pen to the left side` and wait for the preflight result;
7. expand the physical safety bar and review every exact joint target. Only a
   plan whose base-frame confirmation, camera hash, IK, joint limits, sampled
   robot/table/self collision, object clearance, and confidence checks all pass
   can be approved;
8. acknowledge that you reviewed the exact targets and cleared the physical
   workspace, then click **Execute this plan once** while keeping **Stop +
   disconnect** within reach.

The approval is bound to a hash of that exact joint sequence and is consumed
once. LeRobot sends the bounded targets to the follower at low speed. Completion
means only that the commanded joint sequence finished within tracking tolerance;
it does **not** prove the object was placed. The portal invalidates occupied
scene memory after every attempted physical motion—even a stopped or failed
attempt—reports an unverified result, and requires visual inspection plus a
fresh scan.

This is an open-loop teaching path for small, light, non-fragile objects in a
cleared workspace. It does not check unmodeled cables or clamps, grasp force, or
object slip. Never use it near people, liquids, sharp objects, or valuables.
**Stop + disconnect is a software stop, not a certified emergency stop.** Keep
the motor power connection physically reachable and cut power if software does
not stop the arm immediately.
