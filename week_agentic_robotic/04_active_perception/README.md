# 04 — Active perception

The wrist camera cannot see the whole desk from the home pose. This lesson
moves the arm to three safe survey poses, stops, captures an image, and stores
the camera geometry that belongs to those pixels. The saved frames are portrait
because the wrist sensor is rotated 90° counter-clockwise; their intrinsics and
camera poses already describe that orientation.

The wrist-roll target is identical in the left, center, and right poses. The
camera changes position and viewing direction with the other joints, producing
parallax and overlapping coverage without flipping the image between captures.

```bash
python 04_active_perception/scan_workspace.py
```

The default command opens an animated MuJoCo scan window. Watch the arm move
through the left, center, and right survey poses; the upper-right inset is the
live calibrated wrist camera, and its border flashes green when a frame is
saved. Press `Esc` to cancel safely. Use the fast non-visual path for automated
evaluation:

```bash
python 04_active_perception/scan_workspace.py --headless
```

Captures are written to a temporary sibling directory and published only after
all frames and `scan.json` are complete. Cancelling or interrupting a scan keeps
the previous valid output intact.

The default output is `04_active_perception/output/latest_scan/`:

```text
latest_scan/
├── scan.json
├── contact_sheet.png
├── survey_left.png
├── survey_center.png
└── survey_right.png
```

Open `contact_sheet.png` first. Then inspect `scan.json`. Every frame contains:

- the RGB filename;
- the camera intrinsics `K`;
- `T_base_camera_cv`, the camera pose in the robot base frame, defined by
  `p_base = T_base_camera_cv @ p_camera_cv`;
- the six named robot joint positions;
- simulation time and a basic brightness/contrast check.

That basic check does not measure object coverage, blur, or occlusion. Inspect
the contact sheet before treating the scan as useful scene memory.

The scan is intentionally RGB-only. It never calls MuJoCo's depth renderer and
never stores object body positions. Camera intrinsics and the eye-in-hand pose
are allowed because the physical pipeline obtains the same quantities from
camera calibration, robot kinematics, and hand-eye calibration. `scan.json`
declares `calibrated_multi_view_rgb_scan_v2`; later stages reject old
depth-buffer scans instead of silently mixing sensor contracts. Existing
RGB-only v1 captures remain readable because their stored evidence is
identical; new captures use the clearer generic scan name. Metric depth is
predicted later from these RGB frames.

The simulated arm is clamped to the near edge of a compact standing desk. The
office objects rest on its `z = 0` tabletop; the floor, walls, desk frame, and
clamp are visual context shared by every workshop scene.

Do not rotate an old PNG to repair its appearance. Camera pixels, intrinsics,
and `T_base_camera_cv` must come from the same capture. If the camera model or
mount changes, rerun both the empty and occupied scans.

Only robot joints are stored. MuJoCo also keeps the object poses in `qpos`, but
reading those would leak the answer that perception is supposed to estimate.
Objects must remain still from the first survey view until action completion;
otherwise the three images no longer describe one scene.

To compare with the earlier colored-primitives lesson:

```bash
python 04_active_perception/scan_workspace.py --scene primitives
```

The comparison is written to `output/primitives_scan/`, so it does not replace
the office scan used by the next lesson.

To inspect a different layout, randomize the office scene:

```bash
python 04_active_perception/scan_workspace.py --seed 7
```

This writes to `output/seed_007_scan/`.

The participant-facing `scan.json` records only `layout: "randomized"`; it does
not store the seed or generated coordinates. The randomized evaluator owns the
seed separately so it can reveal truth after reconstruction without exposing
that truth to the perception or agent pipeline.
