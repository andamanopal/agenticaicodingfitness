# 05 — Reconstruct the scene

This milestone begins with geometry, not object names. It compares the object
scan with an empty-table calibration, predicts metric depth from each RGB view,
aligns those predictions to the known table plane, and fuses only 3D points
that agree across calibrated camera poses.

The RGB camera remains the only physical sensor. Depth Anything V2 produces an
estimate from those pixels; MuJoCo's depth buffer and hidden object coordinates
remain unavailable. The empty images serve two purposes: foreground separation
and metric alignment to the robot-frame `z = 0` table plane. Robot and shadow
pixels are treated as unknown because they may hide an object.

Run these commands from the repository root:

```bash
python 04_active_perception/scan_workspace.py --scene empty
python 04_active_perception/scan_workspace.py
python 05_semantic_scene/reconstruct_scene.py
```

Then inspect:

```text
05_semantic_scene/output/latest_scene/
├── geometry.json
├── fused_cloud.ply
├── topdown.png
├── height_map.png
├── depth/
│   ├── diagnostics.json
│   ├── raw_depth_contact_sheet.png
│   ├── aligned_depth_contact_sheet.png
│   └── support_contact_sheet.png
└── masks/
    ├── survey_left.png
    ├── survey_left_occluder.png
    ├── survey_center.png
    ├── survey_center_occluder.png
    ├── survey_right.png
    └── survey_right_occluder.png
```

The first run downloads the 24.8M-parameter
`Depth-Anything-V2-Metric-Indoor-Small-hf` checkpoint into the Hugging Face
cache. Device selection is automatic: CUDA, then Apple Metal (`mps`), then CPU.
Use `--device cpu` when debugging portability.

`raw_depth_contact_sheet.png` is the model output before correction.
`aligned_depth_contact_sheet.png` shows the affine correction fitted from
visible table pixels to calibrated metric table depth. `support_contact_sheet.png`
shows whether each foreground point agrees with one, two, or three views. Only
points supported by at least two views enter `fused_cloud.ply`.

`geometry.json` contains anonymous `object_1`, `object_2`, and `object_3`
records. Their positions, dimensions, and yaw come from the fused point cloud.
`height_map.png` is the maximum fused surface height in each robot-base XY
cell. `topdown.png` overlays fused points in gray with the colored footprint
rectangles used by the planner.

Both images use a robot-centric presentation: the robot is at the bottom, `+X`
points forward/up the page, and `+Y` points toward the robot's left/left on the
page. The wrist camera does rotate in the world as upstream arm joints move.
Each frame's full `T_base_camera_cv` rotation accounts for that during
back-projection; manually rotating a captured image without also rotating its
intrinsics would make the geometry wrong.

On the real desk, the same method requires calibrated camera intrinsics,
eye-in-hand extrinsics, robot joint state, and a known table plane. Capture the
empty reference after mounting the arm and camera, then do not move the mount or
table. Learned monocular depth is still uncertain: inspect table RMSE,
cross-view retention, point-cloud shape, and planner confidence before motion.

After inspecting the estimated output, run the simulation-only evaluator:

```bash
python 05_semantic_scene/evaluate_geometry.py
```

That script is deliberately separate because it reads hidden MuJoCo object
positions. It is a teacher and debugging tool, never an input to the agent.

## Add object meaning

First project each anonymous 3D object back into the RGB views:

```bash
python 05_semantic_scene/make_object_crops.py
```

Open `output/latest_scene/crops/contact_sheet.png`. By default, the labeling
script sends that image to OpenRouter and uses the `openrouter/free` route:

```bash
export OPENROUTER_API_KEY="..."
python 05_semantic_scene/label_scene.py
```

The free route still requires an OpenRouter account and key. Image input travels
through OpenRouter to an available image-capable provider, so model availability
and latency can vary. To request a particular free model, pass its full slug:

```bash
python 05_semantic_scene/label_scene.py \
  --model "<provider>/<model>:free"
```

For a deterministic offline lesson, supply one manual label per object. The
presence of `--label` values selects manual mode and makes no hosted request:

```bash
python 05_semantic_scene/label_scene.py \
  --label "object_1=small cardboard box" \
  --label "object_2=eraser" \
  --label "object_3=whiteboard marker" \
  --alias "object_1=box,package" \
  --alias "object_3=marker,pen,writing tool"
```

The installed `openai` package is only an OpenAI-compatible HTTP client; these
requests use OpenRouter. The hosted model receives crops and object IDs only.
It returns labels, aliases, visible attributes, and semantic confidence. Metric
positions and dimensions always remain the reconstruction's values. Current
OpenRouter free limits are 20 requests per minute and 50 requests per day
without a $10 credit purchase, or 1,000 per day after purchasing at least $10
in credits.

Crop selection is deliberately simple: prefer a view with margin from the
image edge, then prefer a larger crop. It does not yet score gripper occlusion,
so inspect the contact sheet and use manual labels when the best crop is poor.
