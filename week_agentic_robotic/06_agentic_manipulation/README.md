# 06 — Agentic manipulation

The language model is the planner, not the motor controller. It may choose from
six narrow tools; deterministic Python validates and executes every motion.

Before this lesson, complete Milestone 05 so that
`05_semantic_scene/output/latest_scene/semantic_scene.json` exists.

## Learn the tool loop without an API

Plan and simulate only:

```bash
python 06_agentic_manipulation/guided_demo.py \
  --target "writing tool" --destination "right side"
```

Execute in MuJoCo and verify with matched wrist-camera views:

```bash
python 06_agentic_manipulation/guided_demo.py \
  --target "writing tool" --destination "right side" --execute
```

The script pauses for the exact word `APPROVE`, then opens an animated full-arm
rollout with a live wrist-camera inset. The display runs at 6× playback speed;
the underlying physics steps and controller targets are unchanged. Add
`--headless` only for automated checks. `--approve` bypasses the prompt and is
provided only for simulation tests; do not copy that behavior to hardware.

## Let a hosted language model choose the tools

```bash
export OPENROUTER_API_KEY="..."
python 06_agentic_manipulation/llm_agent.py \
  "Pick up the writing tool and place it on the right side of the table."
```

`llm_agent.py` uses OpenRouter and defaults to `openrouter/free`. The installed
`openai` package is the OpenAI-compatible HTTP client, not the request provider.
Free routing still needs an OpenRouter account and key; availability and latency
can vary. The free router selects a compatible model on the first turn, and the
script keeps that model for the rest of the tool loop. Select a particular free
model when needed:

```bash
python 06_agentic_manipulation/llm_agent.py \
  "Pick up the writing tool and place it on the right side." \
  --model "<provider>/<model>:free"
```

Current OpenRouter free limits are 20 requests per minute and 50 requests per
day without a $10 credit purchase, or 1,000 requests per day after purchasing
at least $10 in credits.

The model can call:

1. `scan_workspace`
2. `find_objects`
3. `plan_pick_and_place`
4. `simulate_plan`
5. `execute_plan`
6. `verify_plan`

`execute_plan` still asks the human for approval inside the tool handler. A
model-generated string can never approve its own action.

`simulate_plan` is intentionally a **kinematic preflight**. It checks IK,
wrist limits, reconstructed clearance, object width, observation quality,
table-depth alignment, clipped geometry, and semantic/geometry confidence. It
does not yet roll out the full trajectory or test swept-volume collision and
grasp contact physics.

Verification checks three pieces of visual evidence: the requested object left
its source, something appeared at the destination, and the removed/arrived
chromaticity signatures match. A failed motion locks the old scene and
requires a rescan; the same open-loop plan cannot be retried against stale
memory.

Execution currently accepts only the canonical office scan whose model
fingerprint matches `scene_office.xml`. Randomized scenes are used for geometry
evaluation, not silently recreated as a different execution world.

“Left” and “right” use the robot's perspective while it faces the workspace:
positive base-frame Y is left, and negative Y is right.

This milestone still uses MuJoCo. LeRobot will implement the same tool boundary
later; it is not yet a hardware execution path.
