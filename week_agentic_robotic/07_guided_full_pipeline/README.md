# 07 — Guided full-pipeline practice

This is an instructor-led recap, not a challenge. The runner executes the same
numbered scripts participants have already used, in the same order:

```text
preflight → empty calibration → wrist scan → reconstruction → crops
          → semantics → find → plan → simulate → approve → execute → verify
```

Prepare everything and stop after planning/simulation:

```bash
export OPENROUTER_API_KEY="..."
python 07_guided_full_pipeline/run_guided_workshop.py
```

Also execute the simulated marker task:

```bash
python 07_guided_full_pipeline/run_guided_workshop.py --execute
```

The script pauses for `APPROVE`. Add `--approve` only for an unattended
simulation check. Hosted semantic labeling is the default and uses
`openrouter/free` through OpenRouter. The free route still requires an account
and `OPENROUTER_API_KEY`; image inputs are routed through OpenRouter's available
providers, so availability and latency can vary.

The empty-table and occupied-table scans open the animated MuJoCo scan window
by default. For an automated or remote preparation run, add
`--headless-scans`. Approved execution also opens an animated manipulation
rollout; add `--headless-execution` only for automated checks.

For a deterministic offline session, inspect the crop sheet and use the built-in
manual labels:

```bash
python 07_guided_full_pipeline/run_guided_workshop.py --manual-labels
```

The installed `openai` package is only the OpenAI-compatible client used to
contact OpenRouter. Current free limits are 20 requests per minute and 50
requests per day without a $10 credit purchase, or 1,000 per day after
purchasing at least $10 in credits.

Use `--skip-prepare` to reuse the current scan and semantic scene while
practicing different supported target phrases or destinations.

For step-by-step teaching, run Stages 04–06 directly. Stage 07 is an
orchestration wrapper: without `--skip-prepare`, it intentionally repeats setup,
both scans, reconstruction, crops, and labeling before entering the tool loop.

## Check that coordinates are not memorized

The simulation-only evaluator randomizes positions and yaw, reconstructs each
scene, and then reveals MuJoCo truth only to calculate centroid error:

```bash
python 07_guided_full_pipeline/evaluate_randomized_geometry.py
python 07_guided_full_pipeline/evaluate_randomized_geometry.py --episodes 50
```

This checks geometry, not semantic labeling or grasp success. Its truth access
is isolated from the participant pipeline.

The evaluator is the acceptance gate for the current learned metric-depth
fusion. It still starts from RGB and never feeds MuJoCo depth or hidden object
poses into reconstruction. Run it after changes and report the measured pass
rate; localization, grasp, collision, verification, and hardware-calibration
gates remain separate.
