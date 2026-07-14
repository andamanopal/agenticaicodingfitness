# 00 — Getting started

This milestone answers one question: **can this laptop run the workshop?**

From the repository root:

```bash
python3.11 -m venv .venv          # macOS / Linux
source .venv/bin/activate          # macOS / Linux
python -m pip install -r requirements.txt

python 00_getting_started/check_setup.py
python 01_sim_manual_control/view.py
```

On Windows, create the environment with `py -3.11 -m venv .venv`, activate with
`.venv\Scripts\activate`, and then use `python` for the remaining commands.

The preflight checks MuJoCo, Pillow, PyTorch, and Transformers; reports whether
learned depth will use CUDA, Apple Metal, or CPU; loads the SO-101; advances
physics; and renders one wrist frame. It does not download the depth model.
The second command opens the interactive MuJoCo window so you can see the arm
and use the Control sliders.

The keyboard teleoperation lesson uses its own shortcut-free GLFW window on
every platform:

```bash
python 01_sim_manual_control/teleop.py
```

The upper-right inset shows the wrist-mounted camera in real time while the
main view shows the complete arm. The portrait feed is the full camera frame
rotated 90° counter-clockwise, matching the simulated wrist mount.

On macOS, later lessons that explicitly use MuJoCo's passive viewer—replay and
the visual kinematic demo—still use `mjpython`.

## OpenRouter for the hosted lessons

Milestones 00–04 run locally. The default semantic-labeling and language-agent
paths in later milestones use OpenRouter:

```bash
export OPENROUTER_API_KEY="..."
```

On Windows PowerShell, use `$env:OPENROUTER_API_KEY="..."` instead.

`openrouter/free` is the default model route, but free access still requires an
OpenRouter account and API key. The `openai` package in `requirements.txt` is
only the OpenAI-compatible client used to send requests to OpenRouter; it does
not send workshop requests to OpenAI. Image inputs are routed through
OpenRouter's available providers.

Free-model availability and latency can change. Current limits are 20 requests
per minute and 50 requests per day without a $10 credit purchase. Purchasing at
least $10 in credits raises the free-model daily limit to 1,000 requests. Use
manual labels when you need a deterministic offline run.
