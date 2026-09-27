# Week 24 · Typed AI decisions with Jev (TypeSafe System One)

Hands-on course: learn to use **Jev**, TypeSafe's System One model. Jev doesn't
generate text; it returns **typed judgments** (`choice`, `noul`, `score`) that
your code can consume directly. Every module teaches the same idea: **Jev
judges, and your code decides**. Actions stay behind deterministic policy and
human approval.

## Launch the course (web app)

```bash
.venv/bin/python week24/00_jev_lab_runner/tutorial_server.py
# → http://127.0.0.1:8124   (auto-picks a free port; override with JEV_GUIDE_PORT)
```

The **Jev Lab Runner** has the same shape as Week 23's Lab Runner, with these features:

- one continuous step-by-step course with a sidebar, ✓ checkpoints, and progress saved server-side
- **▶ Run** on every lab and exercise file, with output streamed live from this machine
- **⚡ Ask Jev** inline blocks: edit a real request in the page and see typed answers drawn as probability bars
- a **⚡ Live / 📄 Dry run** switch plus a model picker (`jev-1.13.0` pinned, or the `jev-latest` / `jev-preview` aliases)
- 📊 architecture, sequence and chart diagrams per module (`diagrams.json`), and a built-in ⌨ terminal

### LIVE vs DRY

| Mode | What happens | Cost |
|---|---|---|
| **LIVE** | Real calls to `https://api.typesafe.ai/v1/systemone` | $0.042 per million input tokens, output free; a typical lab call is ≈ $0.00002 |
| **DRY** | No network. Answers recorded from real `jev-1.13.0` calls are replayed and labelled `RECORDED`. A request that was never recorded (e.g. one you edited) gets a neutral `PLACEHOLDER` (flat probabilities, noul = 0.5), which is **not a prediction** | $0 |

### The API key

`TYPESAFE_API_KEY` is read from the environment or the repo-root `.env` (already
set up). The server exports it only to lab subprocesses and the built-in terminal.
It is never sent to the browser, printed, or written to results. `.env` is
gitignored.

## Layout

```text
week24/
  00_jev_lab_runner/   the web app (tutorial_server.py + static/guide.html)
  common/jevkit.py     ~350-line stdlib Jev client every lab uses (ask/show/choice/noul/score)
  common/recorded/     real jev-1.13.0 answers replayed in DRY mode
  common/record_inline.py   maintainer tool: record every ```jev block (needs the key)
  NN_module/
    TUTORIAL.md        the course text (rendered by the runner, readable on GitHub too)
    diagrams.json      visuals for the 📊 section
    labs/labNN_*.py    runnable, commented labs
    exercises/exNN_*.py        starter files with TODOs + a free offline checker
    exercises/solutions/…      reference solutions
  jev_lab/             the original research scripts, cleanly named and runnable:
    jev_lab.py              7-workflow reference lab (selftest: 46 offline checks)
    jev_laya_benchmark.py   shared Jev/Laya benchmark runner
    test_jev_laya_benchmark.py   32 offline tests
  JEV practical tutorials … .md   the source research write-up (unchanged)
```

Run any lab without the web app:

```bash
.venv/bin/python week24/01_hello_jev/labs/lab01_first_call_raw.py
JEV_MODE=dry .venv/bin/python week24/01_hello_jev/labs/lab02_read_the_answer.py   # $0 replay
```

Refresh the DRY recordings after changing a lab (maintainers only; this makes live calls):

```bash
JEV_RECORD=1 .venv/bin/python week24/NN_module/labs/labNN_x.py
JEV_RECORD=1 .venv/bin/python week24/common/record_inline.py
```

## Changes from the original research files

- `jev_lab/jev_lab.py` and `jev_lab/jev_laya_benchmark.py` default to the **TypeSafe
  direct** provider (`--provider typesafe` / `--jev-provider typesafe`) because this
  course ships a TypeSafe key. OpenRouter still works with `--provider openrouter`
  plus `OPENROUTER_API_KEY`. The two offline tests that patched `OPENROUTER_API_KEY`
  now patch `TYPESAFE_API_KEY`, and all 32 pass.
- The original files with spaces in their names are left untouched.
