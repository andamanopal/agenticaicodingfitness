#!/usr/bin/env python3
"""Interactive, explainable tutorial for **Sovereign AI on a DGX**.

A small control plane that serves a clickable web guide (static/guide.html) and,
for each chapter, lets you read the CONCEPT, view the demo SOURCE, and RUN it.

Two modes, auto-detected (see config.py):
  • REAL — a live OpenAI-compatible endpoint (Ollama / vLLM / llama.cpp on this
    laptop, or a DGX you point DGX_BASE_URL at). Genuine on-device inference.
  • SIM  — no endpoint reachable → a faithful DGX Spark simulator runs instead,
    so every concept is learnable with no GPU. Real commands are always shown.

Either way cloud cost is $0.00.

Launch (auto-picks a free port if 8092 is taken):

    .venv/bin/python week19/sovereign_dgx/tutorial_server.py
    # → http://127.0.0.1:8092
"""
from __future__ import annotations

import asyncio
import os
import shutil
import socket
import sys
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

PKG = Path(__file__).resolve().parent                 # …/week19/sovereign_dgx
ROOT = PKG.parents[1]                                 # …/agenticaicodingfitness
PY = str(ROOT / ".venv" / "bin" / "python")
if not Path(PY).exists():
    PY = sys.executable
DEMOS = PKG / "demos"

GUIDE_PORT = int(os.environ.get("NT_GUIDE_PORT", "8100"))


def _port_busy(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _pick_free_port(preferred: int, span: int = 40) -> int:
    for p in range(preferred, preferred + span):
        if not _port_busy(p):
            return p
    return preferred


STEPS = [
    {"id": "intro", "group": "Foundations", "kind": "concept",
     "title": "Ch 1 · Agent = Model + Harness", "level": "beginner",
     "desc": "Week 20 · Tutorial 01 of 12 · Phase: The Model (start here). This is the "
     "reasoning MODEL that everything else in Week 20 is a harness around — NVIDIA's open "
     "Nemotron 3 family, 'built for long-running, self-evolving agents'. An AGENT = a MODEL "
     "that reasons + a HARNESS (context, tools, memory, security); here you meet the model.\n\n"
     "In this tutorial:\n"
     "  • Ch 2 — The Nemotron 3 family, pick by task: compare Nano/Super/Ultra specs, "
     "DGX-Spark fit, and which model each multi-agent role wants.\n"
     "  • Ch 3 — Mamba-Transformer MoE + 1M context: why the hybrid architecture makes a "
     "1M-token context an agent can afford.\n"
     "  • Ch 4 — Reasoning (RLM), think then answer: watch a private REASON channel emitted "
     "before the ANSWER, live or simulated.\n"
     "  • Ch 5 — Tool-calling, a sovereign sub-agent: native function-calling reads a sensor "
     "and dispatches maintenance.\n"
     "  • Ch 6 — Run it, 1 Spark & 2 Sparks: real commands + fit math for Nano/Super on one "
     "Spark, or two linked over 200GbE for Ultra.\n\n"
     "Why it matters for sovereign agents:\n"
     "  • Open weights — run + fine-tune on YOUR DGX; nothing leaves the box.\n"
     "  • Hybrid Mamba-Transformer MoE — 1M-token context, few active params/token.\n"
     "  • RL-post-trained (NeMo Gym) — reasoning + tool-calling trained in.\n"
     "  • A family, not one model — Nano/Super/Ultra (+ RAG/Speech/Safety).\n\n"
     "Where it fits:\n"
     "This is the foundation — no prerequisites, start here. Everything else in Week 20 "
     "serves, orchestrates, guards, observes, and improves THIS model. "
     "Next: 02_nim_microservices (serve it).\n\n"
     "How to run:\n"
     "Click Run on each chapter: REAL against a Nemotron NIM/DGX endpoint via the 🔌 "
     "Connection panel, or SIM with no GPU ($0)."},
    {"id": "step01", "group": "The family", "kind": "run", "demo": "step01_family.py",
     "title": "Ch 2 · The Nemotron 3 family — pick by task", "level": "beginner",
     "desc": "Nano 30B-A3B · Super 120B-A12B · Ultra 550B-A55B (+ RAG/Speech/Safety). "
     "See the specs, DGX-Spark fit (1 vs 2 Sparks), and which model each multi-agent role wants."},
    {"id": "step02", "group": "The family", "kind": "run", "demo": "step02_architecture.py",
     "title": "Ch 3 · Mamba-Transformer MoE + 1M context", "level": "intermediate",
     "desc": "Why the architecture makes it 'built for long-running agents': near-linear "
     "Mamba layers + precise Transformer recall + MoE efficiency → a 1M-token context "
     "an agent can afford."},
    {"id": "step03", "group": "Reasoning", "kind": "run", "demo": "step03_reasoning.py",
     "title": "Ch 4 · Reasoning (RLM) — think, then answer", "level": "intermediate",
     "desc": "'RLM is the next thinking'. Nemotron emits a private REASON channel before "
     "the ANSWER — watch both, live or simulated. On your DGX the thinking stays yours."},
    {"id": "step04", "group": "Reasoning", "kind": "run", "demo": "step04_tool_calling.py",
     "title": "Ch 5 · Tool-calling — a sovereign sub-agent", "level": "intermediate",
     "desc": "Native function-calling turns Nemotron into a sub-agent: it reads a sensor "
     "and dispatches maintenance. Nano runs many cheap sub-agents; Super orchestrates."},
    {"id": "step05", "group": "Deploy", "kind": "run", "demo": "step05_run_on_dgx.py",
     "title": "Ch 6 · Run it — 1 Spark & 2 Sparks", "level": "advanced",
     "desc": "Stand up Nemotron: Nano/Super on one Spark (Ollama/NIM); link two Sparks over "
     "the QSFP 200GbE cable to run Ultra with tensor parallelism. Real commands + fit math."},
    {"id": "outro", "group": "Deploy", "kind": "concept",
     "title": "Appendix · How to get started", "level": "all levels",
     "desc": "Get the models + cards at build.nvidia.com; skills at github.com/NVIDIA/skills.\n\n"
     "Where this sits in Week 20 (Agent = Model + Harness):\n"
     "  App 1 (THIS) Nemotron — the MODEL.  Harness → NIM (serve), Dynamo (scale),\n"
     "  Data Flywheel + NeMo Gym (self-improve), OpenShell/NemoClaw (safe autonomy).\n\n"
     "You now have the open, reasoning, 1M-context model family that the rest of the\n"
     "stack turns into long-running, self-evolving, sovereign agents."},
]
STEP_BY_ID = {s["id"]: s for s in STEPS}

app = FastAPI(title="Nemotron Open Models — interactive tutorial")
_run_lock = asyncio.Lock()

# the model the user picked in the UI; injected into demo runs via DGX_MODEL.
SELECTED = {"model": config.MODEL}


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(PKG / "static" / "guide.html",
                        headers={"Cache-Control": "no-store, max-age=0"})


@app.get("/api/steps")
async def steps() -> dict:
    def public(s):
        return {k: s.get(k) for k in ("id", "group", "title", "desc", "kind", "level")} | \
               {"demo": s.get("demo")}
    real = config.MODE == "real"
    import ntsim as dgxsim
    models = config.list_local_models() if real else dgxsim.installed_models()
    if SELECTED["model"] not in models:        # keep the selection valid
        SELECTED["model"] = (models[0] if models else config.MODEL)
    return {"steps": [public(s) for s in STEPS], "mode": config.MODE,
            "conn": config.CONN, "conn_human": config.conn_human(),
            "model": SELECTED["model"], "base_url": config.BASE_URL, "models": models}


class ModelRequest(BaseModel):
    model: str


@app.post("/api/select_model")
async def select_model(req: ModelRequest) -> dict:
    SELECTED["model"] = req.model
    return {"ok": True, "model": req.model}


class ConnRequest(BaseModel):
    conn: str = "local"
    url: str | None = None
    key: str | None = None
    auth: str | None = None


@app.post("/api/connect")
async def connect(req: ConnRequest) -> dict:
    """Re-point the connection at runtime (local / tunnel / cloud) and re-detect."""
    config.apply_connection(req.model_dump())
    models = config.list_local_models()
    SELECTED["model"] = config.MODEL if config.MODEL in models else (models[0] if models else config.MODEL)
    return {"ok": True, "conn": config.CONN, "mode": config.MODE,
            "base_url": config.safe_base_url(), "endpoint_up": config.endpoint_up(),
            "model": SELECTED["model"], "models": models}


@app.get("/api/source/{step_id}")
async def source(step_id: str) -> dict:
    step = STEP_BY_ID.get(step_id)
    if not step or not step.get("demo"):
        return {"source": "(no source for this step)"}
    path = DEMOS / step["demo"]
    if not path.exists():
        return {"source": f"(missing file: {step['demo']})"}
    return {"source": path.read_text(), "filename": step["demo"]}


def _stream_demo(demo: str, timeout: float):
    async def gen():
        start = time.time()
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "DGX_MODEL": SELECTED["model"]}
        proc = await asyncio.create_subprocess_exec(
            PY, str(DEMOS / demo), cwd=str(PKG), env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        try:
            while True:
                try:
                    line = await asyncio.wait_for(
                        proc.stdout.readline(),
                        timeout=max(1, start + timeout - time.time()))
                except asyncio.TimeoutError:
                    proc.kill()
                    yield (f"\n⏱  step exceeded {timeout:.0f}s — killed.\n"
                           f"__EXIT__ 124 {time.time()-start:.1f}\n")
                    return
                if not line:
                    break
                yield line.decode(errors="replace")
            await proc.wait()
            yield f"__EXIT__ {proc.returncode} {time.time()-start:.1f}\n"
        finally:
            if proc.returncode is None:
                proc.kill()
    return gen()


class RunRequest(BaseModel):
    step_id: str


@app.post("/api/run")
async def run_step(req: RunRequest):
    step = STEP_BY_ID.get(req.step_id)
    if step is None or step.get("kind") != "run":
        async def err():
            yield f"step {req.step_id!r} is not runnable\n__EXIT__ 1 0\n"
        return StreamingResponse(err(), media_type="text/plain")

    # REAL inference + multi-step demos get more headroom.
    timeout = 360.0 if config.MODE == "real" else 120.0

    async def body():
        if _run_lock.locked():
            yield "⚠  another demo is already running — wait for it to finish.\n__EXIT__ 1 0\n"
            return
        async with _run_lock:
            yield f"$ {Path(PY).name} demos/{step['demo']}\n\n"
            async for chunk in _stream_demo(step["demo"], timeout):
                yield chunk
    return StreamingResponse(body(), media_type="text/plain")


@app.post("/api/cleanup")
async def cleanup() -> dict:
    removed = []
    sb = PKG / ".sandbox"
    if sb.exists():
        shutil.rmtree(sb); removed.append(".sandbox/")
    for pyc in PKG.rglob("__pycache__"):
        shutil.rmtree(pyc, ignore_errors=True)
    removed.append("__pycache__")
    return {"messages": [f"removed: {removed}"]}


if __name__ == "__main__":
    import uvicorn

    port = _pick_free_port(GUIDE_PORT)
    banner = ["", "  ▣  Nemotron Open Models — long-running, self-evolving, sovereign"]
    if config.MODE == "real":
        banner += [f"      ✓ REAL endpoint: {config.MODEL} @ {config.BASE_URL}",
                   "        demos run for real, fully on-device — cloud cost $0.00."]
    else:
        banner += ["      ◈ SIM mode — no endpoint reachable, simulating a DGX Spark.",
                   "        every concept is learnable with no GPU. Go REAL anytime:",
                   "        ollama run qwen3.6:35b-a3b-q8_0   (or set DGX_BASE_URL)"]
    if port != GUIDE_PORT:
        banner += [f"      ⚠ port {GUIDE_PORT} busy — using {port} (set DGX_GUIDE_PORT)."]
    banner += [f"      open  →  http://127.0.0.1:{port}", ""]
    print("\n".join(banner), flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port)
