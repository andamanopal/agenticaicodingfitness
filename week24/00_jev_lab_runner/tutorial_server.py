#!/usr/bin/env python3
"""Jev Lab Runner — the step-by-step web app for Week 24 (typed AI decisions with Jev).

Same shape as Week 23's Lab Runner: it parses every week24/NN_*/TUTORIAL.md,
serves them as one continuous course (static/guide.html renders it), and runs
each module's labs/*.py and exercises/*.py server-side, streaming output live.

Jev-specific additions:
  • LIVE / DRY switch — LIVE calls api.typesafe.ai with TYPESAFE_API_KEY (read
    from the environment or the repo-root .env, server-side only, never sent to
    the browser); DRY replays answers recorded from real jev-1.13.0 calls ($0).
  • /api/jev — a localhost-only proxy behind the tutorial's inline "⚡ Ask Jev"
    blocks, so learners can edit a request in the page and see typed answers.

Launch (auto-picks a free port if 8124 is taken):

    .venv/bin/python week24/00_jev_lab_runner/tutorial_server.py
    # → http://127.0.0.1:8124
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import socket
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel

PKG = Path(__file__).resolve().parent                 # …/week24/00_jev_lab_runner
WEEK = PKG.parent                                     # …/week24
ROOT = WEEK.parent                                    # …/agenticaicodingfitness
sys.path.insert(0, str(WEEK / "common"))
import jevkit  # noqa: E402

PY = str(ROOT / ".venv" / "bin" / "python")
if not Path(PY).exists():
    PY = sys.executable
STATIC = PKG / "static"
GUIDE_PORT = int(os.environ.get("JEV_GUIDE_PORT", "8124"))

# The key lives server-side only: export it into THIS process so lab
# subprocesses and the built-in terminal inherit it (jev_lab.py needs the env).
if not os.environ.get("TYPESAFE_API_KEY") and jevkit.api_key():
    os.environ["TYPESAFE_API_KEY"] = jevkit.api_key()

MODELS = ["jev-1.13.0", "jev-latest", "jev-preview"]  # pinned first — the course default
RUN_ENV_KEYS = ("JEV_MODE", "JEV_MODEL")              # the ONLY env a browser may set
RUN_TIMEOUT = float(os.environ.get("LAB_RUN_TIMEOUT", "150"))
MODULE_RE = re.compile(r"^\d\d_[a-z0-9_]+$")
FILE_RE = re.compile(r"^(labs/lab\d\d_[a-z0-9_]+|exercises/ex\d\d_[a-z0-9_]+|"
                     r"exercises/solutions/ex\d\d_[a-z0-9_]+)\.py$")


def _port_busy(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _pick_free_port(preferred: int, span: int = 40) -> int:
    for p in range(preferred, preferred + span):
        if not _port_busy(p):
            return p
    return preferred


def modules() -> list[str]:
    """Every week24/NN_name/ folder with a TUTORIAL.md, in order (00 is this app)."""
    return [p.parent.name for p in sorted(WEEK.glob("[0-9][0-9]_*/TUTORIAL.md"))
            if MODULE_RE.match(p.parent.name) and not p.parent.name.startswith("00_")]


# ── TUTORIAL.md parser — pure-stdlib line scanner over H2 headings ─────────────
def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "section"


def _kind_of(title: str) -> str:
    t = title.strip()
    if re.match(r"^0\s*·", t):
        return "setup"
    if t.startswith("Labs"):
        return "labs"
    if t.startswith("Try it yourself"):
        return "exercises"
    if t.startswith("Troubleshooting"):
        return "troubleshooting"
    if t.startswith("Next") or t.startswith("What to build next"):
        return "next"
    return "step"


def _split_sections(text: str) -> list[dict]:
    """Split raw markdown on top-level '## ' headings, fence-aware."""
    lines = text.splitlines(keepends=True)
    bounds: list[tuple[int, str]] = []
    fence = False
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("```"):
            fence = not fence
            continue
        if not fence and ln.startswith("## "):
            bounds.append((i, ln[3:].strip()))
    first = bounds[0][0] if bounds else len(lines)
    sections = [{"id": "intro", "kind": "intro", "title": "Introduction", "md": "".join(lines[:first])}]
    seen: dict[str, int] = {"intro": 1}
    for j, (i, title) in enumerate(bounds):
        end = bounds[j + 1][0] if j + 1 < len(bounds) else len(lines)
        sid = _slug(title)
        seen[sid] = seen.get(sid, 0) + 1
        if seen[sid] > 1:
            sid = f"{sid}-{seen[sid]}"
        sections.append({"id": sid, "kind": _kind_of(title), "title": title, "md": "".join(lines[i:end])})
    return sections


def _doc_title(text: str, folder: str) -> str:
    m = re.search(r"^#\s+(.+)$", text, re.M)
    if not m:
        return folder
    h1 = m.group(1).strip()
    return h1.split("—", 1)[1].strip() if "—" in h1 else h1


def _doc_meta(text: str) -> dict:
    t = re.search(r"\*\*Time\*\*\s*([^·\n]+)", text)
    d = re.search(r"\*\*Difficulty\*\*\s*([^·\n]+)", text)
    c = re.search(r"\*\*Cost\*\*\s*([^\n]+)", text)
    return {"time": t.group(1).strip() if t else None,
            "difficulty": d.group(1).strip() if d else None,
            "cost": c.group(1).strip() if c else None}


def _docstring_title(path: Path) -> str:
    """First docstring line, minus a 'Lab 01-2 ·' / 'Exercise 01 ·' prefix."""
    try:
        head = path.read_text(encoding="utf-8", errors="replace")[:1500]
    except OSError:
        return ""
    m = re.search(r'"""\s*(.+)', head)
    if not m:
        return ""
    line = m.group(1).strip().rstrip('".').strip()
    return line.split("·", 1)[1].strip() if "·" in line else line


def _file_title(text: str, rel: str, path: Path | None = None) -> str:
    """'**labs/<file>** — title.' blurb from TUTORIAL.md, else the docstring, else the name."""
    fname = rel.rsplit("/", 1)[-1]
    fallback = (_docstring_title(path) if path else "") or fname[:-3].split("_", 1)[-1].replace("_", " ")
    m = re.search(r"\*\*" + re.escape(rel) + r"\*\*", text)
    if not m:
        return fallback
    line = text[m.end():].splitlines()[0].lstrip("*").strip().lstrip("—–-·:").strip()
    cuts = [p for p in (line.find(". "),) if p > 0]
    if cuts:
        line = line[:min(cuts)]
    line = line.rstrip(".").strip()
    return (line[:107].rstrip() + "…") if len(line) > 110 else (line or fallback)


def _list_files(folder: str, text: str) -> tuple[list[dict], list[dict]]:
    base = WEEK / folder
    labs = [{"file": f"labs/{p.name}", "title": _file_title(text, f"labs/{p.name}", p)}
            for p in sorted((base / "labs").glob("lab*.py")) if FILE_RE.match(f"labs/{p.name}")]
    exercises = []
    for p in sorted((base / "exercises").glob("ex*.py")):
        rel = f"exercises/{p.name}"
        if not FILE_RE.match(rel):
            continue
        sol = base / "exercises" / "solutions" / p.name
        exercises.append({"file": rel, "title": _file_title(text, rel, p),
                          "solution": f"exercises/solutions/{p.name}" if sol.is_file() else None})
    return labs, exercises


def _next_folder(sections: list[dict]) -> str | None:
    for s in sections:
        if s["kind"] == "next":
            m = re.search(r"\.\./(\d\d_[a-z0-9_]+)/", s["md"])
            if m:
                return m.group(1)
    return None


DIAGRAM_KEYS = ("architecture", "sequence", "charts")


def _load_diagrams(folder: str) -> dict | None:
    path = WEEK / folder / "diagrams.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("top level is not a JSON object")
    except Exception as e:  # noqa: BLE001 — a bad visuals file must never break the course
        print(f"  ⚠ {folder}/diagrams.json ignored: {e}", file=sys.stderr, flush=True)
        return None
    return {k: data.get(k) for k in DIAGRAM_KEYS}


def _insert_visualize(sections: list[dict]) -> None:
    sec = {"id": "visualize", "kind": "diagrams",
           "title": "📊 Visualize it — architecture · sequence · numbers", "md": ""}
    for i, s in enumerate(sections):
        if s.get("kind") == "labs":
            sections.insert(i, sec)
            return
    sections.append(sec)


def _build_entry(folder: str) -> dict:
    path = WEEK / folder / "TUTORIAL.md"
    diagrams = _load_diagrams(folder)
    entry: dict = {"num": folder[:2], "folder": folder, "title": folder,
                   "meta": {"time": None, "difficulty": None, "cost": None},
                   "sections": [], "labs": [], "exercises": [], "next": None, "diagrams": diagrams}
    try:
        text = path.read_text(encoding="utf-8")
        sections = _split_sections(text)
        labs, exercises = _list_files(folder, text)
        entry.update(title=_doc_title(text, folder), meta=_doc_meta(text), sections=sections,
                     labs=labs, exercises=exercises, next=_next_folder(sections))
    except Exception as e:  # noqa: BLE001 — never crash the course
        entry["parse_warning"] = f"parse failed: {e}"
        entry["sections"] = [{"id": "intro", "kind": "intro", "title": "Introduction", "md": ""}]
    if diagrams is not None:
        _insert_visualize(entry["sections"])
    return entry


_CACHE: dict[str, tuple[tuple, dict]] = {}


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return -1.0


def _dir_sig(path: Path) -> tuple:
    return tuple(sorted(p.name for p in path.glob("*.py"))) if path.is_dir() else ()


def course() -> list[dict]:
    out = []
    for folder in modules():
        base = WEEK / folder
        key = (_mtime(base / "TUTORIAL.md"), _mtime(base / "diagrams.json"),
               _dir_sig(base / "labs"), _dir_sig(base / "exercises"),
               _dir_sig(base / "exercises" / "solutions"))
        hit = _CACHE.get(folder)
        if not hit or hit[0] != key:
            hit = (key, _build_entry(folder))
            _CACHE[folder] = hit
        out.append(hit[1])
    return out


# ── the app ────────────────────────────────────────────────────────────────────
app = FastAPI(title="Jev Lab Runner — Week 24")
_run_lock = asyncio.Lock()


def _local_only(request: Request) -> None:
    """Guard against DNS rebinding: only accept requests addressed to localhost."""
    host = (request.headers.get("host") or "").rsplit(":", 1)[0].lower()
    if host not in ("127.0.0.1", "localhost", "[::1]", "::1"):
        raise HTTPException(403, "available on localhost only")


@app.get("/")
async def index():
    guide = STATIC / "guide.html"
    if guide.exists():
        return FileResponse(guide, headers={"Cache-Control": "no-store, max-age=0"})
    return PlainTextResponse("Jev Lab Runner backend is up — static/guide.html missing.")


@app.get("/static/{fname:path}")
async def static_file(fname: str):
    base = STATIC.resolve()
    target = (base / fname).resolve()
    if not str(target).startswith(str(base) + os.sep) or not target.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(target, headers={"Cache-Control": "no-store, max-age=0"})


@app.get("/api/course")
async def api_course() -> list[dict]:
    return course()


def _file_path(folder: str, rel: str) -> Path:
    """Strict allowlist: folder ∈ discovered modules, rel ∈ labs/ex/solutions patterns."""
    if folder not in modules() or not FILE_RE.match(rel or ""):
        raise HTTPException(404, "not found")
    path = WEEK / folder / rel
    if not path.is_file():
        raise HTTPException(404, "not found")
    return path


@app.get("/api/source")
async def api_source(folder: str, file: str):
    return PlainTextResponse(_file_path(folder, file).read_text(encoding="utf-8", errors="replace"))


# ── run history (sparklines) ───────────────────────────────────────────────────
HISTORY_PATH = PKG / ".run_history.json"
HISTORY_MAX = 50
_COST_RE = re.compile(r"\$(\d+\.\d+)")


def _load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _record_run(key: str, mode: str, code: int, secs: float, cost: float) -> None:
    try:
        hist = _load_json(HISTORY_PATH)
        runs = hist.setdefault(key, [])
        runs.append({"ts": int(time.time()), "mode": mode, "code": code,
                     "seconds": round(secs, 1), "cost": round(cost, 7)})
        del runs[:-HISTORY_MAX]
        HISTORY_PATH.write_text(json.dumps(hist), encoding="utf-8")
    except Exception as e:  # noqa: BLE001 — history must never break a run
        print(f"⚠ run history not saved: {e}", file=sys.stderr)


def _child_env(extra: dict[str, str]) -> dict[str, str]:
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    env.pop("JEV_RECORD", None)                       # learners never overwrite recordings
    for k in RUN_ENV_KEYS:
        v = extra.get(k)
        if v is not None:
            env[k] = str(v)
    if env.get("JEV_MODE") not in ("live", "dry"):
        env.pop("JEV_MODE", None)
    if env.get("JEV_MODEL") and env["JEV_MODEL"] not in MODELS:
        env.pop("JEV_MODEL", None)
    return env


class RunRequest(BaseModel):
    folder: str
    file: str
    env: dict[str, str] | None = None


@app.get("/api/history")
async def api_history() -> dict:
    return _load_json(HISTORY_PATH)


@app.post("/api/run")
async def api_run(req: RunRequest, request: Request):
    _local_only(request)
    path = _file_path(req.folder, req.file)
    env = _child_env(req.env or {})
    mode = env.get("JEV_MODE") or jevkit.mode()

    async def body():
        if _run_lock.locked():
            yield "⚠  another lab is already running — wait for it to finish.\n__EXIT__ 1 0\n"
            return
        async with _run_lock:
            start, cost = time.time(), 0.0
            yield f"$ {Path(PY).name} week24/{req.folder}/{req.file}   [{mode.upper()}]\n\n"
            proc = await asyncio.create_subprocess_exec(
                PY, str(path), cwd=str(ROOT), env=env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            try:
                while True:
                    try:
                        line = await asyncio.wait_for(proc.stdout.readline(),
                                                      timeout=max(1, start + RUN_TIMEOUT - time.time()))
                    except asyncio.TimeoutError:
                        proc.kill()
                        _record_run(f"{req.folder}/{req.file}", mode, 124, time.time() - start, cost)
                        yield f"\n⏱  exceeded {RUN_TIMEOUT:.0f}s — killed.\n__EXIT__ 124 {time.time()-start:.1f}\n"
                        return
                    if not line:
                        break
                    text = line.decode(errors="replace")
                    if text.lstrip().startswith("◆") and "LIVE" in text:
                        m = _COST_RE.search(text)
                        if m:
                            cost += float(m.group(1))
                    yield text
                await proc.wait()
                _record_run(f"{req.folder}/{req.file}", mode, proc.returncode or 0, time.time() - start, cost)
                yield f"__EXIT__ {proc.returncode} {time.time()-start:.1f}\n"
            finally:
                if proc.returncode is None:
                    proc.kill()
    return StreamingResponse(body(), media_type="text/plain")


# ── inline "⚡ Ask Jev" proxy ──────────────────────────────────────────────────
class JevRequest(BaseModel):
    state: object
    questions: dict
    model: str | None = None
    mode: str | None = None


_jev_sem = asyncio.Semaphore(2)
MAX_BODY = 64_000


@app.post("/api/jev")
async def api_jev(req: JevRequest, request: Request) -> dict:
    _local_only(request)
    model_id = req.model if req.model in MODELS else jevkit.DEFAULT_MODEL
    mode = req.mode if req.mode in ("live", "dry") else jevkit.mode()
    if mode == "live" and not jevkit.api_key():
        mode = "dry"
    size = len(json.dumps({"state": req.state, "questions": req.questions}, ensure_ascii=False))
    if size > MAX_BODY:
        raise HTTPException(413, f"request is {size} chars — keep inline experiments under {MAX_BODY}")
    if not req.questions or len(req.questions) > 64:
        raise HTTPException(422, "send between 1 and 64 questions")
    for k, q in req.questions.items():
        if not isinstance(q, dict) or q.get("type") not in ("choice", "noul", "score"):
            raise HTTPException(422, f"question '{k}' needs \"type\": \"choice\" | \"noul\" | \"score\"")
        if not q.get("instructions"):
            raise HTTPException(422, f"question '{k}' needs \"instructions\"")
    async with _jev_sem:
        try:
            resp = await asyncio.to_thread(jevkit.ask, req.state, req.questions, model_id=model_id,
                                           quiet=True, mode_override=mode)
        except jevkit.JevError as e:
            raise HTTPException(502, str(e))
    resp["_cost_usd"] = jevkit.cost_usd(resp)
    return resp


# ── checkpoint progress — server-side so it survives a browser switch ─────────
PROGRESS_PATH = PKG / "progress.json"
_progress_lock = threading.Lock()
_CKEY_RE = re.compile(r"^[0-9a-z_]+/[a-z0-9-]+/\d+$")


class ProgressRequest(BaseModel):
    key: str
    done: bool


@app.get("/api/progress")
async def api_progress_get() -> dict:
    return {"checkpoints": _load_json(PROGRESS_PATH)}


@app.post("/api/progress")
async def api_progress_set(req: ProgressRequest, request: Request) -> dict:
    _local_only(request)
    if not _CKEY_RE.match(req.key or ""):
        raise HTTPException(400, "bad checkpoint key")
    with _progress_lock:
        cks = _load_json(PROGRESS_PATH)
        if req.done:
            cks[req.key] = True
        else:
            cks.pop(req.key, None)
        try:
            PROGRESS_PATH.write_text(json.dumps(cks), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            raise HTTPException(500, f"could not save progress: {e}")
    return {"ok": True, "count": len(cks)}


# ── built-in terminal (local, single-user learning console — like Jupyter) ────
SHELL_TIMEOUT = 600.0
SHELL_BIN = os.environ.get("SHELL") or "/bin/zsh"
_shell_lock = asyncio.Lock()
_shell_proc: asyncio.subprocess.Process | None = None


class ShellRequest(BaseModel):
    cmd: str
    env: dict[str, str] | None = None


def _kill_tree(proc: asyncio.subprocess.Process) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass


@app.post("/api/shell")
async def api_shell(req: ShellRequest, request: Request):
    _local_only(request)
    cmd = (req.cmd or "").strip()
    env = {**_child_env(req.env or {}), "TERM": "dumb"}

    async def body():
        global _shell_proc
        if not cmd:
            yield "type a command first.\n__EXIT__ 1 0\n"
            return
        if _shell_lock.locked():
            yield "⚠  another command is already running — wait for it or hit ■ Stop.\n__EXIT__ 1 0\n"
            return
        async with _shell_lock:
            start = time.time()
            proc = await asyncio.create_subprocess_exec(
                SHELL_BIN, "-lc", cmd, cwd=str(ROOT), env=env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, start_new_session=True)
            _shell_proc = proc
            try:
                while True:
                    try:
                        line = await asyncio.wait_for(proc.stdout.readline(),
                                                      timeout=max(1, start + SHELL_TIMEOUT - time.time()))
                    except asyncio.TimeoutError:
                        _kill_tree(proc)
                        yield f"\n⏱  command exceeded {SHELL_TIMEOUT:.0f}s — killed.\n__EXIT__ 124 {time.time()-start:.1f}\n"
                        return
                    if not line:
                        break
                    yield line.decode(errors="replace")
                await proc.wait()
                yield f"__EXIT__ {proc.returncode} {time.time()-start:.1f}\n"
            finally:
                _shell_proc = None
                if proc.returncode is None:
                    _kill_tree(proc)
    return StreamingResponse(body(), media_type="text/plain")


@app.post("/api/shell/stop")
async def api_shell_stop(request: Request) -> dict:
    _local_only(request)
    proc = _shell_proc
    if proc is None or proc.returncode is not None:
        return {"stopped": False, "detail": "nothing is running"}
    _kill_tree(proc)
    return {"stopped": True, "detail": "killed the command's process group"}


# ── status: is a key present, and does TypeSafe accept it? ────────────────────
_KEY_CHECK: dict = {"checked": 0.0, "ok": None, "detail": ""}


def _probe_key() -> None:
    if not jevkit.api_key():
        _KEY_CHECK.update(checked=time.time(), ok=None, detail="no key")
        return
    try:
        names = [m.get("name") for m in jevkit.list_models()]
        _KEY_CHECK.update(checked=time.time(), ok=True, detail=", ".join(n for n in names if n))
    except Exception as e:  # noqa: BLE001
        _KEY_CHECK.update(checked=time.time(), ok=False, detail=f"{type(e).__name__}: {e}"[:160])


@app.get("/api/status")
async def api_status() -> dict:
    if time.time() - _KEY_CHECK["checked"] > 600:
        await asyncio.to_thread(_probe_key)
    has_key = bool(jevkit.api_key())
    recorded = len(list(jevkit.RECORDED.glob("*.json"))) if jevkit.RECORDED.is_dir() else 0
    if has_key and _KEY_CHECK["ok"] is not False:
        detail = (f"LIVE available — TYPESAFE_API_KEY found (not shown); api.typesafe.ai lists: "
                  f"{_KEY_CHECK['detail'] or 'unchecked'}")
    elif has_key:
        detail = f"key found but TypeSafe rejected it or is unreachable ({_KEY_CHECK['detail']}) — use DRY"
    else:
        detail = "DRY only — no TYPESAFE_API_KEY in the environment or repo-root .env"
    return {"has_key": has_key, "key_ok": _KEY_CHECK["ok"], "default_mode": jevkit.mode(),
            "models": MODELS, "default_model": jevkit.DEFAULT_MODEL, "recorded": recorded,
            "price_per_mtok": jevkit.PRICE_PER_MTOK, "run_timeout": RUN_TIMEOUT, "detail": detail}


if __name__ == "__main__":
    import uvicorn

    port = _pick_free_port(GUIDE_PORT)
    parsed = course()
    banner = ["", "  ⚡  Jev Lab Runner — Week 24 · typed AI decisions, one lab at a time"]
    if jevkit.api_key():
        banner += ["      ✓ TYPESAFE_API_KEY found (server-side only) — LIVE calls to jev-1.13.0,",
                   "        ≈ $0.00002 per lab call. Flip to DRY in the header for $0 replays."]
    else:
        banner += ["      ◈ DRY mode — no TYPESAFE_API_KEY; labs replay recorded jev-1.13.0 answers."]
    banner += [f"      ▤ course: {len(parsed)} modules · {sum(len(e['sections']) for e in parsed)} sections · "
               f"{sum(len(e['labs']) for e in parsed)} labs · {sum(len(e['exercises']) for e in parsed)} exercises"]
    if port != GUIDE_PORT:
        banner += [f"      ⚠ port {GUIDE_PORT} busy — using {port} (set JEV_GUIDE_PORT)."]
    banner += [f"      open  →  http://127.0.0.1:{port}", ""]
    print("\n".join(banner), flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port)
