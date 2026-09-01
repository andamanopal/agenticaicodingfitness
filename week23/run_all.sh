#!/usr/bin/env bash
# Week 23 — launch the whole course with one command.
#   ./week23/run_all.sh          start everything: both hubs (8112/8113) + apps 01-12 (8100-8111)
#   ./week23/run_all.sh stop     stop everything
#   ./week23/run_all.sh status   show what's running
# Logs: /tmp/week23-<folder>.log
set -u
cd "$(dirname "$0")/.."

PY=".venv/bin/python"
[ -x "$PY" ] || { echo "✗ $PY not found — create it first:  uv venv .venv && uv pip install -r week23/01_nemotron_models/requirements.txt"; exit 1; }

APPS=(
  "00_stack_navigator:8112"  "00_lab_runner:8113"
  "01_nemotron_models:8100"  "02_nim_microservices:8101" "03_dynamo_serving:8102"
  "04_agent_skills:8103"     "05_aiq_research_lab:8104"  "06_nemoclaw:8105"
  "07_guardrails_openshell:8106" "08_nemo_relay:8107"    "09_inference_economics:8108"
  "10_nemo_gym_rl:8109"      "11_data_flywheel:8110"     "12_capstone_smart_hotel:8111"
)

up() { curl -s -o /dev/null --max-time 1 "http://127.0.0.1:$1/"; }

case "${1:-start}" in
  stop)
    pkill -f "week23/.*/tutorial_server.py" && echo "✓ all Week 23 servers stopped" || echo "nothing was running"
    ;;
  status)
    for a in "${APPS[@]}"; do f="${a%%:*}"; p="${a##*:}"
      if up "$p"; then echo "  ● $f  http://127.0.0.1:$p"; else echo "  ○ $f  :$p  (down)"; fi
    done
    ;;
  start|*)
    started=0; skipped=0
    for a in "${APPS[@]}"; do f="${a%%:*}"; p="${a##*:}"
      if up "$p"; then skipped=$((skipped+1)); continue; fi
      nohup "$PY" "week23/$f/tutorial_server.py" > "/tmp/week23-$f.log" 2>&1 &
      started=$((started+1))
    done
    sleep 3
    bad=0
    for a in "${APPS[@]}"; do p="${a##*:}"; up "$p" || { echo "  ✗ :$p failed — see /tmp/week23-*.log"; bad=1; }; done
    echo "✓ Week 23 up — started $started, already running $skipped"
    echo "  map  → http://127.0.0.1:8112   (Stack Navigator — start here)"
    echo "  labs → http://127.0.0.1:8113   (Lab Runner — hands-on track)"
    [ "$bad" = 0 ] || exit 1
    ;;
esac
