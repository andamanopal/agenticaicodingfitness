#!/usr/bin/env python3
"""PART 1 · Observe — capture every tool & LLM call  [BEGINNER]

A long-running agent (Hermes, on OpenShell) makes many tool + model calls per turn.
NeMo Relay sits underneath and records each step as a telemetry SPAN (kind =
tool / llm / agent). This demo walks ONE Hermes turn being observed, printing each
captured span as it happens — the raw material Agent Insights + the flywheel use.

Run:  python demos/step01_observe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import view  # noqa: E402

INSTRUMENT = """\
# instrument the agent once — the Relay captures every call automatically
pip install nemo-relay openinference-instrumentation
from nemo_relay import Relay
relay = Relay(service="hermes-agent", export=["phoenix"])   # OTel under the hood
relay.instrument()          # now every tool + LLM call becomes a span
"""

# one Hermes turn, as the Relay observes it (each row = a captured span)
TURN = [
    ("hermes-agent", "relay", "span start: hermes-turn", "call"),
    ("hermes-agent", "tool:terminal", "ls ./repo && cat TODO.md", "call"),
    ("tool:terminal", "hermes-agent", "3 files · TODO.md (412 B)", "ret"),
    ("hermes-agent", "llm", "plan next action (gpt-4.4-mini)", "call"),
    ("llm", "hermes-agent", "call execute_code to run tests", "ret"),
    ("hermes-agent", "tool:execute_code", "pytest -q", "call"),
    ("tool:execute_code", "hermes-agent", "12 passed in 1.8s", "ret"),
]


def main() -> None:
    view.banner("PART 1", "Observe — capture every tool & LLM call", "BEGINNER")
    view.mode_line()

    print("Hermes is a long-running agent on OpenShell. NeMo Relay observes it:\n")
    print(INSTRUMENT)
    print("One Hermes turn, as the Relay captures it (each line = one span):\n")
    for i, (frm, to, msg, kind) in enumerate(TURN, 1):
        arrow = "←" if kind == "ret" else "→"
        knd = "tool" if "tool:" in frm + to else ("llm" if "llm" in frm + to else "agent")
        print(f"  span {i:>2} [{knd:<5}] {frm} {arrow} {to}")
        print(f"           {msg}")
    print("\n7 spans captured for one turn. That is the OBSERVE layer: nothing is guessed,")
    print("every tool and model call is recorded with timing + cost, ready to export.\n")

    print("A one-line natural-language summary of what was observed "
          f"({config.MODEL}):\n")
    view.generate("In two sentences, why must a long-running agent record every tool and "
                  "LLM call as telemetry before you can improve it?", max_tokens=200,
                  title="why observe every call")
    print("\nTakeaway: you can't optimize what you can't see. Next: read these spans in")
    print("Agent Insights (Phoenix) — the trace tree with status, latency and cost.")


if __name__ == "__main__":
    main()
