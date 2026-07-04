#!/usr/bin/env python3
"""PART 2 · Agent Insights with Phoenix — read the trace  [INTERMEDIATE]

Agent Insights is the Phoenix UI over the spans NeMo Relay exports. This demo
renders a Phoenix-style TRACE of one Hermes turn: the span table (hermes-turn →
llm → tool: terminal / execute_code) with Status ✓, Latency and Total Cost, then a
per-span DETAIL view (the assistant message; the raw tool result is omitted). This
is how you LEARN from a run — spot the slow span, the expensive span, the failure.

Run:  python demos/step02_agent_insights.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import view  # noqa: E402

OPEN_UI = """\
# Agent Insights = a Phoenix instance reading the Relay's OTel spans
phoenix serve                       # → http://localhost:6006
relay = Relay(service="hermes-agent", export=["phoenix"]); relay.instrument()
# open the trace for a turn by its trace_id in the Phoenix Traces table
"""

# a Phoenix-style span tree for one trace (name, kind, status, latency_ms, cost_usd, depth)
SPANS = [
    ("hermes-turn",          "agent", "✓", 2840, 0.0121, 0),
    ("llm · plan",           "llm",   "✓",  610, 0.0032, 1),
    ("tool · terminal",      "tool",  "✓",  140, 0.0000, 1),
    ("llm · decide",         "llm",   "✓",  520, 0.0028, 1),
    ("tool · execute_code",  "tool",  "✓", 1180, 0.0000, 2),
    ("llm · summarize",      "llm",   "✓",  390, 0.0061, 1),
]


def _tree() -> None:
    print("  Phoenix · Traces  →  trace  a1f9c2  (Hermes turn)")
    print("  ┌──────────────────────────────┬───────┬────────┬──────────┬──────────┐")
    print("  │ span                         │ kind  │ status │ latency  │ cost     │")
    print("  ├──────────────────────────────┼───────┼────────┼──────────┼──────────┤")
    for name, kind, status, lat, cost, depth in SPANS:
        indent = "  " * depth
        label = (indent + name)[:28].ljust(28)
        print(f"  │ {label} │ {kind:<5} │   {status}    │ {lat:>5} ms │ ${cost:0.4f} │")
    print("  └──────────────────────────────┴───────┴────────┴──────────┴──────────┘")
    total_ms = SPANS[0][3]
    total_cost = sum(s[4] for s in SPANS[1:])
    print(f"  Status ✓   ·   Total Latency {total_ms} ms   ·   Total Cost ${total_cost:0.4f}")


def _detail() -> None:
    print("\n  ── span detail · llm · summarize ─────────────────────────────")
    print("  attributes:")
    print('    llm.model_name   = "gpt-4.4-mini"')
    print("    llm.token_count.prompt      = 812")
    print("    llm.token_count.completion  = 143")
    print("  input.value (assistant message):")
    print('    "Tests pass (12/12). I fixed the null check in parse() and')
    print('     re-ran the suite. Opening a PR against main."')
    print("  output / tool result: (omitted — expand the tool span to view)")


def main() -> None:
    view.banner("PART 2", "Agent Insights with Phoenix — read the trace", "INTERMEDIATE")
    view.mode_line()

    print("NeMo Relay exports OTel spans; Phoenix (Agent Insights) reads them:\n")
    print(OPEN_UI)
    print("The trace tree for one Hermes turn, as Agent Insights shows it:\n")
    _tree()
    _detail()

    print("\nReading the trace, you can immediately see: execute_code is the slow span")
    print("(1180 ms) and summarize is the expensive one ($0.0061) — candidates to optimize.\n")

    print(f"A one-line takeaway from the trace ({config.MODEL}):\n")
    view.generate("In two sentences, what does a Phoenix trace tree (spans with status, "
                  "latency and cost) let an engineer do that raw logs do not?",
                  max_tokens=200, title="what the trace tells you")
    print("\nTakeaway: the trace turns a black-box turn into a readable tree — you SEE the")
    print("slow and costly spans. Next: OPTIMIZE — route each call to the right-sized model.")


if __name__ == "__main__":
    main()
