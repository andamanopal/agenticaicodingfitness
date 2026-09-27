#!/usr/bin/env python3
"""Record every ```jev inline block in week24/*/TUTORIAL.md so DRY mode can replay it.

Maintainer tool. Needs TYPESAFE_API_KEY. Makes one live call per block.
    .venv/bin/python week24/common/record_inline.py            # all modules
    .venv/bin/python week24/common/record_inline.py 03_question_design
"""
import json
import os
import sys
from pathlib import Path

os.environ["JEV_RECORD"] = "1"
os.environ.setdefault("JEV_MODE", "live")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import jevkit  # noqa: E402


def blocks(md: str):
    out, buf, inside = [], [], False
    for line in md.splitlines():
        t = line.strip()
        if not inside and t.startswith("```jev"):
            inside, buf = True, []
        elif inside and t.startswith("```"):
            inside = False
            out.append("\n".join(buf))
        elif inside:
            buf.append(line)
    return out


def main():
    only = set(sys.argv[1:])
    n = bad = 0
    for tut in sorted(jevkit.WEEK24.glob("[0-9][0-9]_*/TUTORIAL.md")):
        if only and tut.parent.name not in only:
            continue
        for i, raw in enumerate(blocks(tut.read_text(encoding="utf-8")), 1):
            try:
                spec = json.loads(raw)
                resp = jevkit.ask(spec["state"], spec["questions"], model_id=spec.get("model"), quiet=True)
                jevkit.validate(resp, spec["questions"])
                n += 1
                print(f"✓ {tut.parent.name} block {i} · {resp['usage']['input_tokens']} tok")
            except Exception as e:  # noqa: BLE001
                bad += 1
                print(f"✕ {tut.parent.name} block {i}: {e}")
    print(f"═ recorded {n} inline block(s), {bad} failure(s)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
