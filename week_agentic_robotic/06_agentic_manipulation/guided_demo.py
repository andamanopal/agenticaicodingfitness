"""Follow the complete tool loop without requiring a language-model API.

Usage:
    python 06_agentic_manipulation/guided_demo.py
    python 06_agentic_manipulation/guided_demo.py --execute
"""

import argparse
import json

from robot_tools import DESTINATIONS, ExecutionCancelled, RobotTools


def show(title, value):
    print(f"\n{title}")
    print(json.dumps(value, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="writing tool")
    parser.add_argument("--destination", choices=DESTINATIONS, default="right side")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approve", action="store_true")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="execute without the animated full-arm and wrist-camera window",
    )
    args = parser.parse_args()

    tools = RobotTools(
        visualize_execution=args.execute and not args.headless,
    )
    show("1 · scan_workspace", tools.scan_workspace())
    found = tools.find_objects(args.target)
    show("2 · find_objects", found)
    if found["status"] != "found":
        raise SystemExit("target must resolve to exactly one object")

    plan = tools.plan_pick_and_place(found["matches"][0]["id"], args.destination)
    show("3 · plan_pick_and_place", plan)
    preflight = tools.simulate_plan(plan["plan_id"])
    show("4 · simulate_plan (kinematic preflight)", preflight)
    if not preflight["safe"] or not args.execute:
        return

    approval = "APPROVE" if args.approve else input(
        "\nType APPROVE to execute this plan in MuJoCo: "
    ).strip()
    try:
        execution = tools.execute_plan(plan["plan_id"], approval)
    except ExecutionCancelled as error:
        show("5 · execute_plan", {
            "status": "cancelled",
            "reason": str(error),
        })
        return
    show("5 · execute_plan", execution)
    if execution["status"] != "executed_unverified":
        return

    verification = tools.verify_plan(plan["plan_id"])
    show("6 · verify_plan", verification)
    raise SystemExit(0 if verification["verified"] else 1)


if __name__ == "__main__":
    main()
