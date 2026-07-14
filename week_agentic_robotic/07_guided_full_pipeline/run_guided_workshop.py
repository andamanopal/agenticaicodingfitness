"""Run the complete workshop pipeline in its teaching order.

Usage:
    python 07_guided_full_pipeline/run_guided_workshop.py
    python 07_guided_full_pipeline/run_guided_workshop.py --manual-labels
    python 07_guided_full_pipeline/run_guided_workshop.py --execute
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run(arguments):
    command = [PYTHON, *arguments]
    print(f"\n$ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def prepare_scene(use_manual_labels, headless_scans):
    if not use_manual_labels and not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit(
            "OPENROUTER_API_KEY is not set. Add a free OpenRouter key, or rerun "
            "with --manual-labels."
        )

    run(["00_getting_started/check_setup.py"])
    scan_arguments = ["--headless"] if headless_scans else []
    run([
        "04_active_perception/scan_workspace.py",
        "--scene",
        "empty",
        *scan_arguments,
    ])
    run(["04_active_perception/scan_workspace.py", *scan_arguments])
    run(["05_semantic_scene/reconstruct_scene.py"])
    run(["05_semantic_scene/make_object_crops.py"])

    label_command = ["05_semantic_scene/label_scene.py"]
    if use_manual_labels:
        label_command.extend([
            "--label", "object_1=small cardboard box",
            "--label", "object_2=eraser",
            "--label", "object_3=whiteboard marker",
            "--alias", "object_1=box,package",
            "--alias", "object_3=marker,pen,writing tool",
        ])
    run(label_command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="writing tool")
    parser.add_argument(
        "--destination",
        choices=("left side", "center", "right side"),
        default="right side",
    )
    parser.add_argument("--manual-labels", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument(
        "--headless-scans",
        action="store_true",
        help="prepare both camera scans without animated MuJoCo windows",
    )
    parser.add_argument(
        "--headless-execution",
        action="store_true",
        help="execute without the animated manipulation rollout window",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approve", action="store_true")
    args = parser.parse_args()

    if not args.skip_prepare:
        prepare_scene(args.manual_labels, args.headless_scans)

    demo_command = [
        "06_agentic_manipulation/guided_demo.py",
        "--target", args.target,
        "--destination", args.destination,
    ]
    if args.execute:
        demo_command.append("--execute")
    if args.approve:
        demo_command.append("--approve")
    if args.headless_execution:
        demo_command.append("--headless")
    run(demo_command)


if __name__ == "__main__":
    main()
