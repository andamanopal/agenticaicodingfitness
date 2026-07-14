"""Compare reconstruction centroids with hidden MuJoCo truth.

This evaluator is not part of the perception pipeline. Its output may be shown
to participants for debugging, but its object positions must never reach the
planner or language-model tools.

Usage:
    python 05_semantic_scene/evaluate_geometry.py
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GEOMETRY = Path(__file__).resolve().parent / "output" / "latest_scene" / "geometry.json"
MODEL_PATH = ROOT / "models" / "so101" / "scene_office.xml"
TRUTH_ORDER = ["small_box", "eraser", "marker"]
TRUTH_DIMENSIONS = {
    "small_box": [0.044, 0.032, 0.050],
    "eraser": [0.064, 0.038, 0.022],
    "marker": [0.128, 0.018, 0.018],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    args = parser.parse_args()

    geometry = json.loads(args.geometry.read_text())
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    for _ in range(int(0.5 / model.opt.timestep)):
        mujoco.mj_step(model, data)

    if len(geometry["objects"]) != len(TRUTH_ORDER):
        raise SystemExit(
            f"expected {len(TRUTH_ORDER)} clusters, found {len(geometry['objects'])}"
        )

    print("object     truth name      XY error   height err  footprint err")
    print("---------  --------------  ---------  ----------  -------------")
    xy_errors = []
    height_errors = []
    footprint_errors = []
    for estimate, truth_name in zip(geometry["objects"], TRUTH_ORDER):
        estimated_position = np.asarray(estimate["position"])
        truth_position = data.body(truth_name).xpos.copy()
        xy_error = np.linalg.norm(estimated_position[:2] - truth_position[:2]) * 1000
        estimated_dimensions = np.asarray(estimate["dimensions"])
        truth_dimensions = np.asarray(TRUTH_DIMENSIONS[truth_name])
        height_error = abs(estimated_dimensions[2] - truth_dimensions[2]) * 1000
        estimated_footprint = np.sort(estimated_dimensions[:2])[::-1]
        truth_footprint = np.sort(truth_dimensions[:2])[::-1]
        footprint_error = np.max(np.abs(estimated_footprint - truth_footprint)) * 1000
        xy_errors.append(xy_error)
        height_errors.append(height_error)
        footprint_errors.append(footprint_error)
        print(
            f"{estimate['id']:<10} {truth_name:<15} "
            f"{xy_error:>7.1f}mm  {height_error:>8.1f}mm  "
            f"{footprint_error:>11.1f}mm"
        )

    print(f"\nmean XY centroid error: {np.mean(xy_errors):.1f}mm")
    print(f"mean height error: {np.mean(height_errors):.1f}mm")
    print(f"mean maximum footprint error: {np.mean(footprint_errors):.1f}mm")
    print("evaluator-only: these truth positions are not written into geometry.json")


if __name__ == "__main__":
    main()
