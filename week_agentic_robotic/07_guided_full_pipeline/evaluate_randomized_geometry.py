"""Evaluate learned metric-depth fusion across randomized object layouts.

Usage:
    python 07_guided_full_pipeline/evaluate_randomized_geometry.py
    python 07_guided_full_pipeline/evaluate_randomized_geometry.py --episodes 50
"""

import argparse
import io
import itertools
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "randomized_geometry"
BACKGROUND = ROOT / "04_active_perception" / "output" / "empty_scan" / "scan.json"

sys.path.insert(0, str(ROOT / "04_active_perception"))
sys.path.insert(0, str(ROOT / "05_semantic_scene"))
from reconstruct_scene import DEFAULT_VOXEL_SIZE, reconstruct  # noqa: E402
from scan_workspace import randomize_office_layout, scan_workspace  # noqa: E402


def truth_positions(seed):
    model_path = ROOT / "models" / "so101" / "scene_office.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    randomize_office_layout(model, data, seed)
    for _ in range(int(0.5 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    return np.asarray([
        data.body(name).xpos[:2].copy()
        for name in ("marker", "eraser", "small_box")
    ])


def match_centroids(estimated, truth):
    best_errors = None
    for order in itertools.permutations(range(len(estimated))):
        ordered = estimated[list(order)]
        errors = np.linalg.norm(ordered - truth, axis=1)
        if best_errors is None or errors.sum() < best_errors.sum():
            best_errors = errors
    return best_errors


def run_episode(seed):
    episode_dir = OUTPUT_DIR / f"seed_{seed:03d}"
    scan_dir = episode_dir / "scan"
    scene_dir = episode_dir / "scene"
    with redirect_stdout(io.StringIO()):
        scan_workspace(output_dir=scan_dir, scene="office", seed=seed)
        geometry = reconstruct(
            scan_dir / "scan.json",
            BACKGROUND,
            scene_dir,
            voxel_size=DEFAULT_VOXEL_SIZE,
        )

    if len(geometry["objects"]) != 3:
        return {"seed": seed, "passed": False, "object_count": len(geometry["objects"])}

    estimated = np.asarray([obj["position"][:2] for obj in geometry["objects"]])
    errors = match_centroids(estimated, truth_positions(seed))
    return {
        "seed": seed,
        "passed": bool(errors.max() <= 0.02),
        "object_count": 3,
        "mean_xy_error_mm": round(float(errors.mean() * 1000), 2),
        "max_xy_error_mm": round(float(errors.max() * 1000), 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--start-seed", type=int, default=0)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")

    if not BACKGROUND.exists():
        scan_workspace(output_dir=BACKGROUND.parent, scene="empty")

    results = []
    for seed in range(args.start_seed, args.start_seed + args.episodes):
        result = run_episode(seed)
        results.append(result)
        if result["passed"]:
            print(
                f"seed {seed:03d}: PASS  mean={result['mean_xy_error_mm']:.1f}mm "
                f"max={result['max_xy_error_mm']:.1f}mm"
            )
        else:
            print(f"seed {seed:03d}: REVIEW  clusters={result['object_count']}")

    passed = sum(result["passed"] for result in results)
    measured = [result for result in results if "mean_xy_error_mm" in result]
    summary = {
        "evaluation": "randomized learned metric-depth geometry",
        "episodes": len(results),
        "passed": passed,
        "pass_rate": round(passed / len(results), 3),
        "mean_xy_error_mm": None if not measured else round(
            float(np.mean([result["mean_xy_error_mm"] for result in measured])), 2
        ),
        "results": results,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\n{passed}/{len(results)} episodes passed the 20mm centroid threshold")
    print(f"summary -> {OUTPUT_DIR / 'summary.json'}")
    raise SystemExit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
