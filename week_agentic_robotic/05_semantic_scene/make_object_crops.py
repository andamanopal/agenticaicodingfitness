"""Project reconstructed objects into RGB frames and save evidence crops.

Usage:
    python 05_semantic_scene/make_object_crops.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GEOMETRY = Path(__file__).resolve().parent / "output" / "latest_scene" / "geometry.json"
DEFAULT_SCAN = ROOT / "04_active_perception" / "output" / "latest_scan" / "scan.json"


def project(points_base, frame):
    transform = np.asarray(frame["T_base_camera_cv"], dtype=float)
    rotation = transform[:3, :3]
    position = transform[:3, 3]
    points_camera = (points_base - position) @ rotation
    intrinsics = np.asarray(frame["intrinsics"]["K"], dtype=float)
    depth = points_camera[:, 2]
    u = intrinsics[0, 0] * points_camera[:, 0] / depth + intrinsics[0, 2]
    v = intrinsics[1, 1] * points_camera[:, 1] / depth + intrinsics[1, 2]
    return u, v, depth


def object_box_points(obj):
    z_center = obj["position"][2]
    half_height = obj["dimensions"][2] / 2.0
    z_values = [max(0.0, z_center - half_height), z_center + half_height]
    return np.asarray([
        [corner[0], corner[1], z]
        for corner in obj["footprint_corners"]
        for z in z_values
    ])


def crop_object(image, frame, obj, padding=20):
    u, v, depth = project(object_box_points(obj), frame)
    if np.any(depth <= 0.01):
        return None

    left = max(0, int(np.floor(u.min())) - padding)
    top = max(0, int(np.floor(v.min())) - padding)
    right = min(image.width, int(np.ceil(u.max())) + padding)
    bottom = min(image.height, int(np.ceil(v.max())) + padding)
    if right - left < 12 or bottom - top < 12:
        return None

    return image.crop((left, top, right, bottom)), [left, top, right, bottom]


def save_contact_sheet(crops_dir, records):
    tile_width, tile_height, label_height = 320, 230, 42
    sheet = Image.new(
        "RGB",
        (tile_width * len(records), tile_height + label_height),
        "#11161c",
    )
    draw = ImageDraw.Draw(sheet)

    for index, record in enumerate(records):
        with Image.open(crops_dir / record["file"]) as crop:
            crop.thumbnail((tile_width - 20, tile_height - 20))
            x = index * tile_width + (tile_width - crop.width) // 2
            y = label_height + (tile_height - crop.height) // 2
            sheet.paste(crop, (x, y))
        draw.text(
            (index * tile_width + 10, 12),
            f"{record['object_id']} · {record['viewpoint']}",
            fill="#f1f5f9",
        )

    sheet.save(crops_dir / "contact_sheet.png")


def make_crops(geometry_path, scan_path):
    geometry_path = Path(geometry_path)
    geometry = json.loads(geometry_path.read_text())
    scan_path = Path(scan_path)
    scan = json.loads(scan_path.read_text())
    if geometry["scene_id"] != scan["scan_id"]:
        raise ValueError(
            "geometry and RGB scan describe different scenes; reconstruct first"
        )
    frames = {frame["viewpoint"]: frame for frame in scan["frames"]}
    crops_dir = geometry_path.parent / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    best_records = []
    objects = []
    for obj in geometry["objects"]:
        candidates = []
        for viewpoint in obj["evidence_views"]:
            if viewpoint not in frames:
                raise ValueError(f"scan is missing evidence view {viewpoint!r}")
            frame = frames[viewpoint]
            with Image.open(scan_path.parent / frame["rgb"]) as image:
                result = crop_object(image, frame, obj)
                image_size = image.size
            if result is None:
                continue

            crop, pixel_box = result
            filename = f"{obj['id']}_{viewpoint}.png"
            crop.save(crops_dir / filename)
            candidates.append({
                "viewpoint": viewpoint,
                "file": filename,
                "pixel_box": pixel_box,
                "pixel_area": crop.width * crop.height,
                "edge_margin": min(
                    pixel_box[0],
                    pixel_box[1],
                    image_size[0] - pixel_box[2],
                    image_size[1] - pixel_box[3],
                ),
            })

        if not candidates:
            raise RuntimeError(f"could not crop {obj['id']} in any evidence view")
        # A fully visible crop is better evidence than a larger crop clipped by
        # the image border. Area breaks ties between similarly centered views.
        candidates.sort(
            key=lambda candidate: (candidate["edge_margin"], candidate["pixel_area"]),
            reverse=True,
        )
        best = {"object_id": obj["id"], **candidates[0]}
        best_records.append(best)
        objects.append({
            "id": obj["id"],
            "best_crop": candidates[0]["file"],
            "crops": candidates,
        })

    crops_record = {"scene_id": geometry["scene_id"], "objects": objects}
    (crops_dir / "crops.json").write_text(json.dumps(crops_record, indent=2) + "\n")
    save_contact_sheet(crops_dir, best_records)
    print(f"saved crops for {len(objects)} objects -> {crops_dir}")
    return crops_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--scan", type=Path, default=DEFAULT_SCAN)
    args = parser.parse_args()
    make_crops(args.geometry, args.scan)


if __name__ == "__main__":
    main()
