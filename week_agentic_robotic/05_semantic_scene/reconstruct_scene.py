"""Reconstruct desk objects by fusing calibrated learned metric depth.

Usage:
    python 05_semantic_scene/reconstruct_scene.py
    python 05_semantic_scene/reconstruct_scene.py --device mps
"""

import argparse
import json
import shutil
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCAN = ROOT / "04_active_perception" / "output" / "latest_scan" / "scan.json"
DEFAULT_BACKGROUND = ROOT / "04_active_perception" / "output" / "empty_scan" / "scan.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output" / "latest_scene"

SCAN_CONTRACT = "calibrated_multi_view_rgb_scan_v2"
SUPPORTED_SCAN_CONTRACTS = {
    SCAN_CONTRACT,
    "calibrated_multi_view_rgb_visual_hull_v1",
}
GEOMETRY_CONTRACT = "calibrated_multi_view_metric_depth_fusion_v1"
RECONSTRUCTION_METHOD = "table-aligned learned metric-depth fusion"
GEOMETRY_MODALITIES = ["rgb", "predicted_metric_depth"]
DEFAULT_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"

WORKSPACE_MIN = np.array([0.12, -0.17, 0.004])
WORKSPACE_MAX = np.array([0.34, 0.17, 0.12])
TABLE_MIN = np.array([-0.08, -0.38])
TABLE_MAX = np.array([0.60, 0.38])
FOREGROUND_THRESHOLD = 24.0
DEFAULT_VOXEL_SIZE = 0.003
DEFAULT_CONSISTENCY_TOLERANCE = 0.05
DEFAULT_MINIMUM_VIEWS = 1
MIN_TABLE_SAMPLES = 300
MIN_OBJECT_POINTS = 12
MAX_BACKGROUND_POSE_TRANSLATION_METERS = 0.035
MAX_BACKGROUND_POSE_ROTATION_DEGREES = 10.0
CLUSTER_COLORS = ["#55c7ff", "#55d6a8", "#b59cff", "#ffb86b"]


def configure_workspace_bounds(scan):
    """Use the calibrated physical workspace when the scan provides one."""
    global WORKSPACE_MIN, WORKSPACE_MAX, TABLE_MIN, TABLE_MAX
    workspace = scan.get("workspace_bounds")
    table = scan.get("table_bounds")
    if workspace is not None:
        minimum = np.asarray(workspace.get("minimum"), dtype=float)
        maximum = np.asarray(workspace.get("maximum"), dtype=float)
        if minimum.shape != (3,) or maximum.shape != (3,) or np.any(maximum <= minimum):
            raise ValueError("scan contains invalid workspace bounds")
        WORKSPACE_MIN = minimum
        WORKSPACE_MAX = maximum
    if table is not None:
        minimum = np.asarray(table.get("minimum"), dtype=float)
        maximum = np.asarray(table.get("maximum"), dtype=float)
        if minimum.shape != (2,) or maximum.shape != (2,) or np.any(maximum <= minimum):
            raise ValueError("scan contains invalid table bounds")
        TABLE_MIN = minimum
        TABLE_MAX = maximum


def load_scan(path):
    path = Path(path)
    scan = json.loads(path.read_text())
    frames = {frame["viewpoint"]: frame for frame in scan["frames"]}
    return path.parent, scan, frames


def write_diagnostics(output_dir, diagnostics):
    depth_dir = Path(output_dir) / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)
    destination = depth_dir / "diagnostics.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(diagnostics, indent=2) + "\n")
    temporary.replace(destination)


def match_background_photometry(rgb, background):
    """Compensate modest auto-exposure changes before background subtraction."""
    corrected = rgb.astype(float).copy()
    source = rgb[::3, ::3].reshape(-1, 3).astype(float)
    target = background[::3, ::3].reshape(-1, 3).astype(float)
    for channel in range(3):
        x = source[:, channel]
        y = target[:, channel]
        keep = np.ones(len(x), dtype=bool)
        gain = 1.0
        offset = 0.0
        for _ in range(4):
            design = np.column_stack([x[keep], np.ones(keep.sum())])
            gain, offset = np.linalg.lstsq(design, y[keep], rcond=None)[0]
            residual = np.abs(gain * x + offset - y)
            threshold = np.quantile(residual, 0.70)
            keep = residual <= max(threshold, 3.0)
        gain = float(np.clip(gain, 0.65, 1.5))
        offset = float(np.clip(offset, -55.0, 55.0))
        corrected[..., channel] = corrected[..., channel] * gain + offset
    return np.clip(corrected, 0, 255).astype(np.uint8)


def register_background_to_view(background, background_frame, frame):
    """Warp the empty image onto the occupied view through the table plane."""
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "background registration requires opencv-python from requirements.txt"
        ) from error

    background_transform = np.asarray(
        background_frame["T_base_camera_cv"],
        dtype=float,
    )
    transform = np.asarray(frame["T_base_camera_cv"], dtype=float)
    translation_delta = float(np.linalg.norm(
        transform[:3, 3] - background_transform[:3, 3]
    ))
    relative_rotation = background_transform[:3, :3].T @ transform[:3, :3]
    cosine = np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0)
    rotation_delta = float(np.degrees(np.arccos(cosine)))
    if (
        translation_delta > MAX_BACKGROUND_POSE_TRANSLATION_METERS
        or rotation_delta > MAX_BACKGROUND_POSE_ROTATION_DEGREES
    ):
        raise RuntimeError(
            "empty and occupied camera poses differ too much for table-plane "
            f"registration ({translation_delta * 1000:.1f} mm, "
            f"{rotation_delta:.1f} degrees); repeat both scans"
        )

    # Sample the table plane and keep only points in front of BOTH survey
    # cameras. Extreme table corners can fall behind a wrist camera, which does
    # not define a valid homography; a least-squares fit over the visible points
    # is exact because they are coplanar.
    grid_x, grid_y = np.meshgrid(
        np.linspace(TABLE_MIN[0], TABLE_MAX[0], 9),
        np.linspace(TABLE_MIN[1], TABLE_MAX[1], 9),
    )
    plane_points = np.column_stack([
        grid_x.reshape(-1),
        grid_y.reshape(-1),
        np.zeros(grid_x.size),
    ])
    source_u, source_v, source_depth = project(plane_points, background_frame)
    target_u, target_v, target_depth = project(plane_points, frame)
    visible = (source_depth > 0.05) & (target_depth > 0.05)
    if int(visible.sum()) < 4:
        raise RuntimeError("the calibrated table plane is behind a survey camera")
    source_pixels = np.column_stack([source_u[visible], source_v[visible]]).astype(np.float32)
    target_pixels = np.column_stack([target_u[visible], target_v[visible]]).astype(np.float32)
    homography, _ = cv2.findHomography(source_pixels, target_pixels, 0)
    if homography is None or not np.isfinite(homography).all():
        raise RuntimeError("table-plane registration produced an invalid homography")
    height, width = background.shape[:2]
    aligned = cv2.warpPerspective(
        background,
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    valid = cv2.warpPerspective(
        np.ones((height, width), dtype=np.uint8),
        homography,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    ).astype(bool)
    valid_fraction = float(valid.mean())
    if valid_fraction < 0.70:
        raise RuntimeError(
            "empty-table registration leaves too little overlapping image area; "
            "repeat both scans from the taught poses"
        )
    return aligned, valid, {
        "translation_delta_meters": round(translation_delta, 6),
        "rotation_delta_degrees": round(rotation_delta, 4),
        "valid_pixel_fraction": round(valid_fraction, 4),
    }


def foreground_masks(scan_path, background_path, output_dir):
    scan_dir, scan, frames = load_scan(scan_path)
    background_dir, background_scan, background_frames = load_scan(background_path)
    configure_workspace_bounds(scan)
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    for label, record in (
        ("object scan", scan),
        ("empty-table calibration", background_scan),
    ):
        is_rgb_only = (
            record.get("perception_contract") in SUPPORTED_SCAN_CONTRACTS
            and record.get("sensor_modalities") == ["rgb"]
            and not any("depth" in frame for frame in record.get("frames", []))
        )
        if not is_rgb_only:
            raise ValueError(
                f"{label} uses a legacy or privileged sensor contract; "
                "recapture it with the current RGB-only scanner"
            )

    if scan["camera"] != background_scan["camera"]:
        raise ValueError("foreground and background scans use different cameras")
    if scan.get("workspace_profile_sha256") != background_scan.get(
        "workspace_profile_sha256"
    ):
        raise ValueError(
            "foreground and background scans use different physical workspace profiles"
        )
    if scan.get("workspace_bounds") != background_scan.get("workspace_bounds"):
        raise ValueError("foreground and background scans use different workspace bounds")
    if set(frames) != set(background_frames):
        raise ValueError("foreground and background scans use different viewpoints")

    rgb_images = {}
    masks = {}
    occluders = {}
    registration_records = {}
    aligned_background_tiles = []
    mask_tiles = []
    for viewpoint, frame in frames.items():
        background_frame = background_frames[viewpoint]
        intrinsics = np.asarray(frame["intrinsics"]["K"], dtype=float)
        background_intrinsics = np.asarray(
            background_frame["intrinsics"]["K"], dtype=float
        )
        if frame["intrinsics"]["width"] != background_frame["intrinsics"]["width"]:
            raise ValueError(f"image width mismatch for {viewpoint}")
        if frame["intrinsics"]["height"] != background_frame["intrinsics"]["height"]:
            raise ValueError(f"image height mismatch for {viewpoint}")
        if not np.allclose(intrinsics, background_intrinsics, atol=1e-6, rtol=0.0):
            raise ValueError(f"camera intrinsics mismatch for {viewpoint}")

        rgb = np.asarray(Image.open(scan_dir / frame["rgb"]).convert("RGB"))
        raw_background = np.asarray(
            Image.open(background_dir / background_frame["rgb"]).convert("RGB")
        )
        if rgb.shape != raw_background.shape:
            raise ValueError(f"image size mismatch for {viewpoint}")
        rgb_images[viewpoint] = rgb
        background, valid, registration = register_background_to_view(
            raw_background,
            background_frame,
            frame,
        )
        background[~valid] = rgb[~valid]
        registration_records[viewpoint] = registration
        aligned_background_image = Image.fromarray(background)
        aligned_background_image.save(
            masks_dir / f"{viewpoint}_background_aligned.png"
        )
        aligned_background_tiles.append((viewpoint, aligned_background_image))

        # The empty view identifies robot, gripper, and shadow pixels. They are
        # unknown during consistency checks because they may hide an object.
        table_color = np.median(background[valid], axis=0)
        table_difference = np.abs(background.astype(float) - table_color).mean(axis=2)
        occluder_image = Image.fromarray(
            (table_difference > 30.0).astype(np.uint8) * 255
        ).filter(ImageFilter.MaxFilter(9))
        occluders[viewpoint] = (np.asarray(occluder_image) > 0) | ~valid
        occluder_image.save(masks_dir / f"{viewpoint}_occluder.png")

        matched_rgb = match_background_photometry(rgb, background)
        color_change = np.abs(
            matched_rgb.astype(float) - background.astype(float)
        ).mean(axis=2)
        mask_image = Image.fromarray(
            (color_change > FOREGROUND_THRESHOLD).astype(np.uint8) * 255
        )
        mask_image = mask_image.filter(ImageFilter.MaxFilter(5))
        mask_image = mask_image.filter(ImageFilter.MinFilter(5))
        masks[viewpoint] = (np.asarray(mask_image) > 0) & valid
        registration_records[viewpoint]["foreground_pixel_fraction"] = round(
            float(masks[viewpoint].mean()),
            5,
        )
        mask_image = Image.fromarray(masks[viewpoint].astype(np.uint8) * 255)
        mask_image.save(masks_dir / f"{viewpoint}.png")
        mask_tiles.append((viewpoint, mask_image.convert("RGB")))

    save_contact_sheet(
        masks_dir / "background_alignment_contact_sheet.png",
        aligned_background_tiles,
        "empty table warped to occupied camera pose",
    )
    save_contact_sheet(
        masks_dir / "foreground_contact_sheet.png",
        mask_tiles,
        "white = changed tabletop pixels",
    )

    return (
        scan,
        background_scan,
        frames,
        rgb_images,
        masks,
        occluders,
        registration_records,
    )


@lru_cache(maxsize=4)
def load_depth_backend(model_name, requested_device):
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    except ImportError as error:
        raise RuntimeError(
            "learned depth requires torch and transformers; run "
            "`python -m pip install -r requirements.txt` in this project venv"
        ) from error

    if requested_device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = requested_device

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access CUDA")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but PyTorch cannot access Apple Metal")

    try:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForDepthEstimation.from_pretrained(model_name)
    except Exception as error:
        raise RuntimeError(
            f"could not load depth model {model_name!r}; the first run needs "
            "internet access so Hugging Face can cache the checkpoint"
        ) from error
    model.to(device)
    model.eval()
    return torch, processor, model, device


def predict_depths(rgb_images, model_name, requested_device):
    torch, processor, model, device = load_depth_backend(
        model_name,
        requested_device,
    )
    predictions = {}
    for viewpoint, rgb in rgb_images.items():
        image = Image.fromarray(rgb)
        inputs = processor(images=image, return_tensors="pt")
        inputs = {name: value.to(device) for name, value in inputs.items()}
        with torch.inference_mode():
            predicted = model(**inputs).predicted_depth
        predicted = torch.nn.functional.interpolate(
            predicted.unsqueeze(1),
            size=rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        depth = predicted.detach().float().cpu().numpy()
        depth[~np.isfinite(depth)] = np.nan
        predictions[viewpoint] = depth
    return predictions, device


def table_depth_from_calibration(frame, image_shape):
    height, width = image_shape
    intrinsics = np.asarray(frame["intrinsics"]["K"], dtype=float)
    transform = np.asarray(frame["T_base_camera_cv"], dtype=float)
    rotation = transform[:3, :3]
    camera_position = transform[:3, 3]

    pixel_y, pixel_x = np.indices((height, width))
    rays = np.stack(
        [
            (pixel_x - intrinsics[0, 2]) / intrinsics[0, 0],
            (pixel_y - intrinsics[1, 2]) / intrinsics[1, 1],
            np.ones_like(pixel_x),
        ],
        axis=-1,
    )
    denominator = rays @ rotation[2, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = -camera_position[2] / denominator
    points_camera = rays * depth[..., None]
    points_base = points_camera @ rotation.T + camera_position
    valid = (
        np.isfinite(depth)
        & (depth > 0.05)
        & (points_base[..., 0] >= TABLE_MIN[0])
        & (points_base[..., 0] <= TABLE_MAX[0])
        & (points_base[..., 1] >= TABLE_MIN[1])
        & (points_base[..., 1] <= TABLE_MAX[1])
    )
    return depth, valid


def fit_depth_alignment(predicted, expected, valid):
    x = predicted[valid]
    y = expected[valid]
    finite = np.isfinite(x) & np.isfinite(y) & (x > 0.01) & (y > 0.01)
    x = x[finite]
    y = y[finite]
    if len(x) < MIN_TABLE_SAMPLES:
        raise RuntimeError(
            f"only {len(x)} table pixels were available for metric-depth alignment"
        )

    lower, upper = np.quantile(x, [0.01, 0.99])
    keep = (x >= lower) & (x <= upper)
    x = x[keep]
    y = y[keep]
    for _ in range(5):
        centered = x - x.mean()
        variance = float(centered @ centered)
        if variance < 1e-9 or np.ptp(y) < 0.015:
            scale = float(np.median(y / np.maximum(x, 1e-6)))
            offset = 0.0
        else:
            scale = float(centered @ (y - y.mean()) / variance)
            offset = float(y.mean() - scale * x.mean())
        residual = scale * x + offset - y
        median = float(np.median(residual))
        mad = float(np.median(np.abs(residual - median)))
        threshold = max(0.008, 3.0 * 1.4826 * mad)
        inliers = np.abs(residual - median) <= threshold
        if inliers.all() or inliers.sum() < MIN_TABLE_SAMPLES:
            break
        x = x[inliers]
        y = y[inliers]

    if not 0.1 <= scale <= 5.0 or not -1.0 <= offset <= 1.0:
        raise RuntimeError(
            f"implausible depth alignment scale={scale:.3f}, offset={offset:.3f}m"
        )
    residual = scale * x + offset - y
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    status = "pass" if rmse <= 0.02 else "review"
    return scale, offset, {
        "status": status,
        "table_samples": int(len(x)),
        "scale": round(scale, 6),
        "offset_meters": round(offset, 6),
        "table_rmse_meters": round(rmse, 6),
    }


def align_depths(predictions, frames, masks, occluders):
    aligned = {}
    records = {}
    for viewpoint, predicted in predictions.items():
        expected, table_pixels = table_depth_from_calibration(
            frames[viewpoint],
            predicted.shape,
        )
        table_pixels &= ~masks[viewpoint]
        table_pixels &= ~occluders[viewpoint]
        scale, offset, record = fit_depth_alignment(
            predicted,
            expected,
            table_pixels,
        )
        corrected = predicted * scale + offset
        corrected[(corrected <= 0.02) | (corrected > 2.0)] = np.nan
        aligned[viewpoint] = corrected
        records[viewpoint] = record
    return aligned, records


def project(points_base, frame):
    transform = np.asarray(frame["T_base_camera_cv"], dtype=float)
    rotation = transform[:3, :3]
    position = transform[:3, 3]
    points_camera = (points_base - position) @ rotation
    depth = points_camera[:, 2]
    intrinsics = np.asarray(frame["intrinsics"]["K"], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = intrinsics[0, 0] * points_camera[:, 0] / depth + intrinsics[0, 2]
        v = intrinsics[1, 1] * points_camera[:, 1] / depth + intrinsics[1, 2]
    return u, v, depth


def backproject_foreground(depth, frame, mask, pixel_stride):
    pixel_y, pixel_x = np.nonzero(mask)
    selected = (pixel_x % pixel_stride == 0) & (pixel_y % pixel_stride == 0)
    pixel_x = pixel_x[selected]
    pixel_y = pixel_y[selected]
    z = depth[pixel_y, pixel_x]
    valid = np.isfinite(z) & (z > 0.02)
    pixel_x = pixel_x[valid]
    pixel_y = pixel_y[valid]
    z = z[valid]

    intrinsics = np.asarray(frame["intrinsics"]["K"], dtype=float)
    points_camera = np.column_stack(
        [
            (pixel_x - intrinsics[0, 2]) * z / intrinsics[0, 0],
            (pixel_y - intrinsics[1, 2]) * z / intrinsics[1, 1],
            z,
        ]
    )
    transform = np.asarray(frame["T_base_camera_cv"], dtype=float)
    points_base = points_camera @ transform[:3, :3].T + transform[:3, 3]
    inside = np.all(
        (points_base >= WORKSPACE_MIN) & (points_base <= WORKSPACE_MAX),
        axis=1,
    )
    return points_base[inside], pixel_x[inside], pixel_y[inside]


def cross_view_filter(
    aligned_depths,
    frames,
    masks,
    occluders,
    tolerance,
    minimum_views,
    pixel_stride,
):
    viewpoint_names = list(frames)
    accepted_points = []
    accepted_sources = []
    accepted_support = []
    support_images = {}
    candidate_count = 0

    for source_index, source_name in enumerate(viewpoint_names):
        points, pixel_x, pixel_y = backproject_foreground(
            aligned_depths[source_name],
            frames[source_name],
            masks[source_name],
            pixel_stride,
        )
        candidate_count += len(points)
        support = np.ones(len(points), dtype=np.uint8)
        for target_name in viewpoint_names:
            if target_name == source_name:
                continue
            u, v, projected_depth = project(points, frames[target_name])
            target_x = np.rint(u).astype(int)
            target_y = np.rint(v).astype(int)
            target_depth = aligned_depths[target_name]
            inside = (
                (projected_depth > 0.02)
                & (target_x >= 0)
                & (target_x < target_depth.shape[1])
                & (target_y >= 0)
                & (target_y < target_depth.shape[0])
            )
            comparable = np.zeros(len(points), dtype=bool)
            comparable[inside] = (
                ~occluders[target_name][target_y[inside], target_x[inside]]
                & masks[target_name][target_y[inside], target_x[inside]]
                & np.isfinite(target_depth[target_y[inside], target_x[inside]])
            )
            agrees = np.zeros(len(points), dtype=bool)
            agrees[comparable] = (
                np.abs(
                    target_depth[target_y[comparable], target_x[comparable]]
                    - projected_depth[comparable]
                )
                <= tolerance
            )
            support += agrees

        support_image = np.zeros_like(masks[source_name], dtype=np.uint8)
        support_image[pixel_y, pixel_x] = support
        support_images[source_name] = support_image
        keep = support >= minimum_views
        accepted_points.append(points[keep])
        accepted_sources.append(
            np.full(keep.sum(), source_index, dtype=np.uint8)
        )
        accepted_support.append(support[keep])

    if not accepted_points or not any(len(points) for points in accepted_points):
        raise RuntimeError(
            "no depth points agreed across camera views; inspect aligned depth, "
            "camera calibration, and foreground masks"
        )
    return (
        np.concatenate(accepted_points),
        np.concatenate(accepted_sources),
        np.concatenate(accepted_support),
        support_images,
        candidate_count,
        viewpoint_names,
    )


def fuse_points(points, sources, support, voxel_size):
    indices = np.floor((points - WORKSPACE_MIN) / voxel_size).astype(int)
    unique, inverse = np.unique(indices, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    fused = np.column_stack(
        [np.bincount(inverse, weights=points[:, axis]) for axis in range(3)]
    ) / counts[:, None]
    mean_support = np.bincount(inverse, weights=support) / counts
    view_masks = np.zeros(len(unique), dtype=np.uint8)
    np.bitwise_or.at(view_masks, inverse, np.left_shift(1, sources).astype(np.uint8))
    return fused, view_masks, counts, mean_support


def cluster_fused_points(points, voxel_size):
    cell_size = max(voxel_size, 0.003)
    cells = np.floor((points[:, :2] - WORKSPACE_MIN[:2]) / cell_size).astype(int)
    original_cells = {tuple(cell) for cell in cells}
    expanded = {
        (cell[0] + dx, cell[1] + dy)
        for cell in original_cells
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
    }
    remaining = set(expanded)
    component_by_cell = {}
    component_id = 0
    while remaining:
        start = remaining.pop()
        stack = [start]
        component = [start]
        while stack:
            x, y = stack.pop()
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                neighbor = (x + dx, y + dy)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
                    component.append(neighbor)
        for cell in component:
            component_by_cell[cell] = component_id
        component_id += 1

    groups = {}
    for point_index, cell in enumerate(cells):
        groups.setdefault(component_by_cell[tuple(cell)], []).append(point_index)
    components = [
        np.asarray(indices, dtype=int)
        for indices in groups.values()
        if len(indices) >= MIN_OBJECT_POINTS
    ]
    return components


def convex_hull(points):
    unique = sorted({(float(point[0]), float(point[1])) for point in points})
    if len(unique) <= 1:
        return np.asarray(unique, dtype=float)

    def cross(origin, a, b):
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (
            a[1] - origin[1]
        ) * (b[0] - origin[0])

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def minimum_area_rectangle(points, margin):
    # Trim the farthest points before hull fitting so a few noisy depth
    # outliers cannot inflate the object footprint (single-view depth is noisy).
    points = np.asarray(points, dtype=float)
    if len(points) >= 12:
        centroid = points.mean(axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)
        points = points[distances <= np.percentile(distances, 75)]
    hull = convex_hull(points)
    if len(hull) < 2:
        center = hull[0] if len(hull) else np.zeros(2)
        length = width = margin * 2
        yaw = 0.0
    else:
        best = None
        edges = np.roll(hull, -1, axis=0) - hull
        angles = np.unique(np.arctan2(edges[:, 1], edges[:, 0]) % (np.pi / 2.0))
        for angle in angles:
            long_axis = np.array([np.cos(angle), np.sin(angle)])
            short_axis = np.array([-np.sin(angle), np.cos(angle)])
            basis = np.column_stack([long_axis, short_axis])
            local = hull @ basis
            minimum = local.min(axis=0) - margin
            maximum = local.max(axis=0) + margin
            dimensions = maximum - minimum
            area = float(dimensions[0] * dimensions[1])
            if best is None or area < best[0]:
                best = (area, basis, minimum, maximum)
        _, basis, minimum, maximum = best
        dimensions = maximum - minimum
        if dimensions[1] > dimensions[0]:
            basis = basis[:, ::-1]
            minimum = minimum[::-1]
            maximum = maximum[::-1]
            dimensions = dimensions[::-1]
        center_local = (minimum + maximum) / 2.0
        center = center_local @ basis.T
        length, width = dimensions
        yaw = float(np.arctan2(basis[1, 0], basis[0, 0]))

    long_axis = np.array([np.cos(yaw), np.sin(yaw)])
    short_axis = np.array([-np.sin(yaw), np.cos(yaw)])
    corners = np.asarray(
        [
            center - long_axis * length / 2 - short_axis * width / 2,
            center + long_axis * length / 2 - short_axis * width / 2,
            center + long_axis * length / 2 + short_axis * width / 2,
            center - long_axis * length / 2 + short_axis * width / 2,
        ]
    )
    return center, float(length), float(width), yaw, corners


def project_mask_to_plane(frame, mask):
    """Cast foreground-mask pixels onto the z=0 table plane.

    The XY footprint from a pixel ray intersecting the known table plane does
    not depend on predicted depth, so it is free of the monocular depth-scale
    bias that shifts back-projected points. Returns an Nx2 array of base-frame
    points; tall objects project slightly outside their base, which the
    outlier-trimmed rectangle fit absorbs.
    """
    transform = np.asarray(frame["T_base_camera_cv"], dtype=float)
    intrinsics = np.asarray(frame["intrinsics"]["K"], dtype=float)
    rotation = transform[:3, :3]
    camera_position = transform[:3, 3]
    pixel_y, pixel_x = np.nonzero(mask)
    if len(pixel_x) == 0:
        return np.empty((0, 2))
    rays = np.stack(
        [
            (pixel_x - intrinsics[0, 2]) / intrinsics[0, 0],
            (pixel_y - intrinsics[1, 2]) / intrinsics[1, 1],
            np.ones(len(pixel_x)),
        ],
        axis=-1,
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        denominator = rays @ rotation[2, :]
        depth = -camera_position[2] / denominator
        points_base = (rays * depth[:, None]) @ rotation.T + camera_position
    valid = (
        np.isfinite(depth)
        & (depth > 0.05)
        & (points_base[:, 0] >= TABLE_MIN[0])
        & (points_base[:, 0] <= TABLE_MAX[0])
        & (points_base[:, 1] >= TABLE_MIN[1])
        & (points_base[:, 1] <= TABLE_MAX[1])
    )
    return points_base[valid, :2]


def bit_count(value):
    return int(value).bit_count()


def describe_component(
    indices,
    points,
    view_masks,
    sample_counts,
    mean_support,
    viewpoint_names,
    alignment_records,
    voxel_size,
    footprint_xy=None,
):
    component_points = points[indices]
    # Prefer the depth-bias-free mask-to-plane footprint for XY and extent;
    # fall back to the metric-depth points when too few footprint points fell
    # on this cluster. Height still comes from the depth points below.
    fit_xy = (
        footprint_xy
        if footprint_xy is not None and len(footprint_xy) >= 12
        else component_points[:, :2]
    )
    center_xy, length, width, yaw, corners = minimum_area_rectangle(
        fit_xy,
        margin=voxel_size,
    )
    height = float(np.quantile(component_points[:, 2], 0.95) + voxel_size / 2.0)
    # DEMO: monocular depth on synthetic renders carries no usable object-height
    # signal (every object reads ~10 cm), so cap height low enough that the
    # top-down grasp grazes the tabletop and reaches short objects. On real
    # hardware with in-distribution depth, raise this cap toward WORKSPACE_MAX.
    height = float(np.clip(height, voxel_size * 2.0, 0.035))
    center = [center_xy[0], center_xy[1], height / 2.0]

    combined_views = int(np.bitwise_or.reduce(view_masks[indices]))
    evidence = [
        name
        for index, name in enumerate(viewpoint_names)
        if combined_views & (1 << index)
    ]
    multi_view_fraction = float(
        np.mean([bit_count(value) >= 2 for value in view_masks[indices]])
    )
    average_support = float(np.average(mean_support[indices], weights=sample_counts[indices]))
    boundary_flags = []
    if height >= WORKSPACE_MAX[2] - voxel_size:
        boundary_flags.append("height_ceiling")
    if np.any(component_points[:, :2] <= WORKSPACE_MIN[:2] + voxel_size) or np.any(
        component_points[:, :2] >= WORKSPACE_MAX[:2] - voxel_size
    ):
        boundary_flags.append("workspace_side")
    if any(alignment_records[name]["status"] != "pass" for name in evidence):
        boundary_flags.append("table_alignment_review")
    if len(component_points) < 25:
        boundary_flags.append("sparse_depth")

    confidence = min(
        0.97,
        0.38
        + 0.10 * len(evidence)
        + 0.16 * min(average_support / 3.0, 1.0)
        + 0.12 * multi_view_fraction
        + 0.12 * min(len(component_points) / 80.0, 1.0),
    )
    confidence -= 0.10 * len(boundary_flags)
    return {
        "position": np.round(center, 4).tolist(),
        "dimensions": np.round([length, width, height], 4).tolist(),
        "yaw_radians": round(yaw, 4),
        "footprint_corners": np.round(corners, 4).tolist(),
        "voxel_count": int(len(component_points)),
        "geometry_voxel_count": int(len(component_points)),
        "geometry_basis": "cross_view_consistent_metric_depth",
        "geometry_confidence": round(max(0.1, confidence), 3),
        "multi_view_voxel_fraction": round(multi_view_fraction, 3),
        "mean_cross_view_support": round(average_support, 3),
        "depth_sample_count": int(sample_counts[indices].sum()),
        "boundary_flags": boundary_flags,
        "evidence_views": evidence,
    }


def colorize_depth(depth, minimum, maximum):
    valid = np.isfinite(depth)
    normalized = np.zeros(depth.shape, dtype=float)
    normalized[valid] = np.clip(
        (depth[valid] - minimum) / max(maximum - minimum, 1e-6),
        0.0,
        1.0,
    )
    stops = np.asarray(
        [
            [10, 14, 22],
            [39, 74, 114],
            [38, 177, 190],
            [105, 218, 157],
            [245, 205, 91],
        ],
        dtype=float,
    )
    position = normalized * (len(stops) - 1)
    lower = np.floor(position).astype(int)
    upper = np.minimum(lower + 1, len(stops) - 1)
    fraction = position - lower
    rgb = stops[lower] * (1.0 - fraction[..., None]) + stops[upper] * fraction[..., None]
    rgb[~valid] = 0
    return rgb.astype(np.uint8)


def save_depth_artifacts(output_dir, name, values, fixed_range=None):
    depth_dir = output_dir / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)
    finite_values = np.concatenate(
        [value[np.isfinite(value)] for value in values.values()]
    )
    if fixed_range is None:
        minimum, maximum = np.quantile(finite_values, [0.02, 0.98])
    else:
        minimum, maximum = fixed_range

    tiles = []
    for viewpoint, value in values.items():
        image = Image.fromarray(colorize_depth(value, minimum, maximum))
        image.save(depth_dir / f"{name}_{viewpoint}.png")
        tiles.append((viewpoint, image))
    save_contact_sheet(
        depth_dir / f"{name}_contact_sheet.png",
        tiles,
        f"{minimum:.2f}–{maximum:.2f} m",
    )


def save_support_artifacts(output_dir, support_images):
    colors = np.asarray(
        [[7, 9, 13], [161, 105, 52], [73, 185, 145], [85, 199, 255]],
        dtype=np.uint8,
    )
    depth_dir = output_dir / "depth"
    tiles = []
    for viewpoint, support in support_images.items():
        image = Image.fromarray(colors[np.minimum(support, 3)])
        image.save(depth_dir / f"support_{viewpoint}.png")
        tiles.append((viewpoint, image))
    save_contact_sheet(
        depth_dir / "support_contact_sheet.png",
        tiles,
        "orange=1 · green=2 · blue=3 views",
    )


def save_contact_sheet(path, tiles, subtitle):
    tile_width = 260
    label_height = 46
    target_height = 360
    sheet = Image.new(
        "RGB",
        (tile_width * len(tiles), target_height + label_height),
        "#0b0e12",
    )
    draw = ImageDraw.Draw(sheet)
    for index, (viewpoint, image) in enumerate(tiles):
        copy = image.copy()
        copy.thumbnail((tile_width, target_height), Image.Resampling.LANCZOS)
        x = index * tile_width + (tile_width - copy.width) // 2
        y = label_height + (target_height - copy.height) // 2
        sheet.paste(copy, (x, y))
        label = viewpoint.removeprefix("survey_").replace("_", " ").title()
        draw.text((index * tile_width + 10, 8), label, fill="#f1f5f9")
        draw.text((index * tile_width + 10, 25), subtitle, fill="#8591a1")
    sheet.save(path)


def save_ply(path, points):
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\nend_header\n"
    )
    rows = "\n".join(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in points)
    path.write_text(header + rows + ("\n" if rows else ""))


def save_height_map(output_dir, points, voxel_size):
    shape = np.ceil((WORKSPACE_MAX[:2] - WORKSPACE_MIN[:2]) / voxel_size).astype(int) + 1
    heights = np.zeros(tuple(shape), dtype=float)
    indices = np.floor((points[:, :2] - WORKSPACE_MIN[:2]) / voxel_size).astype(int)
    np.maximum.at(heights, (indices[:, 0], indices[:, 1]), points[:, 2])
    normalized = np.clip(heights / WORKSPACE_MAX[2], 0.0, 1.0)
    image = Image.fromarray((normalized * 255).astype(np.uint8))
    image = image.transpose(Image.Transpose.ROTATE_180).resize(
        (680, 440),
        Image.Resampling.NEAREST,
    )
    image.save(output_dir / "height_map.png")


def save_topdown(output_dir, objects, points):
    width, height = 760, 520
    margin = 55
    canvas = Image.new("RGB", (width, height), "#0b0e12")
    draw = ImageDraw.Draw(canvas)

    def pixel(xy):
        forward = (xy[0] - WORKSPACE_MIN[0]) / (WORKSPACE_MAX[0] - WORKSPACE_MIN[0])
        left = (WORKSPACE_MAX[1] - xy[1]) / (WORKSPACE_MAX[1] - WORKSPACE_MIN[1])
        return (
            int(margin + left * (width - 2 * margin)),
            int(height - margin - forward * (height - 2 * margin)),
        )

    draw.rectangle(
        [pixel([WORKSPACE_MAX[0], WORKSPACE_MAX[1]]), pixel([WORKSPACE_MIN[0], WORKSPACE_MIN[1]])],
        outline="#445163",
        width=2,
    )
    draw.text((margin, 17), "FUSED METRIC DEPTH + PLANNER FOOTPRINTS", fill="#9ca8b7")
    draw.text(
        (margin, 36),
        "GRAY = CROSS-VIEW DEPTH POINTS   COLOR = FITTED OBJECT FOOTPRINT",
        fill="#6f7d8c",
    )
    base_x, base_y = pixel([WORKSPACE_MIN[0], 0.0])
    draw.rectangle(
        [base_x - 30, base_y + 7, base_x + 30, base_y + 29],
        outline="#9ca8b7",
        width=2,
    )
    draw.text((base_x - 18, base_y + 12), "ROBOT", fill="#9ca8b7")

    for point in points:
        draw.point(pixel(point[:2]), fill="#293746")
    for index, obj in enumerate(objects):
        color = CLUSTER_COLORS[index % len(CLUSTER_COLORS)]
        polygon = [pixel(corner) for corner in obj["footprint_corners"]]
        draw.polygon(polygon, outline=color, width=3)
        center = pixel(obj["position"][:2])
        draw.ellipse(
            [center[0] - 4, center[1] - 4, center[0] + 4, center[1] + 4],
            fill=color,
        )
        draw.text((center[0] + 8, center[1] - 12), obj["id"], fill=color)
    canvas.save(output_dir / "topdown.png")


def reconstruct(
    scan_path,
    background_path,
    output_dir,
    voxel_size=DEFAULT_VOXEL_SIZE,
    depth_model=DEFAULT_DEPTH_MODEL,
    device="auto",
    consistency_tolerance=DEFAULT_CONSISTENCY_TOLERANCE,
    minimum_views=DEFAULT_MINIMUM_VIEWS,
    pixel_stride=1,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for directory in ("depth", "masks", "crops"):
        shutil.rmtree(output_dir / directory, ignore_errors=True)
    for filename in (
        "geometry.json",
        "semantic_scene.json",
        "fused_cloud.ply",
        "height_map.png",
        "topdown.png",
    ):
        (output_dir / filename).unlink(missing_ok=True)
    try:
        initial_scan_id = json.loads(Path(scan_path).read_text()).get("scan_id")
        initial_background_id = json.loads(
            Path(background_path).read_text()
        ).get("scan_id")
    except (OSError, json.JSONDecodeError):
        initial_scan_id = None
        initial_background_id = None
    diagnostics = {
        "version": 1,
        "method": RECONSTRUCTION_METHOD,
        "depth_model": depth_model,
        "device": device,
        "stages": {
            "foreground_masks": {"status": "running"},
            "depth_prediction": {"status": "waiting"},
            "table_alignment": {"status": "waiting"},
            "cross_view_consistency": {"status": "waiting"},
            "voxel_fusion": {"status": "waiting"},
            "object_fitting": {"status": "waiting"},
        },
        "views": {},
    }
    if initial_scan_id:
        diagnostics["scene_id"] = initial_scan_id
    if initial_background_id:
        diagnostics["background_scan_id"] = initial_background_id
    write_diagnostics(output_dir, diagnostics)
    try:
        (
            scan,
            background_scan,
            frames,
            rgb_images,
            masks,
            occluders,
            registration_records,
        ) = foreground_masks(scan_path, background_path, output_dir)
        configure_workspace_bounds(scan)
    except Exception as error:
        diagnostics["stages"]["foreground_masks"] = {
            "status": "error",
            "reason": str(error),
        }
        write_diagnostics(output_dir, diagnostics)
        raise
    foreground_fractions = [
        record["foreground_pixel_fraction"]
        for record in registration_records.values()
    ]
    diagnostics["stages"]["foreground_masks"] = {
        "status": (
            "complete"
            if all(0.0002 <= value <= 0.25 for value in foreground_fractions)
            else "review"
        ),
        "views": len(frames),
        "table_plane_registration": registration_records,
    }
    diagnostics["scene_id"] = scan["scan_id"]
    diagnostics["background_scan_id"] = background_scan["scan_id"]
    diagnostics["stages"]["depth_prediction"]["status"] = "running"
    write_diagnostics(output_dir, diagnostics)

    try:
        predictions, resolved_device = predict_depths(
            rgb_images,
            depth_model,
            device,
        )
    except Exception as error:
        diagnostics["stages"]["depth_prediction"] = {
            "status": "error",
            "reason": str(error),
        }
        write_diagnostics(output_dir, diagnostics)
        raise
    diagnostics["device"] = resolved_device
    save_depth_artifacts(output_dir, "raw_depth", predictions)
    diagnostics["stages"]["depth_prediction"] = {
        "status": "complete",
        "views": len(predictions),
        "source": "learned monocular metric depth",
    }
    diagnostics["stages"]["table_alignment"]["status"] = "running"
    write_diagnostics(output_dir, diagnostics)

    try:
        aligned_depths, alignment_records = align_depths(
            predictions,
            frames,
            masks,
            occluders,
        )
    except Exception as error:
        diagnostics["stages"]["table_alignment"] = {
            "status": "error",
            "reason": str(error),
        }
        write_diagnostics(output_dir, diagnostics)
        raise
    diagnostics["views"] = alignment_records
    save_depth_artifacts(output_dir, "aligned_depth", aligned_depths, (0.15, 0.75))
    diagnostics["stages"]["table_alignment"] = {
        "status": (
            "complete"
            if all(record["status"] == "pass" for record in alignment_records.values())
            else "review"
        ),
        "plane": "robot-base z=0 table",
    }
    diagnostics["stages"]["cross_view_consistency"]["status"] = "running"
    write_diagnostics(output_dir, diagnostics)

    try:
        (
            consistent_points,
            source_views,
            support_counts,
            support_images,
            candidate_count,
            viewpoint_names,
        ) = cross_view_filter(
            aligned_depths,
            frames,
            masks,
            occluders,
            consistency_tolerance,
            minimum_views,
            pixel_stride,
        )
    except Exception as error:
        diagnostics["stages"]["cross_view_consistency"] = {
            "status": "error",
            "reason": str(error),
        }
        write_diagnostics(output_dir, diagnostics)
        raise
    save_support_artifacts(output_dir, support_images)
    accepted_fraction = len(consistent_points) / max(candidate_count, 1)
    diagnostics["stages"]["cross_view_consistency"] = {
        "status": (
            "complete"
            if len(consistent_points) >= 100 and accepted_fraction >= 0.05
            else "review"
        ),
        "candidate_points": int(candidate_count),
        "accepted_points": int(len(consistent_points)),
        "accepted_fraction": round(accepted_fraction, 4),
        "minimum_views": minimum_views,
        "tolerance_meters": consistency_tolerance,
    }
    diagnostics["stages"]["voxel_fusion"]["status"] = "running"
    write_diagnostics(output_dir, diagnostics)

    fused_points, view_masks, sample_counts, mean_support = fuse_points(
        consistent_points,
        source_views,
        support_counts,
        voxel_size,
    )
    save_ply(output_dir / "fused_cloud.ply", fused_points)
    diagnostics["stages"]["voxel_fusion"] = {
        "status": "complete",
        "input_points": int(len(consistent_points)),
        "fused_voxels": int(len(fused_points)),
        "voxel_size_meters": voxel_size,
    }
    diagnostics["stages"]["object_fitting"]["status"] = "running"
    write_diagnostics(output_dir, diagnostics)

    components = cluster_fused_points(fused_points, voxel_size)
    # Refine each object's footprint from the foreground mask cast onto the
    # table plane; assign every mask-plane point to its nearest cluster center.
    mask_plane_points = np.concatenate(
        [
            project_mask_to_plane(frames[name], masks[name] & ~occluders[name])
            for name in viewpoint_names
        ]
    )
    cluster_centers = (
        np.array([fused_points[c][:, :2].mean(axis=0) for c in components])
        if components
        else np.empty((0, 2))
    )
    footprints = [np.empty((0, 2)) for _ in components]
    if len(mask_plane_points) and len(cluster_centers):
        assignment = np.argmin(
            np.linalg.norm(
                mask_plane_points[:, None, :] - cluster_centers[None, :, :],
                axis=2,
            ),
            axis=1,
        )
        footprints = [
            mask_plane_points[assignment == index]
            for index in range(len(components))
        ]
    described = [
        describe_component(
            component,
            fused_points,
            view_masks,
            sample_counts,
            mean_support,
            viewpoint_names,
            alignment_records,
            voxel_size,
            footprint_xy=footprints[index],
        )
        for index, component in enumerate(components)
    ]
    described = [
        obj
        for obj in described
        if obj["dimensions"][0] <= 0.18
        and obj["dimensions"][1] <= 0.12
        and obj["dimensions"][2] >= 0.006
    ]
    described.sort(key=lambda obj: obj["position"][1], reverse=True)
    for index, obj in enumerate(described, start=1):
        obj["id"] = f"object_{index}"
    if not described:
        diagnostics["stages"]["object_fitting"] = {
            "status": "error",
            "reason": "no supported tabletop object clusters",
        }
        write_diagnostics(output_dir, diagnostics)
        raise RuntimeError(
            "depth fusion produced no supported tabletop objects; inspect the "
            "intermediate depth and support images in scene memory"
        )

    save_height_map(output_dir, fused_points, voxel_size)
    save_topdown(output_dir, described, fused_points)
    diagnostics["stages"]["object_fitting"] = {
        "status": "complete",
        "objects": len(described),
    }
    write_diagnostics(output_dir, diagnostics)

    geometry = {
        "scene_id": scan["scan_id"],
        "background_scan_id": background_scan["scan_id"],
        "method": RECONSTRUCTION_METHOD,
        "perception_contract": GEOMETRY_CONTRACT,
        "scan_contract": scan["perception_contract"],
        "sensor_modalities": GEOMETRY_MODALITIES,
        "uses_privileged_simulator_data": False,
        "depth_model": depth_model,
        "depth_source": "predicted_from_rgb",
        "depth_alignment": "affine fit to calibrated table plane",
        "fusion_method": "cross-view consistency plus metric voxel averaging",
        "voxel_size_meters": voxel_size,
        "workspace_bounds": {
            "minimum": WORKSPACE_MIN.tolist(),
            "maximum": WORKSPACE_MAX.tolist(),
        },
        "table_plane": [0.0, 0.0, 1.0, 0.0],
        "minimum_consistent_views": minimum_views,
        "consistency_tolerance_meters": consistency_tolerance,
        "raw_depth_points": int(candidate_count),
        "consistent_depth_points": int(len(consistent_points)),
        "fused_voxels": int(len(fused_points)),
        "points": np.round(fused_points, 4).tolist(),
        "fused_cloud": "fused_cloud.ply",
        "diagnostics": "depth/diagnostics.json",
        "height_map": "height_map.png",
        "height_map_kind": "fused_metric_surface_height",
        "top_view": "topdown.png",
        "top_view_kind": "fused_depth_points_with_fitted_footprints",
        "objects": described,
    }
    (output_dir / "geometry.json").write_text(json.dumps(geometry, indent=2) + "\n")

    print(
        f"kept {len(consistent_points)} of {candidate_count} depth points after "
        f"{minimum_views}-view consistency filtering"
    )
    print(f"fused {len(fused_points)} metric voxels")
    print(f"found {len(described)} object clusters")
    print(f"depth model: {depth_model} on {resolved_device}")
    for obj in described:
        xyz = ", ".join(f"{value:.3f}" for value in obj["position"])
        size = ", ".join(f"{value:.3f}" for value in obj["dimensions"])
        print(f"  {obj['id']}: xyz=({xyz}) size=({size}) views={obj['evidence_views']}")
    print(f"saved scene geometry -> {output_dir}")
    return geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, default=DEFAULT_SCAN)
    parser.add_argument("--background", type=Path, default=DEFAULT_BACKGROUND)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--voxel-size", type=float, default=DEFAULT_VOXEL_SIZE)
    parser.add_argument("--depth-model", default=DEFAULT_DEPTH_MODEL)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--consistency-tolerance",
        type=float,
        default=DEFAULT_CONSISTENCY_TOLERANCE,
    )
    parser.add_argument(
        "--minimum-views",
        type=int,
        choices=(1, 2, 3),
        default=DEFAULT_MINIMUM_VIEWS,
    )
    parser.add_argument("--pixel-stride", type=int, default=1)
    args = parser.parse_args()
    if args.voxel_size <= 0:
        parser.error("--voxel-size must be positive")
    if args.consistency_tolerance <= 0:
        parser.error("--consistency-tolerance must be positive")
    if args.pixel_stride < 1:
        parser.error("--pixel-stride must be at least 1")
    reconstruct(
        args.scan,
        args.background,
        args.output,
        voxel_size=args.voxel_size,
        depth_model=args.depth_model,
        device=args.device,
        consistency_tolerance=args.consistency_tolerance,
        minimum_views=args.minimum_views,
        pixel_stride=args.pixel_stride,
    )


if __name__ == "__main__":
    main()
