"""Attach semantic labels to reconstructed objects without changing geometry.

Manual usage:
    python 05_semantic_scene/label_scene.py \
      --label "object_1=small cardboard box" \
      --label "object_2=eraser" \
      --label "object_3=whiteboard marker" \
      --alias "object_3=marker,pen,writing tool"

Hosted vision-model usage:
    export OPENROUTER_API_KEY="..."
    python 05_semantic_scene/label_scene.py
"""

import argparse
import base64
import json
import os
from pathlib import Path


DEFAULT_GEOMETRY = Path(__file__).resolve().parent / "output" / "latest_scene" / "geometry.json"
DEFAULT_MODEL = "openrouter/free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def parse_manual_labels(values, alias_values, object_ids):
    aliases = {object_id: [] for object_id in object_ids}
    for value in alias_values:
        if "=" not in value:
            raise ValueError(f"alias must use object_id=name,name, received {value!r}")
        object_id, names = (part.strip() for part in value.split("=", 1))
        if object_id not in object_ids:
            raise ValueError(f"unknown object ID {object_id!r}")
        aliases[object_id] = [name.strip() for name in names.split(",") if name.strip()]

    labels = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"label must use object_id=name, received {value!r}")
        object_id, label = (part.strip() for part in value.split("=", 1))
        if object_id not in object_ids:
            raise ValueError(f"unknown object ID {object_id!r}")
        if not label:
            raise ValueError(f"label for {object_id} is empty")
        labels[object_id] = {
            "id": object_id,
            "label": label,
            "aliases": aliases[object_id],
            "attributes": [],
            "semantic_confidence": 1.0,
        }

    missing = [object_id for object_id in object_ids if object_id not in labels]
    if missing:
        raise ValueError(f"missing manual labels for: {', '.join(missing)}")
    return [labels[object_id] for object_id in object_ids]


def label_with_openrouter(contact_sheet, object_ids, model):
    try:
        from openai import OpenAI
    except ModuleNotFoundError as error:
        raise SystemExit(
            f"missing {error.name!r}; run: python -m pip install -r requirements.txt"
        ) from error

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENROUTER_API_KEY is not set. Add a free OpenRouter key, or use "
            "manual --label values."
        )

    encoded = base64.b64encode(contact_sheet.read_bytes()).decode("ascii")
    prompt = (
        "Identify each labeled crop as an ordinary desk object. Preserve every "
        f"ID exactly: {object_ids}. Return a concise label, useful aliases, visible "
        "attributes, and calibrated confidence. If uncertain, use label 'unknown'. "
        "Do not provide positions, dimensions, grasp coordinates, or robot commands."
    )
    label_schema = {
        "type": "object",
        "properties": {
            "objects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "label": {"type": "string"},
                        "aliases": {"type": "array", "items": {"type": "string"}},
                        "attributes": {"type": "array", "items": {"type": "string"}},
                        "semantic_confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                    },
                    "required": [
                        "id",
                        "label",
                        "aliases",
                        "attributes",
                        "semantic_confidence",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["objects"],
        "additionalProperties": False,
    }
    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"},
                },
            ],
        }],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "scene_labels",
                "strict": True,
                "schema": label_schema,
            },
        },
        max_tokens=1200,
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("OpenRouter returned no semantic labels")
    try:
        labels = json.loads(content)["objects"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise RuntimeError("OpenRouter returned invalid semantic-label JSON") from error
    if not isinstance(labels, list):
        raise RuntimeError("OpenRouter returned 'objects' in an invalid format")
    for item in labels:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise RuntimeError("OpenRouter returned an object with an invalid ID")
        if not isinstance(item.get("label"), str) or not item["label"].strip():
            raise RuntimeError(f"model returned an invalid label for {item['id']}")
        for field in ("aliases", "attributes"):
            values = item.get(field)
            if not isinstance(values, list) or not all(
                isinstance(value, str) for value in values
            ):
                raise RuntimeError(f"invalid {field} for {item['id']}")
        confidence = item.get("semantic_confidence")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0.0 <= confidence <= 1.0
        ):
            raise RuntimeError(f"invalid confidence for {item['id']}: {confidence!r}")

    returned_ids = [item["id"] for item in labels]
    if sorted(returned_ids) != sorted(object_ids):
        raise RuntimeError(f"model returned IDs {returned_ids}; expected {object_ids}")
    labels_by_id = {item["id"]: item for item in labels}
    ordered_labels = [labels_by_id[object_id] for object_id in object_ids]
    return ordered_labels, response.model


def merge_semantics(geometry, labels, provider):
    labels_by_id = {item["id"]: item for item in labels}
    objects = []
    for geometry_object in geometry["objects"]:
        semantic = labels_by_id[geometry_object["id"]]
        objects.append({
            "id": geometry_object["id"],
            "label": semantic["label"],
            "aliases": semantic["aliases"],
            "attributes": semantic["attributes"],
            "semantic_confidence": semantic["semantic_confidence"],
            "position": geometry_object["position"],
            "dimensions": geometry_object["dimensions"],
            "yaw_radians": geometry_object["yaw_radians"],
            "footprint_corners": geometry_object["footprint_corners"],
            "geometry_confidence": geometry_object["geometry_confidence"],
            "multi_view_voxel_fraction": geometry_object[
                "multi_view_voxel_fraction"
            ],
            "boundary_flags": geometry_object.get("boundary_flags", []),
            "evidence_views": geometry_object["evidence_views"],
        })

    return {
        "scene_id": geometry["scene_id"],
        "background_scan_id": geometry["background_scan_id"],
        "semantic_provider": provider,
        "geometry_method": geometry["method"],
        "perception_contract": geometry["perception_contract"],
        "sensor_modalities": geometry["sensor_modalities"],
        "uses_privileged_simulator_data": geometry[
            "uses_privileged_simulator_data"
        ],
        "objects": objects,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--label", action="append", default=[])
    parser.add_argument("--alias", action="append", default=[])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    if args.alias and not args.label:
        parser.error("--alias is only used with manual --label values")

    geometry = json.loads(args.geometry.read_text())
    object_ids = [obj["id"] for obj in geometry["objects"]]
    crops_dir = args.geometry.parent / "crops"
    contact_sheet = crops_dir / "contact_sheet.png"
    crops_path = crops_dir / "crops.json"
    if not contact_sheet.exists() or not crops_path.exists():
        raise SystemExit("missing crops; run make_object_crops.py first")
    crops = json.loads(crops_path.read_text())
    crop_ids = [obj["id"] for obj in crops["objects"]]
    if crops["scene_id"] != geometry["scene_id"] or crop_ids != object_ids:
        raise SystemExit("crops are stale; run make_object_crops.py again")

    if args.label:
        labels = parse_manual_labels(args.label, args.alias, object_ids)
        provider = "human"
    else:
        labels, routed_model = label_with_openrouter(
            contact_sheet, object_ids, args.model
        )
        provider = f"openrouter:{routed_model}"

    semantic_scene = merge_semantics(geometry, labels, provider)
    output_path = args.geometry.parent / "semantic_scene.json"
    output_path.write_text(json.dumps(semantic_scene, indent=2) + "\n")
    print(f"saved semantic scene -> {output_path}")
    for obj in semantic_scene["objects"]:
        print(f"  {obj['id']}: {obj['label']} at {obj['position']}")


if __name__ == "__main__":
    main()
