#!/usr/bin/env python3
"""Segment and count cell instances in one image with MedSAM3.

The script uses MedSAM3's text prompter through ``SAM3LoRAInference`` and
converts the returned masks into a clean, non-overlapping instance label map.
Every retained instance receives a unique color for visualization.

Example:
    python count_cell_instances.py \
        --config configs/full_lora_config.yaml \
        --weights outputs/sam3_lora_full/best_lora_weights.pt \
        --image examples/cells.png \
        --prompt "cancer cell" \
        --min-area 80 \
        --output-dir outputs/cell_count
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_holes, remove_small_objects

from infer_sam import SAM3LoRAInference


@dataclass
class Candidate:
    """One candidate instance produced from a connected mask component."""

    mask: np.ndarray
    score: float
    prompt: str


def clean_and_split_mask(
    mask: np.ndarray,
    min_area: int,
    max_hole_area: int,
) -> List[np.ndarray]:
    """Clean a binary mask and split disconnected regions into instances."""
    binary = np.asarray(mask, dtype=bool)
    binary = remove_small_objects(binary, min_size=min_area, connectivity=2)

    if max_hole_area > 0:
        binary = remove_small_holes(binary, area_threshold=max_hole_area, connectivity=2)

    components = label(binary, connectivity=2)
    output: List[np.ndarray] = []
    for region in regionprops(components):
        if region.area < min_area:
            continue
        output.append(components == region.label)
    return output


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Return intersection-over-union for two binary masks."""
    intersection = np.logical_and(a, b).sum(dtype=np.int64)
    if intersection == 0:
        return 0.0
    union = np.logical_or(a, b).sum(dtype=np.int64)
    return float(intersection / max(union, 1))


def suppress_duplicate_masks(
    candidates: Sequence[Candidate],
    iou_threshold: float,
) -> List[Candidate]:
    """Apply score-ordered mask NMS to remove duplicate predictions."""
    kept: List[Candidate] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        duplicate = any(
            mask_iou(candidate.mask, accepted.mask) >= iou_threshold
            for accepted in kept
        )
        if not duplicate:
            kept.append(candidate)
    return kept


def build_label_map(
    candidates: Sequence[Candidate],
    min_area: int,
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    """Create a non-overlapping label map, favoring higher-confidence masks."""
    if not candidates:
        raise ValueError("Cannot build a label map without candidates.")

    height, width = candidates[0].mask.shape
    label_map = np.zeros((height, width), dtype=np.uint32)
    records: List[Dict[str, object]] = []

    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        # Prevent the same pixels from being counted twice when masks overlap.
        available = np.logical_and(candidate.mask, label_map == 0)
        if available.sum() < min_area:
            continue

        # Occlusion can split a mask. Count each surviving connected component.
        components = label(available, connectivity=2)
        for region in regionprops(components):
            if region.area < min_area:
                continue

            instance_id = len(records) + 1
            component = components == region.label
            label_map[component] = instance_id

            min_row, min_col, max_row, max_col = region.bbox
            records.append(
                {
                    "instance_id": instance_id,
                    "prompt": candidate.prompt,
                    "score": float(candidate.score),
                    "area_pixels": int(region.area),
                    "bbox_xyxy": [
                        int(min_col),
                        int(min_row),
                        int(max_col),
                        int(max_row),
                    ],
                    "centroid_xy": [
                        float(region.centroid[1]),
                        float(region.centroid[0]),
                    ],
                }
            )

    return label_map, records


def distinct_palette(count: int, seed: int = 13) -> np.ndarray:
    """Generate deterministic, visually distinct RGB colors."""
    if count <= 0:
        return np.zeros((1, 3), dtype=np.uint8)

    # Evenly spaced hues, shuffled so neighboring IDs are visually dissimilar.
    hues = np.linspace(0.0, 1.0, count, endpoint=False)
    rng = np.random.default_rng(seed)
    rng.shuffle(hues)

    colors = np.zeros((count + 1, 3), dtype=np.uint8)
    for index, hue in enumerate(hues, start=1):
        h6 = hue * 6.0
        sector = int(h6) % 6
        fraction = h6 - int(h6)
        value = 0.95
        saturation = 0.72
        p = value * (1.0 - saturation)
        q = value * (1.0 - saturation * fraction)
        t = value * (1.0 - saturation * (1.0 - fraction))
        rgb_options = (
            (value, t, p),
            (q, value, p),
            (p, value, t),
            (p, q, value),
            (t, p, value),
            (value, p, q),
        )
        colors[index] = np.round(np.array(rgb_options[sector]) * 255).astype(np.uint8)
    return colors


def colorize_labels(label_map: np.ndarray, seed: int) -> np.ndarray:
    """Render every instance with a different color on a black background."""
    count = int(label_map.max())
    palette = distinct_palette(count, seed=seed)
    return palette[label_map]


def make_overlay(
    image: Image.Image,
    colored_instances: np.ndarray,
    label_map: np.ndarray,
    alpha: float,
) -> Image.Image:
    """Overlay colored masks on the original image."""
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    colors = colored_instances.astype(np.float32)
    foreground = label_map > 0
    output = base.copy()
    output[foreground] = (
        (1.0 - alpha) * base[foreground] + alpha * colors[foreground]
    )
    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8))


def collect_candidates(
    results: dict,
    min_area: int,
    max_hole_area: int,
) -> List[Candidate]:
    """Convert MedSAM3 output masks into cleaned connected candidates."""
    candidates: List[Candidate] = []
    result_keys = sorted(key for key in results if key != "_image")

    for key in result_keys:
        result = results[key]
        masks = result.get("masks")
        scores = result.get("scores")
        prompt = str(result.get("prompt", "cell"))
        if masks is None:
            continue

        for index, mask in enumerate(masks):
            score = float(scores[index]) if scores is not None else 0.0
            for component in clean_and_split_mask(mask, min_area, max_hole_area):
                candidates.append(Candidate(mask=component, score=score, prompt=prompt))

    return candidates


def save_records(records: Sequence[Dict[str, object]], output_dir: Path) -> None:
    """Save per-instance measurements as JSON and CSV."""
    with (output_dir / "instances.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"count": len(records), "instances": list(records)},
            handle,
            indent=2,
        )

    fieldnames = [
        "instance_id",
        "prompt",
        "score",
        "area_pixels",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "centroid_x",
        "centroid_y",
    ]
    with (output_dir / "instances.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            bbox = record["bbox_xyxy"]
            centroid = record["centroid_xy"]
            writer.writerow(
                {
                    "instance_id": record["instance_id"],
                    "prompt": record["prompt"],
                    "score": record["score"],
                    "area_pixels": record["area_pixels"],
                    "bbox_x1": bbox[0],
                    "bbox_y1": bbox[1],
                    "bbox_x2": bbox[2],
                    "bbox_y2": bbox[3],
                    "centroid_x": centroid[0],
                    "centroid_y": centroid[1],
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prompt MedSAM3 to segment, colorize, and count cell instances."
    )
    parser.add_argument("--config", required=True, help="MedSAM3 LoRA YAML config")
    parser.add_argument("--weights", default=None, help="LoRA checkpoint path")
    parser.add_argument("--image", required=True, help="Path to one input image")
    parser.add_argument(
        "--prompt",
        nargs="+",
        default=["cancer cell"],
        help='Text concept(s), for example --prompt "cancer cell"',
    )
    parser.add_argument("--output-dir", default="outputs/cell_count")
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument(
        "--mask-nms-iou",
        type=float,
        default=0.65,
        help="IoU at which two cleaned masks are treated as duplicates",
    )
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument(
        "--min-area",
        type=int,
        default=64,
        help="Remove and do not count objects smaller than this many pixels",
    )
    parser.add_argument(
        "--max-hole-area",
        type=int,
        default=32,
        help="Fill holes up to this many pixels; use 0 to disable",
    )
    parser.add_argument("--overlay-alpha", type=float, default=0.60)
    parser.add_argument("--color-seed", type=int, default=13)
    parser.add_argument(
        "--device",
        default="cuda",
        help="Requested device. infer_sam.py falls back to CPU when CUDA is absent.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_area < 1:
        raise ValueError("--min-area must be at least 1")
    if not 0.0 <= args.overlay_alpha <= 1.0:
        raise ValueError("--overlay-alpha must be between 0 and 1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompter = SAM3LoRAInference(
        config_path=args.config,
        weights_path=args.weights,
        resolution=args.resolution,
        detection_threshold=args.threshold,
        nms_iou_threshold=args.nms_iou,
        device=args.device,
    )
    results = prompter.predict(args.image, args.prompt)

    candidates = collect_candidates(
        results,
        min_area=args.min_area,
        max_hole_area=args.max_hole_area,
    )
    candidates = suppress_duplicate_masks(candidates, args.mask_nms_iou)

    image = results["_image"].convert("RGB")
    if candidates:
        label_map, records = build_label_map(candidates, min_area=args.min_area)
    else:
        width, height = image.size
        label_map = np.zeros((height, width), dtype=np.uint32)
        records = []

    colored = colorize_labels(label_map, seed=args.color_seed)
    Image.fromarray(colored).save(output_dir / "instances_colored.png")
    make_overlay(image, colored, label_map, args.overlay_alpha).save(
        output_dir / "instances_overlay.png"
    )

    # PNG safely preserves IDs only up to 65535. Also save an exact NumPy label map.
    Image.fromarray(label_map.astype(np.uint16), mode="I;16").save(
        output_dir / "instance_labels.png"
    )
    np.save(output_dir / "instance_labels.npy", label_map)
    save_records(records, output_dir)

    print("\n" + "=" * 64)
    print(f"Cell instances counted: {len(records)}")
    print(f"Small-object cutoff: {args.min_area} pixels")
    print(f"Colored instances: {output_dir / 'instances_colored.png'}")
    print(f"Overlay:           {output_dir / 'instances_overlay.png'}")
    print(f"Measurements:      {output_dir / 'instances.csv'}")
    print("=" * 64)


if __name__ == "__main__":
    main()
