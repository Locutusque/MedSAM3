#!/usr/bin/env python3
"""Segment and count cell instances on a TPU with MedSAM3 + AutoXLA.

This is the TPU companion to ``count_cell_instances.py``. It uses the
repository's existing ``inference_lora.SAM3LoRAInference`` TPU path, then feeds
its predictions into the same small-object removal, mask de-duplication,
instance coloring, measurements, and counting utilities.

Example:
    python count_cell_instances_tpu.py \
        --config configs/full_lora_config.yaml \
        --weights outputs/sam3_lora_full/best_lora_weights.pt \
        --image examples/cells.png \
        --prompt "cancer cell" \
        --min-area 80 \
        --output-dir outputs/cell_count_tpu
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.ops import nms

from inference_lora import SAM3LoRAInference
from count_cell_instances import (
    build_label_map,
    collect_candidates,
    colorize_labels,
    make_overlay,
    save_records,
    suppress_duplicate_masks,
)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for NumPy arrays."""
    values = np.asarray(values, dtype=np.float32)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output


def _postprocess_prediction(
    prediction: dict,
    prompt: str,
    threshold: float,
    nms_iou: float,
) -> dict:
    """Convert raw TPU inference output to count_cell_instances result format."""
    image = prediction["image"].convert("RGB")
    orig_w, orig_h = image.size

    raw_scores = np.asarray(prediction["scores"])
    if raw_scores.ndim == 1:
        scores = raw_scores
    else:
        scores = raw_scores.max(axis=-1)

    keep = scores > threshold
    if not np.any(keep):
        return {
            "prompt": prompt,
            "boxes": None,
            "scores": None,
            "masks": None,
            "num_detections": 0,
        }

    boxes_cxcywh = np.asarray(prediction["boxes"], dtype=np.float32)[keep]
    kept_scores = scores[keep].astype(np.float32)
    cx, cy, width, height = boxes_cxcywh.T
    boxes_xyxy = np.stack(
        [
            (cx - width / 2.0) * orig_w,
            (cy - height / 2.0) * orig_h,
            (cx + width / 2.0) * orig_w,
            (cy + height / 2.0) * orig_h,
        ],
        axis=-1,
    )

    keep_nms = nms(
        torch.from_numpy(boxes_xyxy),
        torch.from_numpy(kept_scores),
        nms_iou,
    ).cpu().numpy()
    boxes_xyxy = boxes_xyxy[keep_nms]
    kept_scores = kept_scores[keep_nms]

    raw_masks = prediction.get("masks")
    masks_np = None
    if raw_masks is not None:
        masks = np.asarray(raw_masks)[keep][keep_nms]
        # inference_lora returns mask logits; convert to probabilities first.
        masks = _sigmoid(masks) > 0.5
        masks_tensor = torch.from_numpy(masks.astype(np.float32)).unsqueeze(0)
        masks_np = (
            F.interpolate(
                masks_tensor,
                size=(orig_h, orig_w),
                mode="nearest",
            )
            .squeeze(0)
            .bool()
            .numpy()
        )

    return {
        "prompt": prompt,
        "boxes": boxes_xyxy,
        "scores": kept_scores,
        "masks": masks_np,
        "num_detections": int(len(keep_nms)),
    }


def run_tpu_prompts(
    prompter: SAM3LoRAInference,
    image_path: str,
    prompts: List[str],
    threshold: float,
    nms_iou: float,
) -> Dict[object, object]:
    """Run every text concept through the TPU prompter."""
    results: Dict[object, object] = {}
    original_image = None

    for index, prompt in enumerate(prompts):
        prediction = prompter.predict(image_path, text_prompt=prompt)
        original_image = prediction["image"].convert("RGB")
        results[index] = _postprocess_prediction(
            prediction,
            prompt=prompt,
            threshold=threshold,
            nms_iou=nms_iou,
        )
        print(
            f"Prompt '{prompt}': "
            f"{results[index]['num_detections']} detections after NMS"
        )

    if original_image is None:
        raise RuntimeError("No prompts were supplied.")
    results["_image"] = original_image
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prompt MedSAM3 on TPU to segment, colorize, and count cells."
    )
    parser.add_argument("--config", required=True, help="MedSAM3 LoRA YAML config")
    parser.add_argument("--weights", required=True, help="LoRA checkpoint path")
    parser.add_argument("--image", required=True, help="Path to one input image")
    parser.add_argument(
        "--prompt",
        nargs="+",
        default=["cancer cell"],
        help='Text concept(s), for example --prompt "cancer cell"',
    )
    parser.add_argument("--output-dir", default="outputs/cell_count_tpu")
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--mask-nms-iou", type=float, default=0.65)
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument(
        "--min-area",
        type=int,
        default=64,
        help="Remove and do not count objects smaller than this many pixels",
    )
    parser.add_argument("--max-hole-area", type=int, default=32)
    parser.add_argument("--overlay-alpha", type=float, default=0.60)
    parser.add_argument("--color-seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_area < 1:
        raise ValueError("--min-area must be at least 1")
    if not 0.0 <= args.overlay_alpha <= 1.0:
        raise ValueError("--overlay-alpha must be between 0 and 1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # inference_lora.py already routes TPU setup through tpu_utils + AutoXLA.
    prompter = SAM3LoRAInference(
        config_path=args.config,
        weights_path=args.weights,
        use_tpu=True,
    )
    prompter.resolution = args.resolution

    results = run_tpu_prompts(
        prompter,
        image_path=args.image,
        prompts=args.prompt,
        threshold=args.threshold,
        nms_iou=args.nms_iou,
    )

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
    Image.fromarray(label_map.astype(np.uint16), mode="I;16").save(
        output_dir / "instance_labels.png"
    )
    np.save(output_dir / "instance_labels.npy", label_map)
    save_records(records, output_dir)

    print("\n" + "=" * 64)
    print("Backend: TPU via torch_xla + AutoXLA")
    print(f"Cell instances counted: {len(records)}")
    print(f"Small-object cutoff: {args.min_area} pixels")
    print(f"Colored instances: {output_dir / 'instances_colored.png'}")
    print(f"Overlay:           {output_dir / 'instances_overlay.png'}")
    print(f"Measurements:      {output_dir / 'instances.csv'}")
    print("=" * 64)


if __name__ == "__main__":
    main()
