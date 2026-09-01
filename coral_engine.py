"""
CoralSCOP Inference & Visualization Engine
Supports Hugging Face model auto-download, CUDA/MPS/CPU acceleration,
and vibrant multi-color coral segmentation overlay rendering.
"""

import os
import io
import json
import colorsys
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import cv2
from PIL import Image
import torch
import pycocotools.mask as mask_util
from huggingface_hub import hf_hub_download

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

HF_REPO_ID = "reefsupport/CoralSCOP"
HF_FILENAME = "models/vit_b_coralscop.pth"


def get_optimal_device(preference: str = "auto") -> torch.device:
    """
    Determines the compute device following the priority:
    1. CUDA GPU (if available)
    2. Apple Silicon MPS (if available)
    3. CPU (fallback)
    """
    if preference.lower() == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif preference.lower() == "mps":
        return torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    elif preference.lower() == "cpu":
        return torch.device("cpu")

    # Auto selection
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device_info() -> Dict[str, Any]:
    """Returns human-readable details about available accelerators."""
    device = get_optimal_device("auto")
    info = {
        "device": str(device),
        "device_type": device.type,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
    elif info["mps_available"]:
        info["gpu_name"] = "Apple Silicon (MPS Metal Acceleration)"
    else:
        info["gpu_name"] = "CPU (General Processor)"
    return info


def download_coralscop_checkpoint(cache_dir: Optional[str] = None) -> str:
    """
    Downloads CoralSCOP ViT-B checkpoint from Hugging Face if not already present.
    Returns the local path to the checkpoint.
    """
    local_default = os.path.join(os.path.dirname(__file__), "models", "vit_b_coralscop.pth")
    if os.path.exists(local_default):
        return local_default

    print(f"Fetching {HF_FILENAME} from Hugging Face repo {HF_REPO_ID}...")
    checkpoint_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME,
        cache_dir=cache_dir,
    )
    return checkpoint_path


def load_coralscop_model(checkpoint_path: Optional[str] = None, device: Optional[torch.device] = None):
    """
    Loads and returns the CoralSCOP SAM ViT-B model.
    """
    if device is None:
        device = get_optimal_device()

    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        checkpoint_path = download_coralscop_checkpoint()

    sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    sam.to(device=device)
    sam.eval()
    return sam


def generate_distinct_colors(n: int) -> List[Tuple[int, int, int]]:
    """
    Generates n visually distinct, vibrant RGB colors using golden-ratio hue distribution.
    """
    colors = []
    golden_ratio_conjugate = 0.618033988749895
    hue = 0.12  # start with vibrant aqua/coral hue

    for _ in range(n):
        hue = (hue + golden_ratio_conjugate) % 1.0
        saturation = 0.85
        value = 0.95
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


def run_segmentation(
    model,
    image: Union[np.ndarray, Image.Image],
    points_per_side: int = 32,
    pred_iou_thresh: float = 0.86,
    stability_score_thresh: float = 0.92,
    min_mask_region_area: int = 100,
    crop_n_layers: int = 0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Executes CoralSCOP segmentation on an input image.

    Returns:
        masks_info: List of processed segment records.
        summary_stats: High-level metrics (total corals, % coverage, mean IoU, etc.).
    """
    if isinstance(image, Image.Image):
        image_np = np.array(image.convert("RGB"))
    else:
        image_np = image.copy()
        if len(image_np.shape) == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

    height, width = image_np.shape[:2]
    total_pixels = height * width

    mask_generator = SamAutomaticMaskGenerator(
        model=model,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        crop_n_layers=crop_n_layers,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=min_mask_region_area,
        output_mode="binary_mask",
    )

    raw_masks = mask_generator.generate(image_np)

    # Sort masks by area descending
    raw_masks = sorted(raw_masks, key=lambda x: x["area"], reverse=True)

    colors = generate_distinct_colors(max(len(raw_masks), 1))
    processed_masks = []
    union_mask = np.zeros((height, width), dtype=bool)

    for idx, ann in enumerate(raw_masks):
        mask = ann["segmentation"]
        union_mask = np.logical_or(union_mask, mask)

        # Compute centroid for label placement
        y_indices, x_indices = np.where(mask)
        if len(y_indices) > 0:
            cx = int(np.mean(x_indices))
            cy = int(np.mean(y_indices))
        else:
            bbox = ann["bbox"]
            cx = int(bbox[0] + bbox[2] / 2)
            cy = int(bbox[1] + bbox[3] / 2)

        area_px = int(ann["area"])
        area_pct = (area_px / total_pixels) * 100.0
        color_rgb = colors[idx]
        color_hex = f"#{color_rgb[0]:02x}{color_rgb[1]:02x}{color_rgb[2]:02x}"

        processed_masks.append({
            "id": idx + 1,
            "mask": mask,
            "bbox": [round(float(v), 1) for v in ann["bbox"]],  # [x, y, w, h]
            "area_px": area_px,
            "area_pct": round(area_pct, 2),
            "predicted_iou": round(float(ann.get("predicted_iou", 0.0)), 4),
            "stability_score": round(float(ann.get("stability_score", 0.0)), 4),
            "category": "Coral",
            "centroid": [cx, cy],
            "color_rgb": color_rgb,
            "color_hex": color_hex,
        })

    coral_covered_pixels = int(np.sum(union_mask))
    coral_coverage_pct = round((coral_covered_pixels / total_pixels) * 100.0, 2)
    mean_iou = round(float(np.mean([m["predicted_iou"] for m in processed_masks])), 4) if processed_masks else 0.0
    mean_stability = round(float(np.mean([m["stability_score"] for m in processed_masks])), 4) if processed_masks else 0.0

    summary_stats = {
        "total_corals_detected": len(processed_masks),
        "coral_coverage_pct": coral_coverage_pct,
        "coral_covered_pixels": coral_covered_pixels,
        "total_image_pixels": total_pixels,
        "image_resolution": f"{width}x{height}",
        "mean_predicted_iou": mean_iou,
        "mean_stability_score": mean_stability,
    }

    return processed_masks, summary_stats


def create_segmentation_overlay(
    image: Union[np.ndarray, Image.Image],
    masks_info: List[Dict[str, Any]],
    alpha: float = 0.45,
    draw_contours: bool = True,
    draw_labels: bool = True,
    draw_boxes: bool = False,
    selected_mask_id: Optional[int] = None,
) -> np.ndarray:
    """
    Renders a vibrant segmentation overlay on the original image with custom styling.
    """
    if isinstance(image, Image.Image):
        image_np = np.array(image.convert("RGB"))
    else:
        image_np = image.copy()
        if len(image_np.shape) == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

    overlay = image_np.copy()

    for item in masks_info:
        mask_id = item["id"]
        mask = item["mask"]
        color = item["color_rgb"]

        # If a single mask is selected, dim other masks
        current_alpha = alpha
        if selected_mask_id is not None:
            if mask_id == selected_mask_id:
                current_alpha = min(alpha + 0.25, 0.85)
            else:
                current_alpha = alpha * 0.2

        # Apply colored overlay
        colored_mask = np.zeros_like(image_np, dtype=np.uint8)
        colored_mask[mask] = color
        overlay[mask] = cv2.addWeighted(image_np[mask], 1.0 - current_alpha, colored_mask[mask], current_alpha, 0)

        # Draw contour borders
        if draw_contours:
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            border_color = (255, 255, 255) if selected_mask_id == mask_id else color
            thickness = 2 if selected_mask_id == mask_id else 1
            cv2.drawContours(overlay, contours, -1, border_color, thickness, cv2.LINE_AA)

        # Draw bounding boxes
        if draw_boxes:
            x, y, w, h = [int(v) for v in item["bbox"]]
            box_color = (255, 255, 255) if selected_mask_id == mask_id else color
            cv2.rectangle(overlay, (x, y), (x + w, y + h), box_color, 1)

        # Draw centroid label
        if draw_labels and (selected_mask_id is None or selected_mask_id == mask_id):
            cx, cy = item["centroid"]
            label_text = f"#{mask_id}"
            font_scale = 0.45
            thickness = 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

            px0 = max(cx - text_w // 2 - 3, 0)
            py0 = max(cy - text_h // 2 - 3, 0)
            px1 = min(px0 + text_w + 6, image_np.shape[1] - 1)
            py1 = min(py0 + text_h + 6, image_np.shape[0] - 1)
            cv2.rectangle(overlay, (px0, py0), (px1, py1), (20, 20, 20), -1)
            cv2.putText(overlay, label_text, (px0 + 3, py0 + text_h + 1), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return overlay


def build_coco_json(image_name: str, width: int, height: int, masks_info: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Formats segmentation output into standard COCO JSON structure."""
    coco_output = {
        "info": {
            "description": "CoralSCOP Coral Segmentation Dataset",
            "model": "reefsupport/CoralSCOP",
            "version": "1.0",
        },
        "images": [
            {
                "id": 1,
                "file_name": image_name,
                "width": width,
                "height": height,
            }
        ],
        "categories": [
            {"id": 1, "name": "Coral", "supercategory": "Marine Life"}
        ],
        "annotations": [],
    }

    for item in masks_info:
        seg_mask = np.asfortranarray(item["mask"])
        compressed_rle = mask_util.encode(seg_mask)
        compressed_rle["counts"] = compressed_rle["counts"].decode("utf-8")

        coco_output["annotations"].append({
            "id": item["id"],
            "image_id": 1,
            "category_id": 1,
            "bbox": item["bbox"],
            "area": item["area_px"],
            "area_percentage": item["area_pct"],
            "predicted_iou": item["predicted_iou"],
            "stability_score": item["stability_score"],
            "centroid": item["centroid"],
            "segmentation": compressed_rle,
        })

    return coco_output
