"""
Download utility for the EPFL-ECEO/coralscapes dataset from Hugging Face.
Supports command-line execution, Jupyter/Colab/Kaggle notebooks, and Streamlit integration.
"""

import os
import argparse
from pathlib import Path
from typing import Optional, Callable
import numpy as np
from PIL import Image
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import json

HF_DATASET_ID = "EPFL-ECEO/coralscapes"


def download_metadata(out_dir: Path):
    """Downloads id2label.json and label2color.json metadata files from HF."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for fn in ["id2label.json", "label2color.json"]:
        dest = out_dir / fn
        if not dest.exists():
            try:
                hf_path = hf_hub_download(repo_id=HF_DATASET_ID, repo_type="dataset", filename=fn)
                with open(hf_path, "r", encoding="utf-8") as f_in, open(dest, "w", encoding="utf-8") as f_out:
                    f_out.write(f_in.read())
            except Exception as e:
                print(f"Warning: Could not download {fn}: {e}")


def download_coralscapes(
    out_dir: str = "coralscapes",
    split: str = "train",
    limit: Optional[int] = None,
    streaming: bool = True,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> int:
    """
    Downloads Coralscapes samples (image + ground-truth mask pairs) to disk.

    Args:
        out_dir: Target root directory (default 'coralscapes').
        split: Dataset split ('train', 'validation', 'test').
        limit: Max number of samples to download (None = all).
        streaming: If True, streams row-by-row without downloading full 4GB archive.
        progress_callback: Optional callback fn(current_index, total_or_limit, status_str).

    Returns:
        Number of samples downloaded and saved.
    """
    out_path = Path(out_dir)
    img_dir = out_path / "images"
    mask_dir = out_path / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    # Save class metadata
    download_metadata(out_path)

    print(f"Loading '{HF_DATASET_ID}' (split='{split}', streaming={streaming})...")
    if progress_callback:
        progress_callback(0, limit or 0, "Connecting to Hugging Face...")

    ds = load_dataset(HF_DATASET_ID, split=split, streaming=streaming)

    # Total count hint
    split_totals = {"train": 1517, "validation": 166, "test": 392}
    total_expected = limit if limit is not None else split_totals.get(split, 0)

    count = 0
    for i, sample in enumerate(ds):
        if limit is not None and i >= limit:
            break

        image = sample.get("image")
        mask = sample.get("mask")
        if mask is None:
            mask = sample.get("label")

        if image is None or mask is None:
            continue

        img_target = img_dir / f"{i:04d}.png"
        mask_target = mask_dir / f"{i:04d}.png"

        if not (img_target.exists() and mask_target.exists()):
            Image.fromarray(np.array(image)).save(img_target)
            Image.fromarray(np.array(mask)).save(mask_target)

        count += 1
        if progress_callback:
            progress_callback(count, total_expected, f"Saved sample {i:04d}")

        if count % 25 == 0 or (limit and count == limit):
            print(f"  Saved {count}/{total_expected or '?'} samples...")

    print(f"Done! Saved {count} image/mask pairs to '{out_path.resolve()}'.")
    return count


def main():
    parser = argparse.ArgumentParser(description="Download Coralscapes dataset from Hugging Face.")
    parser.add_argument("--out", type=str, default="coralscapes", help="Output directory path (default: 'coralscapes').")
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation", "test"], help="Dataset split.")
    parser.add_argument("--limit", type=int, default=None, help="Number of samples to download (leave blank for all).")
    parser.add_argument("--no-streaming", dest="streaming", action="store_false", help="Disable streaming (download full parquet).")
    parser.set_defaults(streaming=True)

    args = parser.parse_args()
    download_coralscapes(
        out_dir=args.out,
        split=args.split,
        limit=args.limit,
        streaming=args.streaming,
    )


if __name__ == "__main__":
    main()
