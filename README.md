# 🪸 CoralSCOP Streamlit Dashboard & Benchmark Studio

An interactive web dashboard for dense coral reef segmentation using the **[reefsupport/CoralSCOP](https://huggingface.co/reefsupport/CoralSCOP)** foundation model (SAM ViT-B with parallel semantic branch), featuring ground-truth **Intersection over Union (IoU)** benchmark evaluation on the **[EPFL-ECEO/coralscapes](https://huggingface.co/datasets/EPFL-ECEO/coralscapes)** dataset.

---

## ✨ Features
- **Ground Truth IoU Benchmark on Coralscapes**:
  - Test immediately with **7 preloaded Coralscapes research quadrat samples** paired with official expert ground truth masks.
  - Computes pixel-exact **Ground Truth Coral IoU**, **Dice / F1 Score**, **Pixel Precision**, and **Pixel Recall**.
  - **Pred vs GT (IoU Diagnostic Map)** visualizer:
    - 🟩 **Green**: True Positive (model prediction correctly overlaps ground truth coral)
    - 🟥 **Red**: False Positive (over-segmentation / non-coral area predicted)
    - 🟦 **Blue**: False Negative (ground truth coral missed by model)
    - ⬜ **White**: Ground Truth Coral Boundary Contours
  - Full multi-class ground truth semantic overlay color-coded using the official 39 benthic class taxonomy.
  - Class distribution table showing exact pixel area breakdown for every benthic taxon annotated in the quadrat.
- **Multi-Source Ingestion & Downloader**:
  - Built-in **one-click Coralscapes Downloader** in the UI with live progress bar.
  - Upload individual or batch images (`.jpg`, `.jpeg`, `.png`, `.tif`, `.webp`) with optional ground truth mask upload.
  - Local directory scanner with automatic ground truth mask matching.
- **Hardware Acceleration**:
  - Automatically selects **CUDA GPU** $\rightarrow$ **Apple Silicon MPS** (Metal acceleration) $\rightarrow$ **CPU** fallback.
- **Vibrant Multi-Color Overlay & Export**:
  - High-contrast, distinct color assignment per detected coral segment with alpha, border, and centroid ID controls.
  - Download COCO JSON annotations (including benchmark IoU metrics) and segmented overlay PNGs.

---

## 📥 How to Download the Coralscapes Dataset

The **Coralscapes** dataset contains dense semantic segmentations for 39 benthic classes across 2,075 coral reef quadrat images.

### Method 1: Directly from the Streamlit UI
1. Launch the web application:
   ```bash
   streamlit run app.py
   ```
2. Navigate to the **"📥 Coralscapes Downloader"** tab.
3. Select your desired split (`train`, `validation`, or `test`) and batch size (`10 samples`, `25 samples`, `50 samples`, `100 samples`, or `All (Full Split)`).
4. Click **"⬇️ Download Samples from Hugging Face"**. A real-time progress bar tracks row-by-row streaming and disk saving into `coralscapes/images` and `coralscapes/masks`.
5. Once completed, click **"🚀 Load Stored coralscapes/ Samples into Studio"** to immediately benchmark the downloaded images!

---

### Method 2: From Jupyter Notebooks, Google Colab, or Kaggle

Run this snippet in any notebook cell to download and save images and masks:

```python
from datasets import load_dataset
from pathlib import Path
import numpy as np
from PIL import Image

# Destination folders (gitignored)
OUT = Path("coralscapes")
(OUT / "images").mkdir(parents=True, exist_ok=True)
(OUT / "masks").mkdir(parents=True, exist_ok=True)

# Load Coralscapes from Hugging Face (streaming=True streams row-by-row without 4GB parquet download)
ds = load_dataset("EPFL-ECEO/coralscapes", split="train", streaming=True)

# Set limit to None to download ALL 1,517 train samples, or an integer (e.g. 50)
LIMIT = 50 

for i, sample in enumerate(ds):
    if LIMIT is not None and i >= LIMIT:
        break
    
    image = sample["image"]
    mask = sample.get("label", sample.get("mask"))

    # Save image
    Image.fromarray(np.array(image)).save(
        OUT / "images" / f"{i:04d}.png"
    )

    # Save corresponding ground-truth mask
    Image.fromarray(np.array(mask)).save(
        OUT / "masks" / f"{i:04d}.png"
    )

    if (i + 1) % 10 == 0:
        print(f"Saved {i + 1} samples...")

print(f"Done! Dataset saved to {OUT.resolve()}")
```

---

### Method 3: Download ALL (Full Dataset) via Command Line

We provide a dedicated download script `download_dataset.py`:

```bash
# Download the entire training split (1,517 images & masks):
python download_dataset.py --split train

# Download validation split (166 images & masks):
python download_dataset.py --split validation

# Download test split (392 images & masks):
python download_dataset.py --split test

# Download a specific number of samples (e.g. 100):
python download_dataset.py --split train --limit 100 --out coralscapes
```

---

## 🚀 Quickstart (Local)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Model weights `vit_b_coralscop.pth` will be automatically downloaded from Hugging Face on first run).*

### 2. Launch the Streamlit App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 3. Run Benchmark Testing
- On the landing page, click **"🌊 Load All 7 Benchmark Samples for IoU Evaluation"**.
- View model predictions alongside real **Ground Truth Coral IoU**, **Dice**, **Precision**, and **Recall**.
- Switch the **Display Layout** to **"Pred vs GT (IoU Map)"** to inspect pixel overlap, over-segmentation, and missed coral contours!

---

## ⚡ Running on Kaggle Notebooks / Google Colab

In a code cell:
```python
# 1. Clone repository
!git clone https://github.com/<your-repo>/coralseg.git
%cd coralseg

# 2. Install requirements
!pip install -r requirements.txt

# 3. Launch with Localtunnel
!python run_colab_kaggle.py
```
Or directly via terminal/cell:
```bash
streamlit run app.py --server.port 8501 --server.headless true & npx localtunnel --port 8501
```
