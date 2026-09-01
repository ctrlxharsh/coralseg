# 🪸 CoralSCOP Streamlit Dashboard

An interactive web dashboard for dense coral reef segmentation using the **[reefsupport/CoralSCOP](https://huggingface.co/reefsupport/CoralSCOP)** foundation model (SAM ViT-B with parallel semantic branch).

---

## ✨ Features
- **Multi-Source Image Ingestion**:
  - Upload individual or batch images (`.jpg`, `.jpeg`, `.png`, `.tif`, `.webp`).
  - Scan a local directory path containing coral images.
  - Test immediately with preloaded underwater coral quadrat research samples.
- **Hardware Acceleration**:
  - Automatically selects **CUDA GPU** (if available) $\rightarrow$ **Apple Silicon MPS** (Metal acceleration) $\rightarrow$ **CPU** fallback.
  - Configurable device preference dropdown.
- **Vibrant Multi-Color Overlay**:
  - High-contrast, distinct color assignment per detected coral segment.
  - Interactive transparency (alpha), contour borders, bounding boxes, and centroid ID badges.
  - Segment isolator / highlighter to inspect individual coral instances.
  - Display modes: Side-by-Side, Overlay Only, Original Only, and Isolated Masks on Black.
- **Rich Model Analytics (Sidebar)**:
  - Total corals detected count and % reef quadrat coverage.
  - Average predicted IoU confidence score and mask stability scores.
  - Tabular segment breakdown (BBox, Area %, IoU, Stability).
  - Expandable JSON data returned by the model.
  - Download COCO JSON annotations and high-res segmented images.

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

---

## ⚡ Running on Kaggle Notebooks / Google Colab

### Kaggle / Colab One-Liner Setup
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
