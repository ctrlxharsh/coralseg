"""
CoralSCOP - Coral Reef Segmentation Streamlit Studio
Two-stage workflow: 
1. Ingestion Page: Upload, Folder Scan, or Demo Samples
2. Analysis Studio: Dedicated clean viewing space with unique standalone Prev/Next pagination line
"""

import os
import glob
import json
import io
from PIL import Image
import numpy as np
import pandas as pd
import streamlit as st

from coral_engine import (
    get_device_info,
    get_optimal_device,
    load_coralscop_model,
    run_segmentation,
    create_segmentation_overlay,
    build_coco_json,
)

# Set page config
st.set_page_config(
    page_title="CoralSCOP - Coral Segmentation Studio",
    page_icon="🪸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling for modern studio feel
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0e7490;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #64748b;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 1px solid #bae6fd;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-card-val {
        font-size: 1.7rem;
        font-weight: 700;
        color: #0369a1;
    }
    .metric-card-lbl {
        font-size: 0.8rem;
        color: #0284c7;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .image-title-bar {
        font-size: 1.25rem;
        font-weight: 600;
        color: #0f172a;
        margin: 0;
        padding: 0;
    }
    .pagination-wrapper {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 8px 16px;
        margin: 12px 0 20px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_cached_model(device_preference: str = "auto"):
    """Loads and caches the CoralSCOP model in memory."""
    device = get_optimal_device(device_preference)
    return load_coralscop_model(device=device), device


def initialize_session_state():
    """Ensures necessary session state variables exist."""
    if "app_stage" not in st.session_state:
        st.session_state["app_stage"] = "upload"  # "upload" or "analysis"
    if "loaded_images" not in st.session_state:
        st.session_state["loaded_images"] = {}  # {filename: PIL.Image}
    if "selected_img_idx" not in st.session_state:
        st.session_state["selected_img_idx"] = 0


def render_upload_page():
    """Screen 1: Clean file upload & selection landing page."""
    st.markdown("<div class='main-title'>🪸 CoralSCOP Segmentation Studio</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-title'>Dense semantic segmentation of coral reef imagery powered by "
        "<b><a href='https://huggingface.co/reefsupport/CoralSCOP' target='_blank'>reefsupport/CoralSCOP</a></b> "
        "(SAM ViT-B with Parallel Semantic Branch)</div>",
        unsafe_allow_html=True,
    )

    tab_upload, tab_folder, tab_demo = st.tabs(["📁 Upload Image(s)", "📂 Local Folder Scan", "🌊 Sample Coral Reefs"])

    stage_images = {}

    with tab_upload:
        st.markdown("#### Upload Underwater Coral Images")
        uploaded_files = st.file_uploader(
            "Select one or more images (.jpg, .jpeg, .png, .tif, .webp)",
            type=["jpg", "jpeg", "png", "tif", "tiff", "webp"],
            accept_multiple_files=True,
            help="Upload one or multiple coral quadrat photos.",
        )
        if uploaded_files:
            for uf in uploaded_files:
                try:
                    img = Image.open(uf).convert("RGB")
                    stage_images[uf.name] = img
                except Exception as ex:
                    st.error(f"Error loading {uf.name}: {ex}")

            if stage_images:
                st.success(f"✅ Loaded {len(stage_images)} image(s) ready for analysis.")
                cols = st.columns(min(len(stage_images), 5))
                for i, (fn, img) in enumerate(stage_images.items()):
                    with cols[i % len(cols)]:
                        st.image(img.resize((160, 160)), caption=fn[:18] + "...", use_container_width=True)

    with tab_folder:
        st.markdown("#### Scan Local Directory")
        folder_path = st.text_input(
            "Enter path to folder containing coral images:",
            value="",
            placeholder="/path/to/coral/images",
        )
        if folder_path and os.path.isdir(folder_path):
            valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.webp")
            found_files = []
            for ext in valid_exts:
                found_files.extend(glob.glob(os.path.join(folder_path, ext)))
                found_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
            found_files = sorted(list(set(found_files)))

            if found_files:
                st.success(f"✅ Found {len(found_files)} images in folder.")
                for fp in found_files:
                    fn = os.path.basename(fp)
                    try:
                        stage_images[fn] = Image.open(fp).convert("RGB")
                    except Exception as ex:
                        st.warning(f"Could not load {fn}: {ex}")

                cols = st.columns(min(len(stage_images), 5))
                for i, (fn, img) in enumerate(list(stage_images.items())[:5]):
                    with cols[i % len(cols)]:
                        st.image(img.resize((160, 160)), caption=fn[:18] + "...", use_container_width=True)
            else:
                st.info("No supported image files found in the directory.")
        elif folder_path:
            st.error("Directory not found.")

    with tab_demo:
        st.markdown("#### Research Sample Datasets")
        demo_dir = os.path.join(os.path.dirname(__file__), "demo_images")
        if os.path.isdir(demo_dir):
            demo_files = sorted(glob.glob(os.path.join(demo_dir, "*.jpg")))
            if demo_files:
                st.write(f"Preloaded with {len(demo_files)} coral reef quadrat samples:")
                cols = st.columns(min(len(demo_files), 4))
                for i, df in enumerate(demo_files):
                    fn = os.path.basename(df)
                    col = cols[i % len(cols)]
                    with col:
                        thumbnail = Image.open(df).resize((180, 180))
                        col.image(thumbnail, caption=f"Sample #{i+1}", use_container_width=True)

                if st.button("🌊 Load All Sample Images for Analysis", type="primary"):
                    for df in demo_files:
                        fn = os.path.basename(df)
                        stage_images[fn] = Image.open(df).convert("RGB")
                    st.session_state["loaded_images"] = stage_images
                    st.session_state["selected_img_idx"] = 0
                    st.session_state["app_stage"] = "analysis"
                    st.rerun()

    # Proceed Button if images are staged
    if stage_images:
        st.divider()
        c_btn, _ = st.columns([2, 3])
        with c_btn:
            if st.button(f"🚀 Analyze {len(stage_images)} Image(s) ➔", type="primary", use_container_width=True):
                st.session_state["loaded_images"] = stage_images
                st.session_state["selected_img_idx"] = 0
                st.session_state["app_stage"] = "analysis"
                st.rerun()


def render_analysis_page(model):
    """Screen 2: Dedicated analysis studio with separate top bar, standalone pagination line, and clean image canvas."""
    images_dict = st.session_state.get("loaded_images", {})
    if not images_dict:
        st.session_state["app_stage"] = "upload"
        st.rerun()

    image_names = list(images_dict.keys())
    total_images = len(image_names)

    # Ensure index is within range
    if st.session_state["selected_img_idx"] >= total_images:
        st.session_state["selected_img_idx"] = 0

    cur_idx = st.session_state["selected_img_idx"]
    current_img_name = image_names[cur_idx]
    current_image = images_dict[current_img_name]

    # ------------------ TOP BAR: BACK BUTTON & TITLE ONLY ------------------
    top_c1, top_c2 = st.columns([1.8, 8.2], vertical_alignment="center")
    with top_c1:
        if st.button("⬅ Back to Upload", use_container_width=True):
            st.session_state["app_stage"] = "upload"
            st.rerun()
    with top_c2:
        st.markdown(
            f"<div class='image-title-bar'>🖼️ <b>{current_img_name}</b> "
            f"<span style='color:#64748b; font-size:1rem; font-weight:normal;'>(Image {cur_idx + 1} of {total_images})</span></div>",
            unsafe_allow_html=True,
        )

    # ------------------ DEDICATED STANDALONE PAGINATION LINE ------------------
    if total_images > 1:
        st.markdown("<div class='pagination-wrapper'>", unsafe_allow_html=True)
        _, nav_prev, nav_nums, nav_next, _ = st.columns([2.0, 1.2, 5.6, 1.2, 2.0], vertical_alignment="center")

        with nav_prev:
            if st.button("◀ Prev", disabled=(cur_idx == 0), use_container_width=True, key="nav_btn_prev"):
                st.session_state["selected_img_idx"] = max(0, cur_idx - 1)
                st.rerun()

        with nav_nums:
            page_options = [f"{i+1}" for i in range(total_images)]
            selected_page = st.segmented_control(
                "Image Page Navigation",
                options=page_options,
                default=page_options[cur_idx],
                label_visibility="collapsed",
                key="nav_page_segmented",
            )
            if selected_page and int(selected_page) - 1 != cur_idx:
                st.session_state["selected_img_idx"] = int(selected_page) - 1
                st.rerun()

        with nav_next:
            if st.button("Next ▶", disabled=(cur_idx >= total_images - 1), use_container_width=True, key="nav_btn_next"):
                st.session_state["selected_img_idx"] = min(total_images - 1, cur_idx + 1)
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    # ------------------ SIDEBAR CONTROLS & PARAMS ------------------
    with st.sidebar:
        # 1. Hyperparameters
        with st.expander("🛠️ Inference Parameters", expanded=False):
            points_per_side = st.slider(
                "Points Per Side (Grid Density)",
                min_value=8,
                max_value=48,
                value=24,
                step=4,
                help="Higher values detect smaller coral instances but take longer to process.",
            )
            iou_thresh = st.slider(
                "Pred IoU Threshold",
                min_value=0.50,
                max_value=0.98,
                value=0.86,
                step=0.02,
                help="Filters masks with low model predicted quality.",
            )
            stability_thresh = st.slider(
                "Stability Score Threshold",
                min_value=0.50,
                max_value=0.99,
                value=0.92,
                step=0.01,
                help="Stability of mask boundaries across threshold cutoffs.",
            )
            min_area_px = st.number_input(
                "Minimum Mask Area (px)",
                min_value=10,
                max_value=50000,
                value=100,
                step=50,
                help="Removes tiny noise fragments below this pixel count.",
            )

        # 2. Run Segmentation
        seg_cache_key = f"seg_{current_img_name}_{points_per_side}_{iou_thresh}_{stability_thresh}_{min_area_px}"
        if seg_cache_key not in st.session_state:
            with st.spinner(f"Segmenting '{current_img_name}' with CoralSCOP..."):
                img_np = np.array(current_image)
                masks_info, summary_stats = run_segmentation(
                    model=model,
                    image=img_np,
                    points_per_side=points_per_side,
                    pred_iou_thresh=iou_thresh,
                    stability_score_thresh=stability_thresh,
                    min_mask_region_area=min_area_px,
                )
                st.session_state[seg_cache_key] = {
                    "masks_info": masks_info,
                    "summary_stats": summary_stats,
                }

        seg_result = st.session_state[seg_cache_key]
        masks_info = seg_result["masks_info"]
        summary_stats = seg_result["summary_stats"]

        # 3. Collapsible Display & Overlay Controls
        with st.expander("🎨 Display & Overlay Controls", expanded=True):
            st.markdown("**Layout View**")
            view_mode = st.segmented_control(
                "Display Layout",
                options=["Side-by-Side", "Overlay Only", "Original Only", "Masks on Black"],
                default="Side-by-Side",
                label_visibility="collapsed",
                key="side_view_mode",
            )
            if not view_mode:
                view_mode = "Side-by-Side"

            alpha_val = st.slider("Overlay Transparency (Alpha)", min_value=0.1, max_value=0.9, value=0.45, step=0.05)

            chk_c1, chk_c2 = st.columns(2)
            with chk_c1:
                draw_contours = st.checkbox("Borders", value=True, help="Draw sharp contour borders around corals")
                draw_labels = st.checkbox("ID Badges", value=True, help="Display segment ID numbers at centroids")
            with chk_c2:
                draw_boxes = st.checkbox("Bounding Boxes", value=False, help="Show bounding box rectangles")

            mask_options = ["All Corals"] + [f"Coral #{m['id']} ({m['area_pct']}% area)" for m in masks_info]
            selected_mask_option = st.selectbox("Highlight Specific Segment", options=mask_options, index=0)
            selected_mask_id = None
            if selected_mask_option != "All Corals":
                selected_mask_id = int(selected_mask_option.split("#")[1].split(" ")[0])

        st.divider()

        # 4. Coral Analysis Metrics
        st.header("📊 Coral Analysis")
        kpi_c1, kpi_c2 = st.columns(2)
        with kpi_c1:
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='metric-card-val'>{summary_stats['total_corals_detected']}</div>
                    <div class='metric-card-lbl'>Corals Found</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi_c2:
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='metric-card-val'>{summary_stats['coral_coverage_pct']}%</div>
                    <div class='metric-card-lbl'>Reef Coverage</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        kpi_c3, kpi_c4 = st.columns(2)
        with kpi_c3:
            st.metric("Avg Predicted IoU", f"{summary_stats['mean_predicted_iou']:.3f}")
        with kpi_c4:
            st.metric("Avg Stability", f"{summary_stats['mean_stability_score']:.3f}")

        st.caption(f"📐 Resolution: `{summary_stats['image_resolution']}` | Coral Pixels: `{summary_stats['coral_covered_pixels']:,}`")

        # 5. Detected Coral Segments Table
        st.subheader("📑 Detected Coral Segments")
        if masks_info:
            df_records = []
            for m in masks_info:
                df_records.append({
                    "ID": f"#{m['id']}",
                    "Area (%)": f"{m['area_pct']}%",
                    "IoU": m["predicted_iou"],
                    "Stability": m["stability_score"],
                    "BBox [x,y,w,h]": f"[{int(m['bbox'][0])}, {int(m['bbox'][1])}, {int(m['bbox'][2])}, {int(m['bbox'][3])}]",
                })
            df = pd.DataFrame(df_records)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No coral segments detected with current threshold settings.")

        # 6. JSON Explorer & Downloads
        with st.expander("🔍 Model Output Text Data (JSON)", expanded=False):
            coco_dict = build_coco_json(
                image_name=current_img_name,
                width=current_image.width,
                height=current_image.height,
                masks_info=masks_info,
            )
            text_export = {
                "summary": summary_stats,
                "segments": [
                    {k: v for k, v in m.items() if k not in ["mask", "color_rgb"]}
                    for m in masks_info
                ],
            }
            st.json(text_export)

        st.subheader("💾 Export Results")
        coco_json_str = json.dumps(coco_dict, indent=2)
        st.download_button(
            label="📥 Download COCO JSON",
            data=coco_json_str,
            file_name=f"{os.path.splitext(current_img_name)[0]}_coralscop_coco.json",
            mime="application/json",
            use_container_width=True,
        )

        # Generate Overlay for Main View and Download
        overlay_img_np = create_segmentation_overlay(
            image=current_image,
            masks_info=masks_info,
            alpha=alpha_val,
            draw_contours=draw_contours,
            draw_labels=draw_labels,
            draw_boxes=draw_boxes,
            selected_mask_id=selected_mask_id,
        )
        overlay_pil = Image.fromarray(overlay_img_np)

        buf = io.BytesIO()
        overlay_pil.save(buf, format="PNG")
        st.download_button(
            label="🖼️ Download Segmented Image",
            data=buf.getvalue(),
            file_name=f"{os.path.splitext(current_img_name)[0]}_segmented.png",
            mime="image/png",
            use_container_width=True,
        )

    # ------------------ MAIN VIEW: IMAGES ONLY ------------------
    if view_mode == "Side-by-Side":
        col_orig, col_seg = st.columns(2, gap="medium")
        with col_orig:
            st.markdown("### 📷 Original Coral Image")
            st.image(current_image, use_container_width=True)
        with col_seg:
            st.markdown("### 🎨 CoralSCOP Segmentation Overlay")
            st.image(overlay_pil, use_container_width=True)

    elif view_mode == "Overlay Only":
        st.markdown("### 🎨 CoralSCOP Segmentation Overlay")
        st.image(overlay_pil, use_container_width=True)

    elif view_mode == "Original Only":
        st.markdown("### 📷 Original Coral Image")
        st.image(current_image, use_container_width=True)

    elif view_mode == "Masks on Black":
        st.markdown("### ⬛ Isolated Coral Masks")
        black_bg = np.zeros_like(np.array(current_image))
        mask_only_np = create_segmentation_overlay(
            image=black_bg,
            masks_info=masks_info,
            alpha=1.0,
            draw_contours=draw_contours,
            draw_labels=draw_labels,
            draw_boxes=draw_boxes,
            selected_mask_id=selected_mask_id,
        )
        st.image(Image.fromarray(mask_only_np), use_container_width=True)


def main():
    initialize_session_state()

    # Hardware Info in Sidebar
    with st.sidebar:
        st.header("⚙️ Hardware Status")
        device_info = get_device_info()
        device_label = device_info.get("gpu_name", "CPU")
        if device_info["cuda_available"]:
            st.success(f"🚀 **CUDA GPU**: {device_label}")
        elif device_info["mps_available"]:
            st.info(f"⚡ **Apple Silicon**: {device_label}")
        else:
            st.warning(f"💻 **CPU Mode**: {device_label}")

        device_pref = st.selectbox(
            "Device Preference",
            options=["auto", "cuda", "mps", "cpu"],
            index=0,
            help="Prioritizes CUDA -> MPS -> CPU by default.",
        )
        st.divider()

    # Load cached model
    with st.spinner("Initializing CoralSCOP Foundation Model..."):
        try:
            model, _ = get_cached_model(device_pref)
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return

    # Render appropriate screen based on state
    if st.session_state["app_stage"] == "upload":
        render_upload_page()
    else:
        render_analysis_page(model)


if __name__ == "__main__":
    main()
