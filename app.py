"""
CoralSCOP - Coral Reef Segmentation Streamlit Dashboard
Powered by reefsupport/CoralSCOP ViT-B Foundation Model
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
    page_title="CoralSCOP - Coral Segmentation Dashboard",
    page_icon="🪸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling for polished theme
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
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 1px solid #bae6fd;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-card-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0369a1;
    }
    .metric-card-lbl {
        font-size: 0.85rem;
        color: #0284c7;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 5px;
    }
    .badge-coral {
        background-color: #fef08a;
        color: #854d0e;
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


def main():
    # Header
    st.markdown("<div class='main-title'>🪸 CoralSCOP Segmentation Dashboard</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-title'>Dense semantic segmentation of coral reef imagery powered by "
        "<b><a href='https://huggingface.co/reefsupport/CoralSCOP' target='_blank'>reefsupport/CoralSCOP</a></b> "
        "(SAM ViT-B with Parallel Semantic Branch)</div>",
        unsafe_allow_html=True,
    )

    # ------------------ SIDEBAR ------------------
    with st.sidebar:
        st.header("⚙️ Model & Hardware")
        device_info = get_device_info()

        # Device status pill
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

        with st.expander("🛠️ Inference Hyperparameters", expanded=False):
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

        st.divider()

    # Load model with status
    with st.spinner("Initializing CoralSCOP Foundation Model..."):
        try:
            model, current_device = get_cached_model(device_pref)
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return

    # ------------------ IMAGE INPUT TABS ------------------
    tab_upload, tab_folder, tab_demo = st.tabs(["📁 Upload Image(s)", "📂 Local Folder Scan", "🌊 Sample Coral Reefs"])

    selected_images = {}  # {filename: PIL.Image}

    with tab_upload:
        uploaded_files = st.file_uploader(
            "Upload single or multiple coral images",
            type=["jpg", "jpeg", "png", "tif", "tiff", "webp"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            for uf in uploaded_files:
                try:
                    img = Image.open(uf).convert("RGB")
                    selected_images[uf.name] = img
                except Exception as ex:
                    st.error(f"Error loading {uf.name}: {ex}")

    with tab_folder:
        folder_path = st.text_input(
            "Enter local directory path containing images:",
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
                st.success(f"Found {len(found_files)} images in folder.")
                for fp in found_files:
                    fn = os.path.basename(fp)
                    try:
                        selected_images[fn] = Image.open(fp).convert("RGB")
                    except Exception as ex:
                        st.warning(f"Could not load {fn}: {ex}")
            else:
                st.info("No matching image files found in the provided directory.")
        elif folder_path:
            st.error("Directory does not exist.")

    with tab_demo:
        demo_dir = os.path.join(os.path.dirname(__file__), "demo_images")
        if os.path.isdir(demo_dir):
            demo_files = sorted(glob.glob(os.path.join(demo_dir, "*.jpg")))
            if demo_files:
                st.markdown("Select from preloaded coral quadrat research samples:")
                cols = st.columns(min(len(demo_files), 4))
                for i, df in enumerate(demo_files):
                    fn = os.path.basename(df)
                    col = cols[i % len(cols)]
                    with col:
                        thumbnail = Image.open(df).resize((180, 180))
                        col.image(thumbnail, caption=fn[:20] + "...", use_container_width=True)
                        if col.button(f"Select #{i+1}", key=f"btn_demo_{i}"):
                            st.session_state["active_image_name"] = fn
                            st.session_state["demo_active_img"] = Image.open(df).convert("RGB")
                            st.rerun()

                if "demo_active_img" in st.session_state:
                    selected_images[st.session_state.get("active_image_name", "demo_sample.jpg")] = st.session_state["demo_active_img"]

    if not selected_images:
        st.info("👈 Please upload an image, specify a folder path, or select a sample image above to start segmentation.")
        return

    # Image Selector if multiple images available
    image_names = list(selected_images.keys())
    if len(image_names) > 1:
        current_img_name = st.selectbox("Select Active Image to Analyze:", image_names)
    else:
        current_img_name = image_names[0]

    current_image = selected_images[current_img_name]

    st.divider()

    # ------------------ SEGMENTATION EXECUTION ------------------
    # Cache key for session state
    seg_cache_key = f"seg_{current_img_name}_{points_per_side}_{iou_thresh}_{stability_thresh}_{min_area_px}"

    col_btn, col_msg = st.columns([1, 4])
    with col_btn:
        run_btn = st.button("🚀 Segment Image", type="primary", use_container_width=True)

    # Auto-run if first time on this image or button pressed
    if run_btn or seg_cache_key not in st.session_state:
        with st.spinner(f"Analyzing and segmenting '{current_img_name}' with CoralSCOP..."):
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

    # ------------------ VISUALIZATION CONTROLS ------------------
    ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4, ctrl_c5 = st.columns([2, 1, 1, 1, 2])
    with ctrl_c1:
        alpha_val = st.slider("Overlay Transparency (Alpha)", min_value=0.1, max_value=0.9, value=0.45, step=0.05)
    with ctrl_c2:
        draw_contours = st.checkbox("Borders", value=True, help="Draw sharp contour borders around corals")
    with ctrl_c3:
        draw_labels = st.checkbox("ID Badges", value=True, help="Display segment ID numbers at centroids")
    with ctrl_c4:
        draw_boxes = st.checkbox("Bounding Boxes", value=False, help="Show bounding box rectangles")
    with ctrl_c5:
        mask_options = ["All Corals"] + [f"Coral #{m['id']} ({m['area_pct']}% area)" for m in masks_info]
        selected_mask_option = st.selectbox("Highlight Specific Segment", options=mask_options, index=0)
        selected_mask_id = None
        if selected_mask_option != "All Corals":
            selected_mask_id = int(selected_mask_option.split("#")[1].split(" ")[0])

    # Generate Overlay
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

    # ------------------ MAIN VIEW (ORIGINAL VS OVERLAY) ------------------
    view_mode = st.radio("Display Layout", ["Side-by-Side", "Overlay Only", "Original Only", "Masks on Black"], horizontal=True)

    if view_mode == "Side-by-Side":
        col_orig, col_seg = st.columns(2)
        with col_orig:
            st.subheader("📷 Original Coral Image")
            st.image(current_image, use_container_width=True)
        with col_seg:
            st.subheader("🎨 CoralSCOP Segmentation Overlay")
            st.image(overlay_pil, use_container_width=True)

    elif view_mode == "Overlay Only":
        st.subheader("🎨 CoralSCOP Segmentation Overlay")
        st.image(overlay_pil, use_container_width=True)

    elif view_mode == "Original Only":
        st.subheader("📷 Original Coral Image")
        st.image(current_image, use_container_width=True)

    elif view_mode == "Masks on Black":
        st.subheader("⬛ Isolated Coral Masks")
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

    # ------------------ SIDEBAR TEXT DATA & METRICS ------------------
    with st.sidebar:
        st.header("📊 Coral Analysis & Model Data")

        # KPI Metric Cards
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

        # Instance breakdown
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

        # Full JSON Explorer
        with st.expander("🔍 Model Output Text Data (JSON)", expanded=False):
            coco_dict = build_coco_json(
                image_name=current_img_name,
                width=current_image.width,
                height=current_image.height,
                masks_info=masks_info,
            )
            # Display metadata summary
            text_export = {
                "summary": summary_stats,
                "segments": [
                    {k: v for k, v in m.items() if k not in ["mask", "color_rgb"]}
                    for m in masks_info
                ],
            }
            st.json(text_export)

        # Download Actions
        st.subheader("💾 Export Results")
        coco_json_str = json.dumps(coco_dict, indent=2)
        st.download_button(
            label="📥 Download COCO JSON",
            data=coco_json_str,
            file_name=f"{os.path.splitext(current_img_name)[0]}_coralscop_coco.json",
            mime="application/json",
            use_container_width=True,
        )

        # Overlay image download
        buf = io.BytesIO()
        overlay_pil.save(buf, format="PNG")
        st.download_button(
            label="🖼️ Download Segmented Image",
            data=buf.getvalue(),
            file_name=f"{os.path.splitext(current_img_name)[0]}_segmented.png",
            mime="image/png",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
