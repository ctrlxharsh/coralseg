"""
CoralSCOP - Coral Reef Segmentation Streamlit Studio
Two-stage workflow: 
1. Ingestion Page: Upload, Folder Scan, or Demo Samples
2. Analysis Studio: Dedicated clean viewing canvas with tight pagination and bottom analysis dashboard
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
    compute_ground_truth_metrics,
    create_gt_comparison_overlay,
    create_gt_semantic_overlay,
    create_gt_coral_overlay,
    load_coralscapes_metadata,
)
from download_dataset import download_coralscapes

# Set page config
st.set_page_config(
    page_title="CoralSCOP - Coral Segmentation Studio",
    page_icon="🪸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load external global CSS
def load_css(css_file: str = "style.css"):
    css_path = os.path.join(os.path.dirname(__file__), css_file)
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")


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
    if "ground_truth_masks" not in st.session_state:
        st.session_state["ground_truth_masks"] = {}  # {filename: PIL.Image or np.ndarray}
    if "selected_img_idx" not in st.session_state:
        st.session_state["selected_img_idx"] = 0
    if "coral_target_mode" not in st.session_state:
        st.session_state["coral_target_mode"] = "all_corals"


def render_upload_page():
    """Screen 1: Clean file upload & selection landing page with Coralscapes benchmark support."""
    st.markdown("<div class='main-title'>🪸 CoralSCOP Segmentation Studio</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-title'>Dense semantic segmentation of coral reef imagery powered by "
        "<b><a href='https://huggingface.co/reefsupport/CoralSCOP' target='_blank'>reefsupport/CoralSCOP</a></b> "
        "with benchmark evaluation on <b><a href='https://huggingface.co/datasets/EPFL-ECEO/coralscapes' target='_blank'>EPFL-ECEO/coralscapes</a></b></div>",
        unsafe_allow_html=True,
    )

    tab_demo, tab_upload, tab_folder, tab_download = st.tabs([
        "🌊 Coralscapes Benchmark Dataset",
        "📁 Upload Image(s)",
        "📂 Local Folder Scan",
        "📥 Coralscapes Downloader",
    ])

    stage_images = {}
    stage_masks = {}

    with tab_demo:
        st.markdown("#### 🏆 EPFL-ECEO/coralscapes Ground Truth Benchmark")
        st.caption(
            "Dense coral reef quadrat imagery paired with official expert-annotated ground truth semantic masks. "
            "Enables pixel-exact **Intersection over Union (IoU)**, **Dice**, **Precision**, and **Recall** evaluation."
        )

        # Collect benchmark images across all dataset folders
        benchmark_entries = []
        seen_filenames = set()

        search_locations = [
            (os.path.join(os.path.dirname(__file__), "coralscapes", "images"), os.path.join(os.path.dirname(__file__), "coralscapes", "masks")),
            (os.path.join(os.path.dirname(__file__), "demo_images"), os.path.join(os.path.dirname(__file__), "demo_images", "masks")),
        ]

        for img_dir, mask_dir in search_locations:
            if os.path.isdir(img_dir):
                patterns = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.webp")
                found = []
                for pat in patterns:
                    found.extend(glob.glob(os.path.join(img_dir, pat)))
                for img_fp in sorted(found):
                    fn = os.path.basename(img_fp)
                    stem = os.path.splitext(fn)[0]
                    if fn not in seen_filenames:
                        seen_filenames.add(fn)
                        # Check matching mask
                        mask_fp = None
                        if os.path.isdir(mask_dir):
                            for candidate in [
                                os.path.join(mask_dir, fn),
                                os.path.join(mask_dir, f"{stem}.png"),
                                os.path.join(mask_dir, f"{stem}.jpg"),
                            ]:
                                if os.path.exists(candidate):
                                    mask_fp = candidate
                                    break
                        benchmark_entries.append({
                            "img_path": img_fp,
                            "filename": fn,
                            "mask_path": mask_fp,
                            "has_gt": mask_fp is not None,
                        })

        total_benchmarks = len(benchmark_entries)
        gt_available_count = sum(1 for b in benchmark_entries if b["has_gt"])

        if total_benchmarks > 0:
            st.write(f"Found **{total_benchmarks}** benchmark images ready in dataset (**{gt_available_count}** with Ground Truth masks):")

            # Thumbnail previews (show first 8)
            cols = st.columns(min(total_benchmarks, 4))
            for i, b in enumerate(benchmark_entries[:8]):
                col = cols[i % len(cols)]
                with col:
                    thumb = Image.open(b["img_path"]).resize((180, 180))
                    badge = "✅ GT Mask" if b["has_gt"] else "No Mask"
                    col.image(thumb, caption=f"#{i+1}: {b['filename'][:16]} ({badge})", width="stretch")

            if total_benchmarks > 8:
                st.caption(f"*Showing 8 of {total_benchmarks} available benchmark images.*")

            # Load options
            col_sel, col_load = st.columns([1.5, 3], vertical_alignment="bottom")
            with col_sel:
                batch_options = [f"All Available ({total_benchmarks})"]
                for n in [10, 25, 50, 100, 250, 500]:
                    if total_benchmarks > n:
                        batch_options.append(f"First {n} Samples")
                selected_batch = st.selectbox("Batch Size to Load", options=batch_options, index=0)

            with col_load:
                load_limit = total_benchmarks
                if "First" in selected_batch:
                    load_limit = int(selected_batch.split(" ")[1])

                if st.button(f"🌊 Load {min(load_limit, total_benchmarks)} Coralscapes Images for Analysis & IoU ➔", type="primary", width="stretch"):
                    for b in benchmark_entries[:load_limit]:
                        fn = b["filename"]
                        stage_images[fn] = Image.open(b["img_path"]).convert("RGB")
                        if b["mask_path"]:
                            stage_masks[fn] = Image.open(b["mask_path"])

                    st.session_state["loaded_images"] = stage_images
                    st.session_state["ground_truth_masks"] = stage_masks
                    st.session_state["selected_img_idx"] = 0
                    st.session_state["app_stage"] = "analysis"
                    st.rerun()

        else:
            st.info("No benchmark images found locally yet. Use the '📥 Coralscapes Downloader' tab to download images.")

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

            with st.expander("Optional: Upload Matching Ground Truth Masks (.png, .tif)", expanded=False):
                st.caption("Upload masks with matching file names to enable Ground Truth IoU benchmark calculations.")
                uploaded_masks = st.file_uploader(
                    "Upload Ground Truth Masks",
                    type=["png", "tif", "tiff", "jpg", "jpeg"],
                    accept_multiple_files=True,
                    key="gt_mask_uploader",
                )
                if uploaded_masks:
                    for um in uploaded_masks:
                        try:
                            stage_masks[um.name] = Image.open(um)
                        except Exception as ex:
                            st.warning(f"Could not load mask {um.name}: {ex}")

            if stage_images:
                st.success(f"✅ Loaded {len(stage_images)} image(s) ({len(stage_masks)} with GT masks) ready for analysis.")
                cols = st.columns(min(len(stage_images), 5))
                for i, (fn, img) in enumerate(stage_images.items()):
                    with cols[i % len(cols)]:
                        st.image(img.resize((160, 160)), caption=fn[:18] + "...", width="stretch")

    with tab_folder:
        st.markdown("#### Scan Local Directory")
        folder_path = st.text_input(
            "Enter path to folder containing coral images (e.g. `coralscapes/images`):",
            value="coralscapes/images" if os.path.exists("coralscapes/images") else "",
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
                gt_count = 0
                for fp in found_files:
                    fn = os.path.basename(fp)
                    stem = os.path.splitext(fn)[0]
                    try:
                        stage_images[fn] = Image.open(fp).convert("RGB")
                    except Exception as ex:
                        st.warning(f"Could not load {fn}: {ex}")
                        continue

                    # Search for matching mask in common layout locations
                    candidate_masks = [
                        os.path.join(folder_path, "masks", fn),
                        os.path.join(folder_path, "masks", f"{stem}.png"),
                        os.path.join(os.path.dirname(folder_path), "masks", fn),
                        os.path.join(os.path.dirname(folder_path), "masks", f"{stem}.png"),
                    ]
                    for cm in candidate_masks:
                        if os.path.exists(cm):
                            try:
                                stage_masks[fn] = Image.open(cm)
                                gt_count += 1
                                break
                            except Exception:
                                pass

                st.success(f"✅ Found {len(found_files)} images ({gt_count} with corresponding ground truth masks).")
                cols = st.columns(min(len(stage_images), 5))
                for i, (fn, img) in enumerate(list(stage_images.items())[:5]):
                    with cols[i % len(cols)]:
                        st.image(img.resize((160, 160)), caption=fn[:18] + "...", width="stretch")
            else:
                st.info("No supported image files found in the directory.")
        elif folder_path:
            st.error("Directory not found.")

    with tab_download:
        st.markdown("#### 📥 Download EPFL-ECEO/coralscapes from Hugging Face")
        st.markdown(
            "Download full splits or batches of the official **Coralscapes** dataset directly to `coralscapes/` on disk. "
            "Images and ground truth masks are automatically paired for testing and IoU benchmarking."
        )

        dl_c1, dl_c2, dl_c3 = st.columns(3)
        with dl_c1:
            dl_split = st.selectbox("Dataset Split", options=["train", "validation", "test"], index=0)
        with dl_c2:
            dl_limit_opt = st.selectbox("Number of Samples", options=["10 samples", "25 samples", "50 samples", "100 samples", "All (Full Split)"], index=0)
        with dl_c3:
            dl_out_dir = st.text_input("Output Directory", value="coralscapes")

        limit_val = None if "All" in dl_limit_opt else int(dl_limit_opt.split(" ")[0])

        if st.button("⬇️ Download Samples from Hugging Face", type="primary"):
            prog_bar = st.progress(0, text="Initializing download...")
            status_box = st.empty()

            def _dl_cb(cur, tot, msg):
                pct = int((cur / max(tot, 1)) * 100) if tot > 0 else 0
                pct = min(pct, 100)
                prog_bar.progress(pct, text=f"[{cur}/{tot or '?'}] {msg}")

            try:
                with st.spinner("Downloading Coralscapes from Hugging Face..."):
                    saved_count = download_coralscapes(
                        out_dir=dl_out_dir,
                        split=dl_split,
                        limit=limit_val,
                        streaming=True,
                        progress_callback=_dl_cb,
                    )
                status_box.success(f"🎉 Successfully downloaded {saved_count} samples to `{dl_out_dir}`!")
            except Exception as e:
                status_box.error(f"Download error: {e}")

        # Shortcut if coralscapes directory exists
        if os.path.exists("coralscapes/images"):
            existing_imgs = glob.glob("coralscapes/images/*.png")
            if existing_imgs:
                st.info(f"📂 Found **{len(existing_imgs)}** samples currently stored in `coralscapes/`.")
                if st.button("🚀 Load Stored `coralscapes/` Samples into Studio", key="btn_load_stored_coral"):
                    for fp in sorted(existing_imgs):
                        fn = os.path.basename(fp)
                        stage_images[fn] = Image.open(fp).convert("RGB")
                        mask_path = os.path.join("coralscapes", "masks", fn)
                        if os.path.exists(mask_path):
                            stage_masks[fn] = Image.open(mask_path)
                    st.session_state["loaded_images"] = stage_images
                    st.session_state["ground_truth_masks"] = stage_masks
                    st.session_state["selected_img_idx"] = 0
                    st.session_state["app_stage"] = "analysis"
                    st.rerun()

    # Proceed Button if images are staged
    if stage_images:
        st.divider()
        c_btn, _ = st.columns([2, 3])
        with c_btn:
            lbl = f"🚀 Analyze {len(stage_images)} Image(s)"
            if stage_masks:
                lbl += f" with IoU Benchmark ({len(stage_masks)} masks)"
            lbl += " ➔"
            if st.button(lbl, type="primary", width="stretch"):
                st.session_state["loaded_images"] = stage_images
                st.session_state["ground_truth_masks"] = stage_masks
                st.session_state["selected_img_idx"] = 0
                st.session_state["app_stage"] = "analysis"
                st.rerun()


def render_analysis_page(model):
    """Screen 2: Dedicated analysis studio with clean navigation, images, and bottom analysis dashboard."""
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

    # ------------------ TOP BAR: BACK BUTTON & FULL TITLE ------------------
    top_c1, top_c2 = st.columns([1.8, 8.2], vertical_alignment="center")
    with top_c1:
        if st.button("⬅ Back to Upload", width="stretch", key="btn_back_to_upload"):
            st.session_state["app_stage"] = "upload"
            st.rerun()
    with top_c2:
        st.markdown(
            f"<div class='image-title-bar'>"
            f"<span>🖼️</span> <span class='img-title-text'><b>{current_img_name}</b></span> "
            f"<span class='img-badge-counter'>Image {cur_idx + 1} of {total_images}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ------------------ DEDICATED PAGINATION ROW (CENTERED) ------------------
    if total_images > 1:
        def _on_page_change():
            pag_key = f"paginator_{total_images}"
            if pag_key in st.session_state:
                st.session_state["selected_img_idx"] = st.session_state[pag_key] - 1

        _, pag_col, _ = st.columns([1, 4, 1])
        with pag_col:
            selected_page = st.pagination(
                num_pages=total_images,
                default=cur_idx + 1,
                key=f"paginator_{total_images}",
                on_change=_on_page_change,
            )
            if selected_page - 1 != cur_idx:
                st.session_state["selected_img_idx"] = selected_page - 1
                st.rerun()

    # Check if Ground Truth mask is available for current image
    gt_masks_dict = st.session_state.get("ground_truth_masks", {})
    current_gt_mask = gt_masks_dict.get(current_img_name)
    has_gt = current_gt_mask is not None

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
                key="param_points_per_side",
                help="Higher values detect smaller coral instances but take longer to process.",
            )
            iou_thresh = st.slider(
                "Pred IoU Threshold",
                min_value=0.50,
                max_value=0.98,
                value=0.86,
                step=0.02,
                key="param_iou_thresh",
                help="Filters masks with low model predicted quality.",
            )
            stability_thresh = st.slider(
                "Stability Score Threshold",
                min_value=0.50,
                max_value=0.99,
                value=0.92,
                step=0.01,
                key="param_stability_thresh",
                help="Stability of mask boundaries across threshold cutoffs.",
            )
            min_area_px = st.number_input(
                "Minimum Mask Area (px)",
                min_value=10,
                max_value=50000,
                value=100,
                step=50,
                key="param_min_area_px",
                help="Removes tiny noise fragments below this pixel count.",
            )

        # 2. Ground Truth Benchmark Options (if GT mask exists)
        target_mode_code = "all_corals"
        if has_gt:
            with st.expander("🎯 Ground Truth IoU Target", expanded=True):
                target_mode_label = st.radio(
                    "Coral Class Definition",
                    options=["All Corals (Live, Bleached, Dead)", "Live Corals Only", "All Non-Background (>0)"],
                    index=0,
                    key="gt_target_mode_radio",
                    help="Defines which classes from the original dataset are counted as Coral.",
                )
                if "Live" in target_mode_label:
                    target_mode_code = "live_corals"
                elif "Non-Background" in target_mode_label:
                    target_mode_code = "all_benthic"
                else:
                    target_mode_code = "all_corals"

        # 3. Run Segmentation with Instant Area Filtering
        base_seg_cache_key = f"base_seg_{current_img_name}_{points_per_side}_{iou_thresh}_{stability_thresh}"
        if base_seg_cache_key not in st.session_state:
            with st.spinner(f"Segmenting '{current_img_name}' with CoralSCOP..."):
                img_np = np.array(current_image)
                base_masks_info, _ = run_segmentation(
                    model=model,
                    image=img_np,
                    points_per_side=points_per_side,
                    pred_iou_thresh=iou_thresh,
                    stability_score_thresh=stability_thresh,
                    min_mask_region_area=0,  # capture all masks
                )
                st.session_state[base_seg_cache_key] = base_masks_info

        all_candidate_masks = st.session_state[base_seg_cache_key]

        # Strictly filter out any masks smaller than min_area_px
        masks_info = [
            dict(m, id=new_idx + 1)
            for new_idx, m in enumerate([m for m in all_candidate_masks if m["area_px"] >= min_area_px])
        ]

        # Recompute summary stats for filtered masks
        img_h, img_w = current_image.height, current_image.width
        total_pixels = img_h * img_w
        union_mask = np.zeros((img_h, img_w), dtype=bool)
        for m in masks_info:
            union_mask = np.logical_or(union_mask, m["mask"])

        coral_covered_pixels = int(np.sum(union_mask))
        coral_coverage_pct = round((coral_covered_pixels / total_pixels) * 100.0, 2)
        mean_iou = round(float(np.mean([m["predicted_iou"] for m in masks_info])), 4) if masks_info else 0.0
        mean_stability = round(float(np.mean([m["stability_score"] for m in masks_info])), 4) if masks_info else 0.0

        summary_stats = {
            "total_corals_detected": len(masks_info),
            "coral_coverage_pct": coral_coverage_pct,
            "coral_covered_pixels": coral_covered_pixels,
            "total_image_pixels": total_pixels,
            "image_resolution": f"{img_w}x{img_h}",
            "mean_predicted_iou": mean_iou,
            "mean_stability_score": mean_stability,
        }

        # Compute Ground Truth Metrics if available
        gt_metrics = None
        if has_gt:
            gt_cache_key = f"gt_{current_img_name}_{base_seg_cache_key}_{min_area_px}_{target_mode_code}"
            if gt_cache_key not in st.session_state:
                st.session_state[gt_cache_key] = compute_ground_truth_metrics(
                    masks_info=masks_info,
                    gt_mask=current_gt_mask,
                    target_mode=target_mode_code,
                    target_shape=(current_image.height, current_image.width),
                )
            gt_metrics = st.session_state[gt_cache_key]

        # 4. Collapsible Display & Overlay Controls
        with st.expander("🎨 Display & Overlay Controls", expanded=True):
            st.markdown("**Layout View**")
            if has_gt:
                layout_opts = [
                    "Pred vs GT (Side-by-Side)",
                    "Original vs Prediction",
                    "IoU Diagnostic Map",
                    "Ground Truth Mask",
                    "Overlay Only",
                    "Masks on Black",
                    "Original Only",
                ]
                default_view = "Pred vs GT (Side-by-Side)"
            else:
                layout_opts = ["Original vs Prediction", "Overlay Only", "Original Only", "Masks on Black"]
                default_view = "Original vs Prediction"

            view_mode = st.segmented_control(
                "Display Layout",
                options=layout_opts,
                default=default_view,
                label_visibility="collapsed",
                key="side_view_mode",
            )
            if not view_mode:
                view_mode = default_view

            alpha_val = st.slider(
                "Overlay Transparency (Alpha)",
                min_value=0.1,
                max_value=0.9,
                value=0.45,
                step=0.05,
                key="ctrl_alpha",
            )

            chk_c1, chk_c2 = st.columns(2)
            with chk_c1:
                draw_contours = st.checkbox(
                    "Borders",
                    value=True,
                    key="ctrl_draw_contours",
                    help="Draw sharp contour borders around corals",
                )
                draw_labels = st.checkbox(
                    "ID Badges",
                    value=True,
                    key="ctrl_draw_labels",
                    help="Display segment ID numbers at centroids",
                )
            with chk_c2:
                draw_boxes = st.checkbox(
                    "Bounding Boxes",
                    value=False,
                    key="ctrl_draw_boxes",
                    help="Show bounding box rectangles",
                )

            mask_options = ["All Corals"] + [f"Coral #{m['id']} ({m['area_pct']}% area)" for m in masks_info]
            selected_mask_option = st.selectbox(
                "Highlight Specific Segment",
                options=mask_options,
                index=0,
                key=f"sel_mask_{current_img_name}",
            )
            selected_mask_id = None
            if selected_mask_option != "All Corals":
                selected_mask_id = int(selected_mask_option.split("#")[1].split(" ")[0])

    # Generate Model Overlay
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

    # ------------------ MAIN VIEW: IMAGES ------------------
    if view_mode == "Pred vs GT (Side-by-Side)" and has_gt and gt_metrics is not None:
        col_gt, col_pred = st.columns(2, gap="medium")
        with col_gt:
            st.markdown(
                f"<div class='image-column-header'>🏷️ Ground Truth (Coralscapes) "
                f"<span class='img-badge-counter' style='background:#ecfdf5; color:#047857; border-color:#a7f3d0;'>"
                f"GT Coral: {gt_metrics['gt_coverage_pct']}%</span></div>",
                unsafe_allow_html=True,
            )
            gt_coral_overlay_np = create_gt_coral_overlay(
                image=current_image,
                gt_coral_mask=gt_metrics["gt_binary_mask"],
                alpha=alpha_val,
                draw_contours=draw_contours,
            )
            st.image(Image.fromarray(gt_coral_overlay_np), width="stretch")
            st.caption(f"Annotated coral pixels: `{gt_metrics['gt_coral_pixels']:,}` | Target: `{target_mode_label}`")

        with col_pred:
            st.markdown(
                f"<div class='image-column-header'>🤖 CoralSCOP Prediction "
                f"<span class='img-badge-counter' style='background:#f0fdf4; color:#059669; border-color:#6ee7b7;'>"
                f"Pred Coral: {gt_metrics['pred_coverage_pct']}% | IoU: {gt_metrics['iou_pct']}%</span></div>",
                unsafe_allow_html=True,
            )
            st.image(overlay_pil, width="stretch")
            st.caption(f"Predicted coral pixels: `{gt_metrics['pred_coral_pixels']:,}` | Detected: `{summary_stats['total_corals_detected']}` segments")

        # Directly below: Overlap diagnostic map
        st.markdown("##### 🎯 Pixel-by-Pixel IoU Diagnostic Overlap Map")
        iou_overlay_np = create_gt_comparison_overlay(
            image=current_image,
            pred_binary_mask=gt_metrics["pred_binary_mask"],
            gt_coral_mask=gt_metrics["gt_binary_mask"],
            alpha=alpha_val,
        )
        st.image(Image.fromarray(iou_overlay_np), width="stretch")
        st.caption("🟩 **Green**: True Positive (Intersection) | 🟥 **Red**: False Positive (Over-seg) | 🟦 **Blue**: False Negative (Missed GT Coral) | ⬜ **White**: GT Coral Boundaries")

    elif view_mode == "Original vs Prediction" or (view_mode == "Side-by-Side"):
        col_orig, col_seg = st.columns(2, gap="medium")
        with col_orig:
            st.markdown("<div class='image-column-header'>📷 Original Coral Image</div>", unsafe_allow_html=True)
            st.image(current_image, width="stretch")
        with col_seg:
            iou_badge = f" <b>(GT IoU: {gt_metrics['iou_pct']}%)</b>" if has_gt and gt_metrics else ""
            st.markdown(
                f"<div class='image-column-header'>🎨 CoralSCOP Prediction Overlay{iou_badge}</div>",
                unsafe_allow_html=True,
            )
            st.image(overlay_pil, width="stretch")

    elif view_mode == "IoU Diagnostic Map" and has_gt and gt_metrics is not None:
        st.markdown("<div class='image-column-header'>🎯 Prediction vs Ground Truth (IoU Diagnostic Map)</div>", unsafe_allow_html=True)
        iou_overlay_np = create_gt_comparison_overlay(
            image=current_image,
            pred_binary_mask=gt_metrics["pred_binary_mask"],
            gt_coral_mask=gt_metrics["gt_binary_mask"],
            alpha=alpha_val,
        )
        st.image(Image.fromarray(iou_overlay_np), width="stretch")
        st.caption("🟩 **Green**: True Positive (Intersection) | 🟥 **Red**: False Positive (Over-seg) | 🟦 **Blue**: False Negative (Missed GT Coral) | ⬜ **White**: GT Coral Boundaries")

    elif view_mode == "Ground Truth Mask" and has_gt:
        st.markdown("<div class='image-column-header'>🏷️ Original Ground Truth Semantic Mask (EPFL-ECEO/coralscapes)</div>", unsafe_allow_html=True)
        gt_sem_np = create_gt_semantic_overlay(
            image=current_image,
            gt_mask=current_gt_mask,
            alpha=alpha_val,
        )
        st.image(Image.fromarray(gt_sem_np), width="stretch")
        st.caption("Colored by official Coralscapes benthic class taxonomy.")

    elif view_mode == "Overlay Only":
        st.markdown("<div class='image-column-header'>🎨 CoralSCOP Segmentation Overlay</div>", unsafe_allow_html=True)
        st.image(overlay_pil, width="stretch")

    elif view_mode == "Original Only":
        st.markdown("<div class='image-column-header'>📷 Original Coral Image</div>", unsafe_allow_html=True)
        st.image(current_image, width="stretch")

    elif view_mode == "Masks on Black":
        st.markdown("<div class='image-column-header'>⬛ Isolated Coral Masks</div>", unsafe_allow_html=True)
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
        st.image(Image.fromarray(mask_only_np), width="stretch")

    # ------------------ MAIN BOTTOM: GROUND TRUTH BENCHMARK ------------------
    st.divider()

    if has_gt and gt_metrics is not None:
        st.markdown(
            f"""
            <div class='gt-banner'>
                <div class='gt-banner-title'>
                    <span>🎯</span> <span><b>Ground Truth Benchmark (EPFL-ECEO/coralscapes Evaluation)</b></span>
                </div>
                <div class='gt-legend'>
                    <div class='gt-legend-item'><span class='gt-dot' style='background:#2ecc71;'></span> <b>True Positives</b>: {gt_metrics['intersection_pixels']:,} px</div>
                    <div class='gt-legend-item'><span class='gt-dot' style='background:#e74c3c;'></span> <b>False Positives</b>: {gt_metrics['false_positives']:,} px</div>
                    <div class='gt-legend-item'><span class='gt-dot' style='background:#3498db;'></span> <b>False Negatives</b>: {gt_metrics['false_negatives']:,} px</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        gt_c1, gt_c2, gt_c3, gt_c4 = st.columns(4)
        with gt_c1:
            st.markdown(
                f"""
                <div class='metric-card-gt'>
                    <div class='metric-card-val'>{gt_metrics['iou_pct']}%</div>
                    <div class='metric-card-lbl'>Ground Truth Coral IoU</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with gt_c2:
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='metric-card-val'>{gt_metrics['dice']:.3f}</div>
                    <div class='metric-card-lbl'>Dice / F1 Score</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with gt_c3:
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='metric-card-val'>{gt_metrics['precision']*100:.1f}%</div>
                    <div class='metric-card-lbl'>Pixel Precision</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with gt_c4:
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='metric-card-val'>{gt_metrics['recall']*100:.1f}%</div>
                    <div class='metric-card-lbl'>Pixel Recall</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Coverage comparison row
        cov_col1, cov_col2 = st.columns(2)
        with cov_col1:
            st.info(
                f"🤖 **Model Predicted Coral Coverage**: `{gt_metrics['pred_coverage_pct']}%` "
                f"({gt_metrics['pred_coral_pixels']:,} pixels)"
            )
        with cov_col2:
            st.success(
                f"🏷️ **Ground Truth Coral Coverage**: `{gt_metrics['gt_coverage_pct']}%` "
                f"({gt_metrics['gt_coral_pixels']:,} pixels)"
            )

        # Detailed benthic class breakdown from GT
        with st.expander("🔍 Ground Truth Benthic Classes in this Image (Original Dataset)", expanded=False):
            classes_df = pd.DataFrame(gt_metrics["classes_present"])
            if not classes_df.empty:
                classes_df = classes_df.rename(columns={
                    "class_id": "Class ID",
                    "class_name": "Taxon / Class Name",
                    "pixel_count": "Pixels",
                    "percentage": "% Area",
                    "is_coral_class": "Target Coral",
                })
                st.dataframe(classes_df, width="stretch", hide_index=True)
            else:
                st.caption("No annotated classes found.")

    else:
        st.caption("💡 *Tip: Load samples from '🌊 Coralscapes Benchmark' or upload a matching mask to view real Ground Truth IoU & Dice scores.*")

    # ------------------ MAIN BOTTOM: MODEL ANALYSIS & KPI ------------------
    st.markdown("<div class='section-header'>📊 Model Inference Statistics</div>", unsafe_allow_html=True)

    # 1. KPI Cards Row
    kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
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
    with kpi_c3:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-card-val'>{summary_stats['mean_predicted_iou']:.3f}</div>
                <div class='metric-card-lbl'>Avg Predicted IoU</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi_c4:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-card-val'>{summary_stats['mean_stability_score']:.3f}</div>
                <div class='metric-card-lbl'>Avg Stability</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 2. Detailed Breakdown Table & Export Options
    col_table, col_export = st.columns([5.8, 4.2], gap="large")

    with col_table:
        st.markdown("#### 📑 Detected Coral Segments Breakdown")
        st.caption(f"📐 Resolution: `{summary_stats['image_resolution']}` | Coral Pixels: `{summary_stats['coral_covered_pixels']:,}`")

        if masks_info:
            df_records = []
            for m in masks_info:
                df_records.append({
                    "ID": f"#{m['id']}",
                    "Area (%)": f"{m['area_pct']}%",
                    "IoU Score": m["predicted_iou"],
                    "Stability": m["stability_score"],
                    "BBox [x,y,w,h]": f"[{int(m['bbox'][0])}, {int(m['bbox'][1])}, {int(m['bbox'][2])}, {int(m['bbox'][3])}]",
                })
            df = pd.DataFrame(df_records)
            st.dataframe(df, width="stretch", hide_index=True, height=280)
        else:
            st.info("No coral segments detected with current threshold settings.")

    with col_export:
        st.markdown("#### 💾 Export & Data Inspector")

        coco_dict = build_coco_json(
            image_name=current_img_name,
            width=current_image.width,
            height=current_image.height,
            masks_info=masks_info,
        )
        if has_gt and gt_metrics is not None:
            coco_dict["benchmark_metrics"] = {
                "ground_truth_iou": gt_metrics["iou"],
                "dice": gt_metrics["dice"],
                "precision": gt_metrics["precision"],
                "recall": gt_metrics["recall"],
            }

        coco_json_str = json.dumps(coco_dict, indent=2)

        buf = io.BytesIO()
        overlay_pil.save(buf, format="PNG")

        exp_c1, exp_c2 = st.columns(2)
        with exp_c1:
            st.download_button(
                label="📥 Download COCO JSON",
                data=coco_json_str,
                file_name=f"{os.path.splitext(current_img_name)[0]}_coralscop_coco.json",
                mime="application/json",
                width="stretch",
            )
        with exp_c2:
            st.download_button(
                label="🖼️ Download Overlay PNG",
                data=buf.getvalue(),
                file_name=f"{os.path.splitext(current_img_name)[0]}_segmented.png",
                mime="image/png",
                width="stretch",
            )

        with st.expander("🔍 Raw Model JSON Data", expanded=False):
            text_export = {
                "summary": summary_stats,
                "benchmark": gt_metrics if has_gt else None,
                "segments": [
                    {k: v for k, v in m.items() if k not in ["mask", "color_rgb"]}
                    for m in masks_info
                ],
            }
            st.json(text_export)


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
