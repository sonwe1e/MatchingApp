"""
Streamlit Web Application for Advanced Feature Point Matching.

This application allows users to:
1. Upload two images.
2. Select feature detection algorithms (OpenCV classics or Kornia's DISK).
3. Select feature matching algorithms (OpenCV Brute-Force or Kornia's LightGlue).
4. Adjust parameters for selected algorithms.
5. Compute and visualize keypoints and matches.
6. Compare results from different algorithm combinations.

Structure:
- Configuration and Initialization: Sets up page, device, and session state.
- UI Layout: Defines sidebar for controls and main area for image display and results.
  - Sidebar: Image uploads, algorithm selection, parameter tuning.
  - Main Area: Original images, current keypoint/match visualization, comparison section.
- Processing Logic: Handles the "Compute" button action.
  - Feature Detection: Calls functions from `core.detectors`.
  - Feature Matching: Calls functions from `core.matchers`.
  - Result Storage: Stores match images and stats in session state for comparison.
- Visualization: Displays keypoints on images and lines connecting matched points.
"""

import streamlit as st
import cv2
import numpy as np

# Project-specific imports
import config  # For constants and default values
from utils import (
    get_device_info,
    load_image_from_upload,
    convert_to_opencv_gray,
    draw_matches_comparison_image,
)
from core.detectors import get_detector_model_or_instance, extract_features
from core.matchers import get_matcher_model_or_instance, perform_matching

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(layout="wide", page_title="Advanced Feature Matching Tool")

# --- Global Variables & Initialization ---
# Get device (CPU/GPU) using the cached utility function
# This is a good practice to ensure device determination is consistent and efficient.
APP_DEVICE, APP_DEVICE_NAME = get_device_info()


# --- Session State Initialization ---
# Ensures that necessary keys exist in st.session_state from the beginning.
# Using keys from config.py for consistency and to avoid typos.
def initialize_session_state():
    """Initializes session state variables if they don't exist."""
    defaults = {
        config.KEY_IMG1_RGB: None,
        config.KEY_IMG2_RGB: None,
        config.KEY_IMG1_GRAY: None,
        config.KEY_IMG2_GRAY: None,
        config.KEY_KP1: [],
        config.KEY_DES1: None,
        config.KEY_KP2: [],
        config.KEY_DES2: None,
        config.KEY_MATCHES: [],
        config.KEY_COMPARISON_RESULTS: {},
        config.KEY_DETECTOR_NAME: config.SUPPORTED_DETECTORS[0],  # Default detector
        config.KEY_MATCHER_NAME: config.SUPPORTED_MATCHERS[0],  # Default matcher
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


initialize_session_state()


# --- UI: Sidebar for Controls ---
with st.sidebar:
    st.header("⚙️ Control Panel")
    st.info(f"PyTorch using: {APP_DEVICE_NAME}")
    if APP_DEVICE_NAME == "CPU" and (
        config.DETECTOR_DISK_KORNIA in st.session_state[config.KEY_DETECTOR_NAME]
        or config.MATCHER_LIGHTGLUE_KORNIA in st.session_state[config.KEY_MATCHER_NAME]
    ):
        st.warning("Kornia methods (DISK, LightGlue) can be slow on CPU.")

    st.subheader("1. Upload Images")
    img_file1_upload = st.file_uploader(
        "Upload Image 1", type=["png", "jpg", "jpeg", "bmp", "tiff"], key="img1_upload"
    )
    img_file2_upload = st.file_uploader(
        "Upload Image 2", type=["png", "jpg", "jpeg", "bmp", "tiff"], key="img2_upload"
    )

    # --- UI: Algorithm Selection and Parameters ---
    st.subheader("2. Feature Detection")
    # Use st.session_state directly for the selectbox default if needed,
    # but also update it on change using on_change or by direct assignment after selection.
    # Here, the value from selectbox is directly assigned to session_state.
    # The key for the widget itself should be unique.
    selected_detector = st.selectbox(
        "Select Detector",
        options=config.SUPPORTED_DETECTORS,
        index=config.SUPPORTED_DETECTORS.index(
            st.session_state.get(
                config.KEY_DETECTOR_NAME, config.SUPPORTED_DETECTORS[0]
            )
        ),
        key="sb_detector_name",
    )
    st.session_state[config.KEY_DETECTOR_NAME] = (
        selected_detector  # Update session state
    )

    detector_params = {}
    if selected_detector == config.DETECTOR_SIFT:
        detector_params["sift_nfeatures"] = st.slider(
            "SIFT: Max Features",
            0,
            10000,
            config.DEFAULT_SIFT_NFEATURES,
            100,
            help="0 means no limit. More features can improve matching but increase computation.",
        )
    elif selected_detector == config.DETECTOR_ORB:
        detector_params["orb_nfeatures"] = st.slider(
            "ORB: Max Features", 100, 10000, config.DEFAULT_ORB_NFEATURES, 50
        )
        detector_params["orb_scaleFactor"] = st.slider(
            "ORB: Scale Factor",
            1.1,
            2.0,
            config.DEFAULT_ORB_SCALE_FACTOR,
            0.05,
            help="Pyramid decimation ratio, should be >1.",
        )
        detector_params["orb_nlevels"] = st.slider(
            "ORB: Pyramid Levels", 1, 16, config.DEFAULT_ORB_NLEVELS, 1
        )
    elif selected_detector == config.DETECTOR_BRISK:
        detector_params["brisk_thresh"] = st.slider(
            "BRISK: FAST Threshold", 10, 100, config.DEFAULT_BRISK_THRESH, 5
        )
        detector_params["brisk_octaves"] = st.slider(
            "BRISK: Detection Octaves", 0, 6, config.DEFAULT_BRISK_OCTAVES, 1
        )
    elif selected_detector == config.DETECTOR_DISK_KORNIA:
        detector_params["disk_top_k"] = st.number_input(
            "DISK: Max Features per Image (top_k)",
            min_value=100,
            max_value=20000,
            value=config.DEFAULT_DISK_TOP_K,
            step=100,
            help="Max number of features returned by DISK, sorted by response.",
        )

    st.subheader("3. Feature Matching")
    selected_matcher = st.selectbox(
        "Select Matcher",
        options=config.SUPPORTED_MATCHERS,
        index=config.SUPPORTED_MATCHERS.index(
            st.session_state.get(config.KEY_MATCHER_NAME, config.SUPPORTED_MATCHERS[0])
        ),
        key="sb_matcher_name",
    )
    st.session_state[config.KEY_MATCHER_NAME] = selected_matcher  # Update session state

    matcher_params = {}
    if selected_matcher == config.MATCHER_BF:
        matcher_params["use_ratio_test"] = st.checkbox("Use Lowe's Ratio Test", True)
        if matcher_params["use_ratio_test"]:
            matcher_params["ratio_thresh"] = st.slider(
                "Ratio Test Threshold",
                0.1,
                1.0,
                config.DEFAULT_RATIO_TEST_THRESH,
                0.01,
                help="Lower values are more restrictive.",
            )
            matcher_params["cross_check_bf"] = (
                False  # Typically False if using ratio test
            )
        else:
            matcher_params["cross_check_bf"] = st.checkbox("Use Cross Check", True)
    elif selected_matcher == config.MATCHER_LIGHTGLUE_KORNIA:
        if not selected_detector.endswith("(Kornia)"):
            st.warning(
                "LightGlue typically performs best with Kornia detectors like DISK. "
                "Compatibility with other detectors (e.g. SIFT) is not guaranteed."
            )
        # LightGlue has internal parameters; for simplicity, not exposing them here yet.

    st.markdown("---")
    # Process button now uses values directly from session state for detector/matcher names
    # and locally scoped params dicts.
    compute_button_pressed = st.button(
        "🚀 Compute & Add to Comparison", type="primary", use_container_width=True
    )
    st.markdown("---")
    st.info("Feature Matching Tool - Enhanced Version")


# --- Image Loading and Preprocessing ---
# This logic runs on every interaction if files are uploaded.
if img_file1_upload:
    loaded_img1_rgb = load_image_from_upload(img_file1_upload)
    if loaded_img1_rgb is not None:
        st.session_state[config.KEY_IMG1_RGB] = loaded_img1_rgb
        st.session_state[config.KEY_IMG1_GRAY] = convert_to_opencv_gray(
            st.session_state[config.KEY_IMG1_RGB]
        )
if img_file2_upload:
    loaded_img2_rgb = load_image_from_upload(img_file2_upload)
    if loaded_img2_rgb is not None:
        st.session_state[config.KEY_IMG2_RGB] = loaded_img2_rgb
        st.session_state[config.KEY_IMG2_GRAY] = convert_to_opencv_gray(
            st.session_state[config.KEY_IMG2_RGB]
        )


# --- UI: Main Area for Display ---
st.title("🖼️ Advanced Feature Point Matching Tool")
st.markdown(
    "Upload two images, select feature detection and matching methods from the sidebar, "
    "tune parameters, and visualize the results. Add multiple configurations to compare."
)

col_main_left, col_main_right = st.columns(2)
with col_main_left:
    st.subheader("Original Image 1")
    if st.session_state[config.KEY_IMG1_RGB] is not None:
        st.image(
            st.session_state[config.KEY_IMG1_RGB],
            caption="Image 1",
            use_container_width=True,
        )
    else:
        st.info("Upload Image 1 using the sidebar.")

with col_main_right:
    st.subheader("Original Image 2")
    if st.session_state[config.KEY_IMG2_RGB] is not None:
        st.image(
            st.session_state[config.KEY_IMG2_RGB],
            caption="Image 2",
            use_container_width=True,
        )
    else:
        st.info("Upload Image 2 using the sidebar.")

st.markdown("---")


# --- Processing Logic (Triggered by Button) ---
if compute_button_pressed:
    if (
        st.session_state[config.KEY_IMG1_GRAY] is None
        or st.session_state[config.KEY_IMG2_GRAY] is None
    ):
        st.sidebar.warning("Please upload both images before computing.")
    else:
        current_detector_name = st.session_state[config.KEY_DETECTOR_NAME]
        current_matcher_name = st.session_state[config.KEY_MATCHER_NAME]

        # 1. Initialize Detector
        # `get_detector_model_or_instance` is cached, so it's efficient.
        # Params are passed for OpenCV, Kornia models might have params applied during extraction.
        detector_model_or_inst = get_detector_model_or_instance(
            current_detector_name, detector_params
        )

        if detector_model_or_inst is None:
            st.error(
                f"Failed to initialize detector: {current_detector_name}. Cannot proceed."
            )
        else:
            # 2. Feature Detection
            with st.spinner(f"Detecting features using {current_detector_name}..."):
                kp1, des1 = extract_features(
                    detector_model_or_inst,
                    st.session_state[config.KEY_IMG1_GRAY],
                    current_detector_name,
                    detector_params,
                )
                kp2, des2 = extract_features(
                    detector_model_or_inst,
                    st.session_state[config.KEY_IMG2_GRAY],
                    current_detector_name,
                    detector_params,
                )
                st.session_state[config.KEY_KP1], st.session_state[config.KEY_DES1] = (
                    kp1,
                    des1,
                )
                st.session_state[config.KEY_KP2], st.session_state[config.KEY_DES2] = (
                    kp2,
                    des2,
                )

            if not kp1 or not kp2:
                st.warning("Could not detect keypoints in one or both images.")
            if des1 is None or des2 is None:
                st.warning(
                    f"Could not compute descriptors for one or both images using {current_detector_name}."
                )
            else:
                st.success(
                    f"Detected {len(kp1)} keypoints in Image 1 and {len(kp2)} keypoints in Image 2 "
                    f"using {current_detector_name}."
                )

                # 3. Initialize Matcher (if Kornia-based, or get name for OpenCV)
                # `detector_name` is passed for LightGlue to select its internal feature type.
                matcher_model_or_name = get_matcher_model_or_instance(
                    current_matcher_name, current_detector_name
                )

                if (
                    matcher_model_or_name is None
                    and current_matcher_name != config.MATCHER_BF
                ):
                    st.error(
                        f"Failed to initialize matcher: {current_matcher_name}. Cannot proceed with matching."
                    )
                else:
                    # 4. Feature Matching
                    with st.spinner(
                        f"Matching features using {current_matcher_name}..."
                    ):
                        matches = perform_matching(
                            matcher_model_or_name,  # Kornia model or MATCHER_BF string
                            st.session_state[config.KEY_DES1],
                            st.session_state[config.KEY_KP1],
                            st.session_state[config.KEY_IMG1_GRAY],
                            st.session_state[config.KEY_DES2],
                            st.session_state[config.KEY_KP2],
                            st.session_state[config.KEY_IMG2_GRAY],
                            current_matcher_name,  # The string name for dispatching
                            matcher_params,
                            current_detector_name,  # Detector name for norm/compatibility
                        )
                        st.session_state[config.KEY_MATCHES] = matches

                    if matches:
                        st.success(
                            f"Found {len(matches)} matches using {current_matcher_name}."
                        )

                        # 5. Store result for comparison
                        # Unique name for this configuration
                        comp_key_name = (
                            f"{current_detector_name} + {current_matcher_name} "
                            f"(KPs: {len(kp1)}|{len(kp2)}, Matches: {len(matches)})"
                        )
                        # Use a more robust way to get image data for drawing if needed
                        img1_rgb_for_draw = st.session_state.get(config.KEY_IMG1_RGB)
                        img2_rgb_for_draw = st.session_state.get(config.KEY_IMG2_RGB)

                        if (
                            img1_rgb_for_draw is not None
                            and img2_rgb_for_draw is not None
                        ):
                            comparison_img = draw_matches_comparison_image(
                                img1_rgb_for_draw,
                                kp1,
                                img2_rgb_for_draw,
                                kp2,
                                matches,
                                max_matches_to_draw=100,  # Draw top 100 for stored image
                            )
                            if comparison_img is not None:
                                st.session_state[config.KEY_COMPARISON_RESULTS][
                                    comp_key_name
                                ] = {
                                    "img_matches_rgb": comparison_img,
                                    "num_kp1": len(kp1),
                                    "num_kp2": len(kp2),
                                    "num_matches": len(matches),
                                    "detector": current_detector_name,
                                    "matcher": current_matcher_name,
                                }
                                st.toast(
                                    f"Result '{comp_key_name}' added to comparison list!",
                                    icon="✅",
                                )
                            else:
                                st.warning("Could not generate comparison match image.")
                        else:
                            st.warning(
                                "RGB images not available for drawing comparison match image."
                            )
                    else:
                        st.warning(f"No matches found with {current_matcher_name}.")


# --- UI: Current Run Visualization ---
st.subheader("📊 Current Run Visualization")

# Keypoint display on original images
show_keypoints_opt = st.checkbox(
    "Show keypoints on original images", value=False, key="cb_show_kps_curr"
)
if show_keypoints_opt:
    col_kp1_viz, col_kp2_viz = st.columns(2)
    with col_kp1_viz:
        if (
            st.session_state[config.KEY_IMG1_GRAY] is not None
            and st.session_state[config.KEY_KP1]
        ):
            img1_kp_display = cv2.drawKeypoints(
                st.session_state[config.KEY_IMG1_GRAY],
                st.session_state[config.KEY_KP1],
                None,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            )
            st.image(
                img1_kp_display,
                caption=f"Image 1 Keypoints ({len(st.session_state[config.KEY_KP1])})",
                use_container_width=True,
            )
        elif st.session_state[config.KEY_IMG1_GRAY] is not None:
            st.caption("No keypoints available or computed for Image 1 in current run.")
    with col_kp2_viz:
        if (
            st.session_state[config.KEY_IMG2_GRAY] is not None
            and st.session_state[config.KEY_KP2]
        ):
            img2_kp_display = cv2.drawKeypoints(
                st.session_state[config.KEY_IMG2_GRAY],
                st.session_state[config.KEY_KP2],
                None,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            )
            st.image(
                img2_kp_display,
                caption=f"Image 2 Keypoints ({len(st.session_state[config.KEY_KP2])})",
                use_container_width=True,
            )
        elif st.session_state[config.KEY_IMG2_GRAY] is not None:
            st.caption("No keypoints available or computed for Image 2 in current run.")


# Match lines display for the current run
st.subheader("🔗 Current Matches Visualization")
current_matches = st.session_state.get(config.KEY_MATCHES, [])
if (
    current_matches
    and st.session_state[config.KEY_IMG1_RGB] is not None
    and st.session_state[config.KEY_IMG2_RGB] is not None
    and st.session_state[config.KEY_KP1]
    and st.session_state[config.KEY_KP2]
):
    num_total_matches = len(current_matches)
    slider_max = min(config.MAX_MATCHES_TO_DRAW_SLIDER, num_total_matches)
    slider_default = min(config.DEFAULT_MATCHES_TO_DRAW_SLIDER, num_total_matches)

    if (
        slider_max < config.MIN_MATCHES_TO_DRAW_SLIDER
    ):  # if fewer matches than min slider val
        num_to_draw_curr = num_total_matches
        if num_total_matches > 0:
            st.caption(f"Displaying all {num_total_matches} matches.")
        # else: # No matches, slider won't be shown anyway
    elif num_total_matches > 0:
        num_to_draw_curr = st.slider(
            "Number of matches to display (by quality)",
            min_value=config.MIN_MATCHES_TO_DRAW_SLIDER,
            max_value=slider_max,
            value=slider_default,
            key="slider_num_matches_curr",
            help="Adjusts how many of the best matches are drawn. For some matchers like LightGlue, 'distance' is 1-confidence.",
        )
    else:  # No matches, but other conditions were met (should not happen if current_matches is empty)
        num_to_draw_curr = 0

    if num_to_draw_curr > 0:
        # `draw_matches_comparison_image` already sorts by distance and takes top N
        img_matches_display_curr = draw_matches_comparison_image(
            st.session_state[config.KEY_IMG1_RGB],
            st.session_state[config.KEY_KP1],
            st.session_state[config.KEY_IMG2_RGB],
            st.session_state[config.KEY_KP2],
            current_matches,  # Pass all current matches
            max_matches_to_draw=num_to_draw_curr,
        )
        if img_matches_display_curr is not None:
            st.image(
                img_matches_display_curr,
                caption=f"Current Matches ({num_to_draw_curr} of {num_total_matches} shown)",
                use_container_width=True,
            )
        else:
            st.warning("Could not generate image for current matches.")
    elif (
        num_total_matches > 0 and num_to_draw_curr == 0
    ):  # User set slider to 0, or an edge case.
        st.info("Set the slider above to a value greater than 0 to display matches.")
    else:  # No matches to draw
        st.info(
            "No matches were found in the current run, or not enough data to display."
        )

elif compute_button_pressed:  # Button was pressed, but conditions for display not met
    st.info(
        "Waiting for computation to complete or ensure valid inputs for match visualization."
    )
else:  # Default state before any computation or if images/kps are missing
    st.info("Compute matches to see the visualization here.")


st.markdown("---")

# --- UI: Comparison Section ---
st.header("🔍 Method Comparison Area")
comparison_data = st.session_state.get(config.KEY_COMPARISON_RESULTS, {})

if not comparison_data:
    st.info("Results added via 'Compute & Add to Comparison' will appear here.")
else:
    # Sort by key name for now (usually reflects addition order if keys are timestamped or unique)
    # A more robust sorting would be to store a timestamp with each result.
    sorted_comp_keys = sorted(comparison_data.keys(), reverse=True)

    if not sorted_comp_keys:  # Should not happen if comparison_data is not empty
        st.info("No comparison results available.")
    else:
        selected_comp_key = st.selectbox(
            "Select a stored result to view:",
            options=sorted_comp_keys,
            # format_func can be used for more descriptive labels if keys are cryptic
            # format_func=lambda k: f"{comparison_data[k]['detector']} + {comparison_data[k]['matcher']} ({comparison_data[k]['num_matches']} matches)"
        )
        if selected_comp_key and selected_comp_key in comparison_data:
            result_to_display = comparison_data[selected_comp_key]
            st.image(
                result_to_display["img_matches_rgb"],
                caption=f"Comparison: {selected_comp_key}",
                use_container_width=True,
            )
            stats_md = (
                f"**Stats**: Img1 KPs: `{result_to_display['num_kp1']}` | "
                f"Img2 KPs: `{result_to_display['num_kp2']}` | "
                f"Matches: `{result_to_display['num_matches']}`"
            )
            method_md = (
                f"**Method**: Detector: `{result_to_display['detector']}` | "
                f"Matcher: `{result_to_display['matcher']}`"
            )
            st.markdown(stats_md)
            st.markdown(method_md)
        else:
            st.warning(
                "Selected comparison result not found. Please try again or clear results."
            )

    if st.button("Clear All Comparison Results", key="btn_clear_comparison"):
        st.session_state[config.KEY_COMPARISON_RESULTS] = {}
        st.toast("All comparison results cleared!", icon="🗑️")
        st.rerun()  # Rerun to reflect the cleared state immediately
