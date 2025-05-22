"""
Handles initialization of feature matchers (OpenCV BFMatcher & Kornia LightGlue) and matching logic.
"""

import streamlit as st
import cv2
import numpy as np
import torch

from config import (
    DEVICE,
    MATCHER_BF,
    MATCHER_LIGHTGLUE_KORNIA,
    DEFAULT_RATIO_TEST_THRESH,
    DETECTOR_DISK_KORNIA,  # For LightGlue compatibility check
)
from utils import get_norm_for_detector
from .kornia_models import get_kornia_lightglue_model, match_with_lightglue


@st.cache_resource(show_spinner="Initializing OpenCV BFMatcher...")
def get_opencv_bfmatcher_instance(norm_type, cross_check):
    """
    Creates and returns an instance of an OpenCV Brute-Force Matcher.

    What it does:
    - Initializes cv2.BFMatcher with a given norm type and crossCheck flag.
    - Caches the instance using st.cache_resource.

    Why this function?
    - Centralizes BFMatcher creation.
    - Caches for efficiency.

    Args:
        norm_type (int): The norm type (e.g., cv2.NORM_L2, cv2.NORM_HAMMING).
        cross_check (bool): Whether to use cross-checking.

    Returns:
        cv2.BFMatcher: The initialized Brute-Force matcher.
    """
    try:
        return cv2.BFMatcher(normType=norm_type, crossCheck=cross_check)
    except Exception as e:
        st.error(f"Failed to initialize OpenCV BFMatcher: {e}")
        return None


def get_matcher_model_or_instance(matcher_name, detector_name_for_glue_compat=None):
    """
    Unified function to get either a Kornia LightGlue model or an OpenCV BFMatcher instance.
    Note: BFMatcher instance creation depends on norm_type and cross_check,
          so it's often better to create it just before matching. This function
          is more for Kornia models that are loaded once.

    What it does:
    - If LightGlue, calls `get_kornia_lightglue_model`.
    - For BFMatcher, this function isn't strictly necessary as its parameters
      (norm_type, cross_check) depend on runtime choices.
      However, it can return the name for dispatching logic.

    Why this function?
    - Consistent interface for obtaining Kornia matcher models.

    Args:
        matcher_name (str): The name of the matcher.
        detector_name_for_glue_compat (str, optional): The detector name, used by LightGlue
                                                      to determine its feature type.

    Returns:
        (kornia.feature.LightGlue, str, or None): The loaded LightGlue model,
                                                   or the name of the BFMatcher, or None.
    """
    if matcher_name == MATCHER_LIGHTGLUE_KORNIA:
        if not detector_name_for_glue_compat:
            st.warning(
                "Detector name not provided for LightGlue compatibility check. This might lead to issues."
            )
        return get_kornia_lightglue_model(detector_name_for_glue_compat)
    elif matcher_name == MATCHER_BF:
        return MATCHER_BF  # Return name, instance created later
    else:
        st.error(f"Unsupported matcher: {matcher_name}")
        return None


def perform_matching(
    matcher_object_or_name,
    des1,
    kp1_cv,
    img1_gray_np,  # kp1_cv, img1_gray_np needed for LightGlue
    des2,
    kp2_cv,
    img2_gray_np,  # kp2_cv, img2_gray_np needed for LightGlue
    matcher_name_str,  # Explicit matcher name string for dispatch
    matcher_params,
    detector_name_str,  # Needed for BFMatcher norm and LightGlue compatibility
):
    """
    Performs feature matching using the specified method.

    What it does:
    - Handles Kornia LightGlue and OpenCV BFMatcher.
    - For LightGlue:
        - Calls `match_with_lightglue`.
        - Warns if used with non-Kornia detectors.
    - For BFMatcher:
        - Converts descriptors to NumPy if they are Tensors.
        - Determines norm type based on the detector.
        - Applies ratio test if enabled.
        - Sorts matches by distance.
    - Performs error handling.

    Why this function?
    - Central dispatcher for all matching operations.
    - Encapsulates logic for different matchers and their parameters.

    Args:
        matcher_object_or_name (object or str): Kornia LightGlue model instance or MATCHER_BF string.
        des1 (np.ndarray or torch.Tensor): Descriptors for the first image.
        kp1_cv (list of cv2.KeyPoint): Keypoints for the first image (for LightGlue).
        img1_gray_np (np.ndarray): First grayscale image (for LightGlue).
        des2 (np.ndarray or torch.Tensor): Descriptors for the second image.
        kp2_cv (list of cv2.KeyPoint): Keypoints for the second image (for LightGlue).
        img2_gray_np (np.ndarray): Second grayscale image (for LightGlue).
        matcher_name_str (str): The string name of the matcher (e.g., config.MATCHER_LIGHTGLUE_KORNIA).
        matcher_params (dict): Parameters for the matcher (e.g., ratio_thresh, use_ratio_test).
        detector_name_str (str): The string name of the detector used.

    Returns:
        list of cv2.DMatch: List of found matches.
    """
    if des1 is None or des2 is None:
        st.warning(
            "Cannot perform matching: Descriptors are missing for one or both images."
        )
        return []

    # Ensure descriptors have content
    des1_empty = (isinstance(des1, np.ndarray) and des1.size == 0) or (
        hasattr(des1, "numel") and des1.numel() == 0
    )
    des2_empty = (isinstance(des2, np.ndarray) and des2.size == 0) or (
        hasattr(des2, "numel") and des2.numel() == 0
    )

    if des1_empty or des2_empty:
        st.warning(
            "Cannot perform matching: Descriptors are empty for one or both images."
        )
        return []

    matches = []
    try:
        if matcher_name_str == MATCHER_LIGHTGLUE_KORNIA:
            lightglue_model = (
                matcher_object_or_name  # This should be the model instance
            )
            if not detector_name_str.endswith("(Kornia)"):  # A simple check
                st.warning(
                    "LightGlue is designed for Kornia features (e.g., DISK). "
                    "Using it with non-Kornia detector features (e.g., SIFT) "
                    "may lead to poor results or errors if descriptor formats are incompatible."
                )
            # match_with_lightglue expects des1, des2 to be torch tensors or numpy arrays.
            # It handles conversion internally if they are numpy.
            # It also needs keypoints (cv2.KeyPoint lists) and images (NumPy arrays).
            matches = match_with_lightglue(
                lightglue_model, kp1_cv, des1, img1_gray_np, kp2_cv, des2, img2_gray_np
            )

        elif matcher_name_str == MATCHER_BF:
            # Convert descriptors to NumPy if they are PyTorch Tensors
            des1_np = des1.cpu().numpy() if isinstance(des1, torch.Tensor) else des1
            des2_np = des2.cpu().numpy() if isinstance(des2, torch.Tensor) else des2

            if (
                des1_np is None
                or des2_np is None
                or des1_np.size == 0
                or des2_np.size == 0
            ):
                st.warning(
                    "Descriptors are empty after conversion; cannot perform BFMatcher."
                )
                return []

            # Ensure descriptors are of a type BFMatcher can handle (e.g. float32 for SIFT, uint8 for ORB)
            # This might need more sophisticated type checking/conversion based on detector.
            # For now, assuming extract_features returns compatible types.
            if des1_np.dtype != des2_np.dtype:
                st.warning(
                    f"Descriptor types mismatch: {des1_np.dtype} vs {des2_np.dtype}. Trying to proceed."
                )
                # Potentially convert one to match the other, or ensure extract_features is consistent.
                # e.g., des2_np = des2_np.astype(des1_np.dtype) - this is risky without knowing semantics.

            current_norm_type = get_norm_for_detector(detector_name_str)
            use_ratio_test = matcher_params.get("use_ratio_test", False)
            # Cross-check is typically True if ratio test is False.
            cross_check_bf = (
                matcher_params.get("cross_check_bf", True)
                if not use_ratio_test
                else False
            )

            bf_matcher = get_opencv_bfmatcher_instance(
                current_norm_type, cross_check_bf
            )
            if bf_matcher is None:
                return []

            if use_ratio_test:
                ratio_thresh_val = matcher_params.get(
                    "ratio_thresh", DEFAULT_RATIO_TEST_THRESH
                )
                # knnMatch expects descriptors to be float32 for NORM_L2, NORM_L1.
                # For NORM_HAMMING, it expects CV_8U.
                # SIFT/DISK are float32. ORB/BRISK/AKAZE are uint8.
                if current_norm_type in [cv2.NORM_L2, cv2.NORM_L1]:
                    if des1_np.dtype != np.float32:
                        des1_np = des1_np.astype(np.float32)
                    if des2_np.dtype != np.float32:
                        des2_np = des2_np.astype(np.float32)
                elif current_norm_type in [cv2.NORM_HAMMING, cv2.NORM_HAMMING2]:
                    if des1_np.dtype != np.uint8:
                        st.warning(
                            f"Converting des1 from {des1_np.dtype} to uint8 for HAMMING norm."
                        )
                        des1_np = np.clip(des1_np, 0, 255).astype(
                            np.uint8
                        )  # Example, might not be correct conversion
                    if des2_np.dtype != np.uint8:
                        st.warning(
                            f"Converting des2 from {des2_np.dtype} to uint8 for HAMMING norm."
                        )
                        des2_np = np.clip(des2_np, 0, 255).astype(np.uint8)

                raw_matches_knn = bf_matcher.knnMatch(des1_np, des2_np, k=2)
                good_matches_bf = []
                if raw_matches_knn:
                    for m_pair in raw_matches_knn:
                        if len(m_pair) == 2:  # Ensure two neighbors were found
                            m, n = m_pair
                            if m.distance < ratio_thresh_val * n.distance:
                                good_matches_bf.append(m)
                matches = good_matches_bf
            else:  # Not using ratio test (implies crossCheck might be True)
                matches = bf_matcher.match(des1_np, des2_np)
                if matches:
                    matches = sorted(matches, key=lambda x: x.distance)
        else:
            st.error(f"Matching logic not implemented for matcher: {matcher_name_str}")

    except cv2.error as cv_err:
        st.error(f"OpenCV error during matching with {matcher_name_str}: {cv_err}")
        if "query descriptor type" in str(cv_err) or "train descriptor type" in str(
            cv_err
        ):
            st.error(
                "This often indicates a mismatch between descriptor data type and the norm type used by the matcher (e.g. using NORM_HAMMING with float descriptors)."
            )
            st.error(
                f"Detector: {detector_name_str}, Des1 type: {des1.dtype if hasattr(des1, 'dtype') else 'N/A'}, Des2 type: {des2.dtype if hasattr(des2, 'dtype') else 'N/A'}"
            )
        matches = []
    except Exception as e:
        st.error(f"Generic error during matching with {matcher_name_str}: {e}")
        import traceback

        st.error(f"Traceback: {traceback.format_exc()}")
        matches = []

    return matches
