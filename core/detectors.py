"""
Handles initialization of feature detectors (OpenCV & Kornia) and feature extraction.
"""

import streamlit as st
import cv2
import numpy as np
import torch

from config import (
    DEVICE,
    DETECTOR_SIFT,
    DETECTOR_ORB,
    DETECTOR_AKAZE,
    DETECTOR_BRISK,
    DETECTOR_DISK_KORNIA,
    DEFAULT_SIFT_NFEATURES,
    DEFAULT_ORB_NFEATURES,
    DEFAULT_ORB_SCALE_FACTOR,
    DEFAULT_ORB_NLEVELS,
    DEFAULT_BRISK_THRESH,
    DEFAULT_BRISK_OCTAVES,
    DEFAULT_DISK_TOP_K,
)
from .kornia_models import get_kornia_disk_model, extract_disk_features


# Caching for OpenCV detector instances
@st.cache_resource(show_spinner="Initializing OpenCV detector...")
def get_opencv_detector_instance(detector_name, params):
    """
    Creates and returns an instance of an OpenCV feature detector.

    What it does:
    - Takes detector name and parameters.
    - Initializes the corresponding cv2 detector (SIFT, ORB, AKAZE, BRISK).
    - Caches the detector instance using st.cache_resource for efficiency.

    Why this function?
    - Centralizes OpenCV detector creation.
    - Avoids re-initializing detectors on every Streamlit rerun if params don't change.

    Args:
        detector_name (str): Name of the detector (e.g., "SIFT", "ORB").
        params (dict): Dictionary of parameters for the detector.

    Returns:
        cv2.Feature2D or None: The initialized OpenCV detector object, or None if name is invalid.
    """
    if detector_name == DETECTOR_SIFT:
        return cv2.SIFT_create(
            nfeatures=params.get("sift_nfeatures", DEFAULT_SIFT_NFEATURES)
        )
    elif detector_name == DETECTOR_ORB:
        return cv2.ORB_create(
            nfeatures=params.get("orb_nfeatures", DEFAULT_ORB_NFEATURES),
            scaleFactor=params.get("orb_scaleFactor", DEFAULT_ORB_SCALE_FACTOR),
            nlevels=params.get("orb_nlevels", DEFAULT_ORB_NLEVELS),
        )
    elif detector_name == DETECTOR_AKAZE:
        # AKAZE has fewer common parameters to tune via simple sliders in this app
        return cv2.AKAZE_create()
    elif detector_name == DETECTOR_BRISK:
        return cv2.BRISK_create(
            thresh=params.get("brisk_thresh", DEFAULT_BRISK_THRESH),
            octaves=params.get("brisk_octaves", DEFAULT_BRISK_OCTAVES),
        )
    st.warning(f"Unknown OpenCV detector name: {detector_name}")
    return None


def get_detector_model_or_instance(detector_name, params):
    """
    Unified function to get either a Kornia model or an OpenCV detector instance.

    What it does:
    - Checks if the detector is Kornia-based.
    - If Kornia, calls the appropriate model loading function (e.g., get_kornia_disk_model).
    - If OpenCV, calls get_opencv_detector_instance.

    Why this function?
    - Provides a single entry point for obtaining any supported detector.
    - Simplifies the main application logic.

    Args:
        detector_name (str): The name of the detector.
        params (dict): Parameters for the detector.

    Returns:
        (kornia.feature.DISK, cv2.Feature2D, or None): The loaded model or instance.
    """
    if detector_name == DETECTOR_DISK_KORNIA:
        return (
            get_kornia_disk_model()
        )  # DISK params like top_k are handled in extract_disk_features
    elif detector_name in [DETECTOR_SIFT, DETECTOR_ORB, DETECTOR_AKAZE, DETECTOR_BRISK]:
        return get_opencv_detector_instance(detector_name, params)
    else:
        st.error(f"Unsupported detector: {detector_name}")
        return None


def extract_features(detector_object, image_gray_np, detector_name, params):
    """
    Detects keypoints and computes descriptors for a given image.

    What it does:
    - Handles both Kornia (DISK) and OpenCV detectors.
    - For DISK:
        - Calls the specialized `extract_disk_features` function.
        - Passes `disk_top_k` from `params`.
    - For OpenCV detectors:
        - Uses the `detectAndCompute` method of the detector instance.
    - Performs error handling and returns empty lists/None on failure.

    Why this function?
    - Acts as a dispatcher for feature extraction based on detector type.
    - Centralizes the core `detectAndCompute` logic.
    - Includes robustness checks for inputs and outputs.

    Args:
        detector_object (object): The initialized Kornia model or OpenCV detector instance.
        image_gray_np (np.ndarray): The grayscale input image.
        detector_name (str): The name of the detector.
        params (dict): Parameters, potentially including `disk_top_k` for DISK.

    Returns:
        tuple: (list of cv2.KeyPoint, np.ndarray or None)
               - List of OpenCV KeyPoint objects.
               - NumPy array of descriptors, or None if not computed/available.
               Returns ([], None) if image_gray_np is None or detector_object is None.
    """
    if image_gray_np is None:
        st.warning("Cannot extract features: Input image is None.")
        return [], None
    if detector_object is None:
        st.error(
            f"Cannot extract features: Detector '{detector_name}' is not initialized."
        )
        return [], None

    try:
        if detector_name == DETECTOR_DISK_KORNIA:
            # The DISK model itself is detector_object. Params for top_k are passed here.
            disk_top_k = params.get("disk_top_k", DEFAULT_DISK_TOP_K)
            return extract_disk_features(
                detector_object, image_gray_np, top_k=disk_top_k
            )
        elif detector_name in [
            DETECTOR_SIFT,
            DETECTOR_ORB,
            DETECTOR_AKAZE,
            DETECTOR_BRISK,
        ]:
            # OpenCV detectors
            return detector_object.detectAndCompute(image_gray_np, None)
        else:
            st.error(
                f"Feature extraction not implemented for detector: {detector_name}"
            )
            return [], None
    except cv2.error as cv_err:  # Catch specific OpenCV errors
        st.error(
            f"OpenCV error during feature extraction with {detector_name}: {cv_err}"
        )
        return [], None
    except Exception as e:
        st.error(f"Generic error during feature extraction with {detector_name}: {e}")
        import traceback

        st.error(f"Traceback: {traceback.format_exc()}")
        return [], None
