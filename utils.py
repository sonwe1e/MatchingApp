"""
Utility functions for the Streamlit application.
"""

import streamlit as st
from PIL import Image
import numpy as np
import cv2
import torch
from config import DEVICE, DEVICE_NAME  # Import from config
import config


@st.cache_resource
def get_device_info():
    """
    Returns the configured PyTorch device and its name.
    This function is cached for efficiency.

    Why this function?
    - Centralizes device determination.
    - Uses st.cache_resource for efficient re-use across app reruns.

    Returns:
        tuple: (torch.device, str) - The PyTorch device and its uppercase name.
    """
    # This print is for debugging during development,
    # will appear in the console where Streamlit runs.
    print(f"PyTorch determined device: {DEVICE_NAME}")
    return DEVICE, DEVICE_NAME


def load_image_from_upload(image_file):
    """
    Loads an image from a Streamlit file uploader object.

    What it does:
    - Opens the image using PIL.
    - Converts to RGB if it's not already in RGB or L (Grayscale) mode.
    - Converts the PIL Image to a NumPy array.

    Why this function?
    - Encapsulates image loading logic.
    - Handles basic mode conversion to ensure consistency.

    Args:
        image_file (UploadedFile or None): The file object from st.file_uploader.

    Returns:
        np.ndarray or None: The loaded image as a NumPy array (RGB), or None if no file.
    """
    if image_file is not None:
        try:
            img_pil = Image.open(image_file)
            # Ensure the image is in a format that OpenCV can easily handle (RGB or Grayscale)
            # Converting to RGB is safer for general use, grayscale conversion can happen later.
            if img_pil.mode not in ["RGB", "L"]:
                img_pil = img_pil.convert("RGB")
            return np.array(img_pil)
        except Exception as e:
            st.error(f"Error loading image '{image_file.name}': {e}")
            return None
    return None


def convert_to_opencv_gray(image_rgb_numpy):
    """
    Converts an RGB or RGBA NumPy image array to grayscale for OpenCV processing.

    What it does:
    - Checks if the input image is valid.
    - Converts 3-channel RGB or 4-channel RGBA to single-channel Grayscale.
    - Returns the image unmodified if it's already grayscale or has an unexpected format.

    Why this function?
    - Standardizes grayscale conversion.
    - Handles common color formats (RGB, RGBA).

    Args:
        image_rgb_numpy (np.ndarray or None): The input image as a NumPy array.

    Returns:
        np.ndarray or None: The grayscale image as a NumPy array, or None if input was None.
    """
    if image_rgb_numpy is None:
        return None
    if len(image_rgb_numpy.shape) == 3:
        if image_rgb_numpy.shape[2] == 3:  # RGB
            return cv2.cvtColor(image_rgb_numpy, cv2.COLOR_RGB2GRAY)
        elif image_rgb_numpy.shape[2] == 4:  # RGBA
            return cv2.cvtColor(image_rgb_numpy, cv2.COLOR_RGBA2GRAY)
    elif len(image_rgb_numpy.shape) == 2:  # Already Grayscale
        return image_rgb_numpy

    # If the image is neither RGB, RGBA, nor Grayscale, return as is or raise error
    # For now, returning as is, but logging a warning might be good.
    st.warning(
        f"Image format not explicitly handled for grayscale conversion: shape {image_rgb_numpy.shape}"
    )
    return image_rgb_numpy


def draw_matches_comparison_image(
    img1_rgb, kp1, img2_rgb, kp2, matches, max_matches_to_draw=100
):
    """
    Draws matches between two images for comparison display.

    What it does:
    - Converts RGB images to BGR for OpenCV drawing.
    - Sorts matches by distance (quality) and takes the top N.
    - Uses cv2.drawMatches to create the visualization.
    - Converts the result back to RGB.

    Why this function?
    - Encapsulates the logic for creating a standard match visualization.
    - Ensures color space conversions are handled correctly.

    Args:
        img1_rgb (np.ndarray): First image in RGB format.
        kp1 (list of cv2.KeyPoint): Keypoints for the first image.
        img2_rgb (np.ndarray): Second image in RGB format.
        kp2 (list of cv2.KeyPoint): Keypoints for the second image.
        matches (list of cv2.DMatch): List of matches.
        max_matches_to_draw (int): Maximum number of best matches to draw.

    Returns:
        np.ndarray or None: The image with matches drawn (RGB), or None if inputs are invalid.
    """
    if img1_rgb is None or img2_rgb is None or not kp1 or not kp2 or not matches:
        st.warning(
            "Cannot draw matches: missing one or more inputs (images, keypoints, or matches)."
        )
        return None

    try:
        img1_bgr = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2BGR)
        img2_bgr = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2BGR)

        # Sort matches by distance (lower is better) and select top N
        matches_to_draw = sorted(matches, key=lambda x: x.distance)[
            :max_matches_to_draw
        ]

        img_matches_bgr = cv2.drawMatches(
            img1_bgr,
            kp1,
            img2_bgr,
            kp2,
            matches_to_draw,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        return cv2.cvtColor(img_matches_bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        st.error(f"Error drawing matches: {e}")
        return None


def get_norm_for_detector(detector_name):
    """
    Returns the appropriate OpenCV norm type for a given detector.

    What it does:
    - Uses a predefined map (config.DETECTOR_NORM_MAP) to find the norm.
    - Defaults to NORM_L2 if the detector is not in the map.

    Why this function?
    - Centralizes the logic for selecting norm types.
    - Makes it easy to update or add new detector-norm mappings.

    Args:
        detector_name (str): The name of the feature detector.

    Returns:
        int: The OpenCV norm type (e.g., cv2.NORM_L2, cv2.NORM_HAMMING).
    """
    return config.DETECTOR_NORM_MAP.get(detector_name, cv2.NORM_L2)
