"""
Application Configuration File
"""

import cv2
import torch

# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_NAME = DEVICE.type.upper()

# --- Detector Configurations ---
DETECTOR_SIFT = "SIFT"
DETECTOR_ORB = "ORB"
DETECTOR_AKAZE = "AKAZE"
DETECTOR_BRISK = "BRISK"
DETECTOR_DISK_KORNIA = "DISK (Kornia)"

SUPPORTED_DETECTORS = [
    DETECTOR_SIFT,
    DETECTOR_ORB,
    DETECTOR_AKAZE,
    DETECTOR_BRISK,
    DETECTOR_DISK_KORNIA,
]

DEFAULT_SIFT_NFEATURES = 1000
DEFAULT_ORB_NFEATURES = 500
DEFAULT_ORB_SCALE_FACTOR = 1.2
DEFAULT_ORB_NLEVELS = 8
DEFAULT_BRISK_THRESH = 30
DEFAULT_BRISK_OCTAVES = 3
DEFAULT_DISK_TOP_K = 5000

# Norm types for OpenCV matchers based on detector
DETECTOR_NORM_MAP = {
    DETECTOR_SIFT: cv2.NORM_L2,
    DETECTOR_ORB: cv2.NORM_HAMMING,
    DETECTOR_AKAZE: cv2.NORM_HAMMING,
    DETECTOR_BRISK: cv2.NORM_HAMMING,
    DETECTOR_DISK_KORNIA: cv2.NORM_L2,  # DISK descriptors are typically float
}

# --- Matcher Configurations ---
MATCHER_BF = "BFMatcher (Brute-Force)"
MATCHER_LIGHTGLUE_KORNIA = "LightGlue (Kornia)"

SUPPORTED_MATCHERS = [
    MATCHER_BF,
    MATCHER_LIGHTGLUE_KORNIA,
]

DEFAULT_RATIO_TEST_THRESH = 0.75

# --- Session State Keys ---
# It's good practice to define session state keys as constants
# to avoid typos and for easier refactoring.
KEY_IMG1_RGB = "img1_rgb"
KEY_IMG2_RGB = "img2_rgb"
KEY_IMG1_GRAY = "img1_gray"
KEY_IMG2_GRAY = "img2_gray"
KEY_KP1 = "kp1"
KEY_DES1 = "des1"
KEY_KP2 = "kp2"
KEY_DES2 = "des2"
KEY_MATCHES = "matches"
KEY_COMPARISON_RESULTS = "comparison_results"
KEY_DETECTOR_NAME = "detector_name"
KEY_MATCHER_NAME = "matcher_name"

# --- UI Defaults ---
MAX_MATCHES_TO_DRAW_SLIDER = 2000
DEFAULT_MATCHES_TO_DRAW_SLIDER = 50
MIN_MATCHES_TO_DRAW_SLIDER = 1
