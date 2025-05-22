"""
Handles loading and interfacing with Kornia models (DISK, LightGlue).
"""

import streamlit as st
import torch
import kornia.feature as KF
import kornia.utils as KU
import kornia.geometry as KG
import torch.nn.functional as F
import numpy as np
import cv2  # For cv2.KeyPoint

from config import DEVICE, DETECTOR_DISK_KORNIA, MATCHER_LIGHTGLUE_KORNIA


@st.cache_resource(show_spinner="Loading Kornia DISK model...")
def get_kornia_disk_model():
    """
    Loads the pre-trained Kornia DISK model.

    What it does:
    - Attempts to load DISK.from_pretrained("depth").
    - Moves the model to the configured DEVICE.
    - Caches the model using st.cache_resource.

    Why this function?
    - Encapsulates model loading logic for DISK.
    - Handles potential errors during loading.
    - Efficiently caches the model to avoid reloading on every script rerun.

    Returns:
        KF.DISK or None: The loaded DISK model, or None if loading fails.
    """
    try:
        model = KF.DISK.from_pretrained("depth").to(DEVICE)
        model.eval()  # Set to evaluation mode
        return model
    except Exception as e:
        st.error(
            f"Failed to load Kornia DISK model: {e}. "
            "Please check your internet connection and Kornia/PyTorch installation."
        )
        return None


@st.cache_resource(show_spinner="Loading Kornia LightGlue model...")
def get_kornia_lightglue_model(detector_name_for_glue="DISK (Kornia)"):
    """
    Loads the Kornia LightGlue model.

    What it does:
    - Determines the feature type for LightGlue based on the detector used (e.g., "disk").
    - Loads KF.LightGlue with the specified features.
    - Moves the model to the configured DEVICE and sets it to evaluation mode.
    - Caches the model using st.cache_resource.

    Why this function?
    - Encapsulates model loading logic for LightGlue.
    - Handles dynamic feature type selection for LightGlue.
    - Efficiently caches the model.

    Args:
        detector_name_for_glue (str): The name of the detector whose features LightGlue will process.
                                      Used to set the 'features' parameter of LightGlue.

    Returns:
        KF.LightGlue or None: The loaded LightGlue model, or None if loading fails.
    """
    try:
        # Simplified: LightGlue's 'features' param often takes 'disk', 'superpoint', etc.
        # We derive this from the detector name.
        if DETECTOR_DISK_KORNIA in detector_name_for_glue:
            lg_feat_type = "disk"
        # Add other mappings if you support more Kornia detectors like SuperPoint
        # elif "SuperPoint" in detector_name_for_glue:
        # lg_feat_type = "superpoint"
        else:
            # Default or raise an error if the detector isn't compatible
            st.warning(
                f"LightGlue is being initialized for an unrecognized Kornia detector type '{detector_name_for_glue}'. "
                "Defaulting to 'disk' features, but this may cause issues."
            )
            lg_feat_type = "disk"  # Or handle as an error

        model = KF.LightGlue(features=lg_feat_type).to(DEVICE).eval()
        return model
    except Exception as e:
        st.error(
            f"Failed to load Kornia LightGlue model: {e}. "
            "Please check your internet connection and Kornia/PyTorch installation."
        )
        return None


def preprocess_image_for_kornia(image_gray_np):
    """
    Prepares a grayscale NumPy image for Kornia models.
    Ensures output is (B, C, H, W) where B=1, C=1 for grayscale.
    """
    # image_to_tensor for grayscale (H,W) typically returns (1,H,W) [C,H,W]
    timg = KU.image_to_tensor(image_gray_np, keepdim=False).float().to(DEVICE) / 255.0

    if (
        len(timg.shape) == 2
    ):  # Should not happen if image_to_tensor worked as expected for gray
        timg = timg.unsqueeze(0).unsqueeze(0)  # H,W -> 1,1,H,W
    elif len(timg.shape) == 3:  # Expected: 1,H,W (C,H,W)
        if timg.shape[0] == 1:  # This is (C,H,W) with C=1
            timg = timg.unsqueeze(0)  # Add batch dim: C,H,W -> B,C,H,W (1,1,H,W)
        else:  # This case should ideally not be hit for grayscale from image_to_tensor
            # If it's (3,H,W) or something else, it's unexpected for grayscale
            st.warning(
                f"Unexpected image tensor shape after image_to_tensor: {timg.shape}. Attempting to add batch dim."
            )
            timg = timg.unsqueeze(0)  # Tentatively add batch dim.
    else:  # Already 4D or other unexpected shape
        st.warning(
            f"Image tensor shape is not 2D or 3D after image_to_tensor: {timg.shape}. Using as is."
        )
        # No unsqueezing if it's already 4D, but check if it's B,1,H,W

    # Final check for B,1,H,W - LightGlue typically expects 1 channel for grayscale feats
    if timg.shape[0] != 1:  # Batch size not 1
        st.warning(
            f"Batch size is not 1 after preprocessing: {timg.shape[0]}. LightGlue might expect B=1."
        )
    if timg.shape[1] != 1:  # Channel size not 1
        st.warning(
            f"Channel size is not 1 after preprocessing: {timg.shape[1]}. LightGlue might expect C=1 for grayscale."
        )
        # If DISK needs 3 channels, that's handled separately where DISK is called.
        # For LightGlue itself with 'disk' features, the image input is often for scale/context,
        # and might still expect grayscale (1 channel).

    return timg


def extract_disk_features(disk_model, image_gray_np, top_k=None):
    """
    Extracts features using the Kornia DISK model.

    What it does:
    - Preprocesses the image.
    - DISK model expects 3-channel input, so grayscale is repeated.
    - Resizes image to nearest multiple of 16 for DISK compatibility.
    - Runs inference with the DISK model.
    - **Handles potential absence of 'scores' or 'descriptors' from DISK output.**
    - Converts Kornia keypoints to cv2.KeyPoint objects.
    - Applies top-K filtering if `top_k` is specified and scores are available.

    Why this function?
    - Encapsulates the entire DISK feature extraction pipeline.
    - Includes necessary pre/post-processing steps specific to DISK.
    - **Crucially, adds robustness for DISK outputs that might lack scores/descriptors.**

    Args:
        disk_model (KF.DISK): The loaded Kornia DISK model.
        image_gray_np (np.ndarray): Grayscale input image.
        top_k (int, optional): Max number of keypoints to return, sorted by response.
                               If None, all detected keypoints are returned.

    Returns:
        tuple: (list of cv2.KeyPoint, np.ndarray or None)
               - List of OpenCV KeyPoint objects.
               - NumPy array of descriptors, or None if not available/extracted.
    """
    if disk_model is None or image_gray_np is None:
        return [], None

    try:
        timg = preprocess_image_for_kornia(image_gray_np)  # B, 1, H, W

        # DISK expects 3 input channels, repeat grayscale
        if timg.shape[1] == 1:
            timg = timg.repeat(1, 3, 1, 1)  # B, 3, H, W

        # Resize to nearest multiple of 16 for DISK
        _, _, h, w = timg.shape
        new_h = max(16, int(round(h / 16.0)) * 16)
        new_w = max(16, int(round(w / 16.0)) * 16)
        if h != new_h or w != new_w:
            timg = F.interpolate(
                timg, size=(new_h, new_w), mode="bilinear", align_corners=False
            )

        with torch.no_grad():
            features_out = disk_model(timg)
            kornia_feats_batch = (
                features_out[0] if isinstance(features_out, list) else features_out
            )

        kps_torch_all = kornia_feats_batch.keypoints
        des_torch_all = getattr(kornia_feats_batch, "descriptors", None)
        scores_torch_all = getattr(kornia_feats_batch, "scores", None)
        if scores_torch_all is None:
            scores_torch_all = getattr(kornia_feats_batch, "responses", None)

        if kps_torch_all is None or kps_torch_all.numel() == 0:
            st.warning("DISK did not detect any keypoints.")
            return [], None

        num_keypoints = kps_torch_all.shape[0]

        if scores_torch_all is not None and scores_torch_all.shape[0] != num_keypoints:
            st.warning(
                f"DISK: Keypoints ({num_keypoints}) and scores ({scores_torch_all.shape[0]}) count mismatch. Ignoring scores for selection."
            )
            scores_torch_all = None

        if des_torch_all is not None and des_torch_all.shape[0] != num_keypoints:
            st.warning(
                f"DISK: Keypoints ({num_keypoints}) and descriptors ({des_torch_all.shape[0]}) count mismatch. Descriptors might be unreliable."
            )

        kps_selected_torch = kps_torch_all
        des_selected_torch = des_torch_all
        scores_selected_torch = scores_torch_all

        if scores_torch_all is not None and top_k is not None and top_k < num_keypoints:
            try:
                indices = torch.argsort(scores_torch_all, descending=True)
                top_k_indices = indices[:top_k]

                kps_selected_torch = kps_torch_all[top_k_indices]
                if des_torch_all is not None:
                    des_selected_torch = des_torch_all[top_k_indices]
                scores_selected_torch = scores_torch_all[top_k_indices]
            except Exception as e_sort:
                st.warning(
                    f"Error during DISK Top-K selection: {e_sort}. Returning all features."
                )

        kps_cv = []
        kps_selected_cpu = kps_selected_torch.cpu().numpy()

        if scores_selected_torch is not None:
            scores_selected_cpu = scores_selected_torch.cpu().numpy()
        else:
            scores_selected_cpu = np.ones(kps_selected_cpu.shape[0], dtype=np.float32)

        for i in range(len(kps_selected_cpu)):
            pt = kps_selected_cpu[i]  # (x, y) coordinates
            response_val = float(scores_selected_cpu[i])
            # --- THIS IS THE CORRECTED LINE ---
            # cv2.KeyPoint(x, y, size, angle=-1, response=0, octave=0, class_id=-1)
            # We provide a default size (e.g., 10.0), default angle (-1),
            # our response_val, default octave (0), and default class_id (-1).
            # Ensure `size` is a float.
            kps_cv.append(cv2.KeyPoint(pt[0], pt[1], 10.0, -1.0, response_val, 0, -1))
            # --- END OF CORRECTION ---

        des_np_final = None
        if des_selected_torch is not None:
            des_np_final = des_selected_torch.cpu().numpy()

        return kps_cv, des_np_final

    except Exception as e:
        st.error(f"Kornia DISK feature extraction failed: {e}")
        import traceback

        st.error(f"Traceback: {traceback.format_exc()}")
        return [], None


def match_with_lightglue(
    lightglue_model,
    kps1_cv,
    des1_torch_or_np,
    img1_gray_np,
    kps2_cv,
    des2_torch_or_np,
    img2_gray_np,
):
    """
    Performs feature matching using Kornia LightGlue.

    What it does:
    - Prepares input data (keypoints, descriptors, images) for LightGlue.
      Descriptors are expected to be PyTorch tensors on the correct device.
      Keypoints (cv2.KeyPoint) are converted to PyTorch tensors.
      Images are converted to tensors.
    - Runs inference with the LightGlue model.
    - Converts LightGlue's output (indices and scores) into a list of cv2.DMatch objects.

    Why this function?
    - Encapsulates the LightGlue matching pipeline.
    - Handles data conversions between OpenCV/NumPy and PyTorch formats.

    Args:
        lightglue_model (KF.LightGlue): The loaded LightGlue model.
        kps1_cv (list of cv2.KeyPoint): Keypoints from the first image.
        des1_torch_or_np (torch.Tensor or np.ndarray): Descriptors for the first image.
        img1_gray_np (np.ndarray): First image in grayscale NumPy format.
        kps2_cv (list of cv2.KeyPoint): Keypoints from the second image.
        des2_torch_or_np (torch.Tensor or np.ndarray): Descriptors for the second image.
        img2_gray_np (np.ndarray): Second image in grayscale NumPy format.

    Returns:
        list of cv2.DMatch: List of good matches found by LightGlue.
    """
    if lightglue_model is None:
        st.error("LightGlue model is not loaded.")
        return []
    if not kps1_cv or not kps2_cv:
        st.error("Keypoint lists are empty, cannot perform LightGlue matching.")
        return []
    if des1_torch_or_np is None or des2_torch_or_np is None:
        st.error("Descriptors are missing, cannot perform LightGlue matching.")
        return []

    try:
        des1_torch = (
            torch.from_numpy(des1_torch_or_np).float()
            if isinstance(des1_torch_or_np, np.ndarray)
            else des1_torch_or_np
        )
        des2_torch = (
            torch.from_numpy(des2_torch_or_np).float()
            if isinstance(des2_torch_or_np, np.ndarray)
            else des2_torch_or_np
        )

        des1_torch = des1_torch.to(DEVICE)
        des2_torch = des2_torch.to(DEVICE)

        kps1_pts_list = [[kp.pt[0], kp.pt[1]] for kp in kps1_cv]
        kps2_pts_list = [[kp.pt[0], kp.pt[1]] for kp in kps2_cv]

        if not kps1_pts_list or not kps2_pts_list:
            st.error("Converted keypoint coordinate lists are empty.")
            return []

        kps1_torch_pts = torch.tensor(kps1_pts_list, device=DEVICE, dtype=torch.float)
        kps2_torch_pts = torch.tensor(kps2_pts_list, device=DEVICE, dtype=torch.float)

        img1_gray_torch = preprocess_image_for_kornia(img1_gray_np)
        img2_gray_torch = preprocess_image_for_kornia(img2_gray_np)

        data_for_lg = {
            "image0": img1_gray_torch,
            "image1": img2_gray_torch,
            "keypoints0": kps1_torch_pts.unsqueeze(0),
            "keypoints1": kps2_torch_pts.unsqueeze(0),
            "descriptors0": des1_torch.unsqueeze(0),
            "descriptors1": des2_torch.unsqueeze(0),
        }
        # --- DEBUGGING PRINT ---
        print("--- LightGlue Input Shapes ---")
        for key, value in data_for_lg.items():
            if isinstance(value, torch.Tensor):
                print(
                    f"{key}: {value.shape}, dtype: {value.dtype}, device: {value.device}"
                )
            elif (
                isinstance(value, list) and value and isinstance(value[0], torch.Tensor)
            ):  # list of tensors
                print(
                    f"{key}: list of {len(value)} tensors, first shape: {value[0].shape}"
                )
            else:
                print(f"{key}: type {type(value)}")
        print("-----------------------------")
        # --- END DEBUGGING PRINT ---
        with torch.no_grad():
            lg_pred = lightglue_model(data_for_lg)

        matches0_indices = lg_pred["matches0"][0].cpu().numpy()
        scores0 = lg_pred["scores0"][0].cpu().numpy()

        good_matches_lg = []
        for i in range(matches0_indices.shape[0]):
            train_idx = int(matches0_indices[i])
            if train_idx > -1:
                distance = 1.0 - scores0[i]
                good_matches_lg.append(
                    cv2.DMatch(_queryIdx=i, _trainIdx=train_idx, _distance=distance)
                )
        return good_matches_lg

    except Exception as e:
        st.error(f"LightGlue matching failed: {e}")
        import traceback

        st.error(f"Traceback: {traceback.format_exc()}")
        return []
