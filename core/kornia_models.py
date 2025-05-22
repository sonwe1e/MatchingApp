"""
Handles loading and interfacing with Kornia models (DISK, LightGlue).
"""

import streamlit as st
import torch
import kornia.feature as KF
import kornia.utils as KU

# import kornia.geometry as KG # Not strictly needed for current LAF creation from points
import torch.nn.functional as F
import numpy as np
import cv2  # For cv2.KeyPoint

from config import (
    DEVICE,
    DETECTOR_DISK_KORNIA,
)  # MATCHER_LIGHTGLUE_KORNIA not used here directly


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
        if DETECTOR_DISK_KORNIA in detector_name_for_glue:
            lg_feat_type = "disk"
        else:
            st.warning(
                f"LightGlue is being initialized for an unrecognized Kornia detector type '{detector_name_for_glue}'. "
                "Defaulting to 'disk' features, but this may cause issues."
            )
            lg_feat_type = "disk"

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
    timg = KU.image_to_tensor(image_gray_np, keepdim=False).float().to(DEVICE) / 255.0

    if len(timg.shape) == 2:
        timg = timg.unsqueeze(0).unsqueeze(0)
    elif len(timg.shape) == 3:
        if timg.shape[0] == 1:
            timg = timg.unsqueeze(0)
        else:
            st.warning(
                f"Unexpected image tensor shape after image_to_tensor: {timg.shape}. Attempting to add batch dim."
            )
            timg = timg.unsqueeze(0)
    # No warning for 4D, assuming it's already correct (B,C,H,W)
    return timg


def extract_disk_features(disk_model, image_gray_np, top_k=None):
    """
    Extracts features using the Kornia DISK model.
    Includes resizing for DISK compatibility and robust handling of DISK outputs.
    """
    if disk_model is None or image_gray_np is None:
        return [], None

    try:
        # Preprocess image to (B,1,H,W) tensor
        timg_gray_norm = preprocess_image_for_kornia(image_gray_np)

        # DISK expects 3 input channels, repeat grayscale if necessary
        timg_for_disk = timg_gray_norm
        if timg_for_disk.shape[1] == 1:
            timg_for_disk = timg_for_disk.repeat(1, 3, 1, 1)  # B, 3, H, W
        elif timg_for_disk.shape[1] != 3:
            st.error(
                f"DISK input image has unexpected channel size: {timg_for_disk.shape[1]}. Expected 1 or 3."
            )
            return [], None

        # Resize to nearest multiple of 16 for DISK
        _, _, h, w = timg_for_disk.shape
        new_h = max(16, int(round(h / 16.0)) * 16)
        new_w = max(16, int(round(w / 16.0)) * 16)
        if h != new_h or w != new_w:
            timg_for_disk = F.interpolate(
                timg_for_disk, size=(new_h, new_w), mode="bilinear", align_corners=False
            )

        with torch.no_grad():
            # DISK model call without top_k or pad_if_not_divisible, as these caused errors previously.
            # These might be version-specific or handled internally by DISK now.
            features_out = disk_model(timg_for_disk)
            # features_out can be a list of Features objects or a single Features object
            kornia_feats_batch = (
                features_out[0] if isinstance(features_out, list) else features_out
            )

        kps_torch_all = kornia_feats_batch.keypoints  # Typically (N, 2)
        des_torch_all = getattr(
            kornia_feats_batch, "descriptors", None
        )  # Typically (N, D)
        # DISK features object might have 'scores' or 'responses'. Prioritize 'scores'.
        scores_torch_all = getattr(
            kornia_feats_batch, "detection_scores", None
        )  # DISK often uses 'detection_scores'
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
            scores_torch_all = (
                None  # Nullify scores if mismatched to avoid indexing errors
            )

        if des_torch_all is not None and des_torch_all.shape[0] != num_keypoints:
            st.warning(
                f"DISK: Keypoints ({num_keypoints}) and descriptors ({des_torch_all.shape[0]}) count mismatch. Descriptors might be unreliable."
            )
            # Optionally, one could try to truncate descriptors or keypoints to match, but it's risky.
            # For now, proceed, but be aware LightGlue might fail if counts don't match.

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
                # scores_selected_torch = scores_torch_all[top_k_indices] # Already have scores for selected
            except Exception as e_sort:
                st.warning(
                    f"Error during DISK Top-K selection: {e_sort}. Returning all features."
                )
        elif top_k is not None and top_k < num_keypoints and scores_torch_all is None:
            st.warning(
                f"DISK: `top_k` specified ({top_k}) but no scores available for sorting. Returning all {num_keypoints} features."
            )

        kps_cv = []
        kps_selected_cpu = kps_selected_torch.cpu().numpy()

        # Use selected scores if available, otherwise generate default responses for cv2.KeyPoint
        final_scores_for_kp = scores_selected_torch
        if final_scores_for_kp is None:  # If scores were nullified or never existed
            final_scores_for_kp = torch.ones(
                kps_selected_torch.shape[0], device=DEVICE, dtype=torch.float32
            )

        scores_selected_cpu = final_scores_for_kp.cpu().numpy()

        for i in range(len(kps_selected_cpu)):
            pt = kps_selected_cpu[i]
            response_val = float(scores_selected_cpu[i])
            kps_cv.append(cv2.KeyPoint(pt[0], pt[1], 10.0, -1.0, response_val, 0, -1))

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
    des1_np,
    img1_gray_np_for_shape,
    kps2_cv,
    des2_np,
    img2_gray_np_for_shape,
    # img1_gray_torch_norm and img2_gray_torch_norm are not used by KF.LightGlue's dict input
    # but could be useful if we were using KF.LightGlueMatcher.
    # For KF.LightGlue object, only image_size is needed in the dict along with kpts and descs.
):
    """
    Performs feature matching using Kornia LightGlue, following official Kornia example
    for the KF.LightGlue object.
    """
    if lightglue_model is None:
        st.error("LightGlue model is not loaded.")
        return []
    if not kps1_cv or not kps2_cv:
        st.error("Keypoint lists are empty, cannot perform LightGlue matching.")
        return []
    if des1_np is None or des2_np is None:
        st.error("Descriptors are missing, cannot perform LightGlue matching.")
        return []
    if des1_np.size == 0 or des2_np.size == 0:
        st.error("Descriptor arrays are empty, cannot perform LightGlue matching.")
        return []

    try:
        # 1. Convert OpenCV Keypoints to PyTorch Tensors (N, 2) for coordinates
        keypoints0_torch = torch.tensor(
            [[kp.pt[0], kp.pt[1]] for kp in kps1_cv], device=DEVICE, dtype=torch.float32
        )
        keypoints1_torch = torch.tensor(
            [[kp.pt[0], kp.pt[1]] for kp in kps2_cv], device=DEVICE, dtype=torch.float32
        )

        # 2. Convert NumPy descriptors to PyTorch Tensors (N, D)
        descriptors0_torch = torch.from_numpy(des1_np).to(
            device=DEVICE, dtype=torch.float32
        )
        descriptors1_torch = torch.from_numpy(des2_np).to(
            device=DEVICE, dtype=torch.float32
        )

        # 3. Get image sizes (W, H) as a tensor (B, 2)
        h0, w0 = img1_gray_np_for_shape.shape[:2]
        h1, w1 = img2_gray_np_for_shape.shape[:2]
        image_size0 = torch.tensor([[w0, h0]], device=DEVICE, dtype=torch.float32)
        image_size1 = torch.tensor([[w1, h1]], device=DEVICE, dtype=torch.float32)

        # 4. Construct the input dictionary for LightGlue (KF.LightGlue object usage)
        input_dict = {
            "image0": {
                "keypoints": keypoints0_torch.unsqueeze(0),
                "descriptors": descriptors0_torch.unsqueeze(0),
                "image_size": image_size0,
            },
            "image1": {
                "keypoints": keypoints1_torch.unsqueeze(0),
                "descriptors": descriptors1_torch.unsqueeze(0),
                "image_size": image_size1,
            },
        }

        with torch.inference_mode():
            lg_pred = lightglue_model(input_dict)

        # 5. Process LightGlue's output
        # Based on the new error message and Kornia examples,
        # lg_pred['matches'] contains pairs of indices (idx0, idx1)
        # lg_pred['scores'] contains the confidences for these matches.

        if "matches" not in lg_pred or "scores" not in lg_pred:
            st.error(
                f"LightGlue output dictionary missing 'matches' or 'scores'. Output keys: {lg_pred.keys()}"
            )
            # Fallback to matches0 and matching_scores0 if they exist and 'matches'/'scores' don't
            if "matches0" in lg_pred and "matching_scores0" in lg_pred:
                st.warning("Using 'matches0' and 'matching_scores0' as fallback.")
                matches0_indices = lg_pred["matches0"][0].cpu().numpy()
                scores_for_matches0 = lg_pred["matching_scores0"][0].cpu().numpy()

                good_matches_lg = []
                for i in range(matches0_indices.shape[0]):
                    train_idx = int(matches0_indices[i])
                    if train_idx > -1:
                        distance = 1.0 - scores_for_matches0[i]
                        good_matches_lg.append(
                            cv2.DMatch(
                                _queryIdx=i, _trainIdx=train_idx, _distance=distance
                            )
                        )
                return good_matches_lg
            return []  # If neither primary nor fallback keys are present

        # Primary path: using lg_pred['matches'] and lg_pred['scores']
        # lg_pred['matches'] shape: (B, num_matched_pairs, 2) -> (1, N_matches, 2)
        # lg_pred['scores'] shape: (B, num_matched_pairs) -> (1, N_matches)
        match_idx_pairs = lg_pred["matches"][0].cpu().numpy()  # Shape: (N_matches, 2)
        match_scores = lg_pred["scores"][0].cpu().numpy()  # Shape: (N_matches,)

        good_matches_lg = []
        for i in range(match_idx_pairs.shape[0]):
            query_idx = int(match_idx_pairs[i, 0])
            train_idx = int(match_idx_pairs[i, 1])
            score = match_scores[i]
            distance = 1.0 - score  # Convert confidence to distance for DMatch
            good_matches_lg.append(
                cv2.DMatch(_queryIdx=query_idx, _trainIdx=train_idx, _distance=distance)
            )

        return good_matches_lg

    except Exception as e:
        st.error(f"LightGlue matching failed: {e}")
        import traceback

        st.error(f"Traceback: {traceback.format_exc()}")
        return []
