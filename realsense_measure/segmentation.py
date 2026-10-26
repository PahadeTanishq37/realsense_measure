"""
Person segmentation for Stage 2 subject isolation.

This is the ONLY file in the project that imports the MediaPipe selfie
segmenter, following the same lazy-singleton + auto-download-into-models/
pattern as landmarks.py.  It can be swapped out later without touching
anything else.

Public API::

    get_person_mask(color_bgr)  ->  np.ndarray | None   # H x W bool mask
    is_model_available()        ->  bool
"""
from __future__ import annotations

import os
import urllib.request

import cv2
import numpy as np

MODEL_DIR  = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "selfie_segmenter.tflite")
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_segmenter/float16/1/selfie_segmenter.tflite"
)

# Pixels with person-confidence >= this threshold become foreground.
_CONFIDENCE_THRESHOLD: float = 0.4
# Morphological cleanup kernel size (pixels).
_MORPH_KERNEL_SIZE: int = 15

_segmenter = None  # lazy singleton


def _ensure_model_downloaded() -> None:
    if os.path.exists(MODEL_PATH):
        return
    os.makedirs(MODEL_DIR, exist_ok=True)
    try:
        print(
            f"  [segmentation] Downloading selfie segmenter model "
            f"to {MODEL_PATH} ..."
        )
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("  [segmentation] Download complete.")
    except Exception as exc:
        raise RuntimeError(
            f"Could not auto-download the selfie segmenter model ({exc}).\n"
            f"Download it manually from:\n  {MODEL_URL}\n"
            f"and save it to:\n  {MODEL_PATH}"
        ) from exc


def _get_segmenter():
    global _segmenter
    if _segmenter is None:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            ImageSegmenter,
            ImageSegmenterOptions,
            RunningMode,
        )
        _ensure_model_downloaded()
        options = ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.IMAGE,
            output_confidence_masks=True,
            output_category_mask=False,
        )
        _segmenter = ImageSegmenter.create_from_options(options)
    return _segmenter


def get_person_mask(color_bgr: np.ndarray) -> "np.ndarray | None":
    """
    Run person segmentation on a BGR frame and return a binary person mask.

    Returns True wherever the model has >= _CONFIDENCE_THRESHOLD confidence
    that a person is present.  After thresholding, applies morphological
    close (fill depth-dropout gaps) and open (remove noise blobs), then
    keeps only the largest connected component (the main subject).

    Parameters
    ----------
    color_bgr : np.ndarray
        H x W x 3 uint8 BGR image, depth-aligned (same pixel grid as depth_m).

    Returns
    -------
    np.ndarray (bool, H x W) or None
        Binary person mask aligned with the input frame, or None if
        segmentation fails for any reason so callers can fall back.
    """
    try:
        import mediapipe as mp

        segmenter = _get_segmenter()
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=color_rgb)

        result = segmenter.segment(mp_image)
        if not result.confidence_masks:
            return None

        # numpy_view() -> H x W x 1 float32
        conf = result.confidence_masks[0].numpy_view()
        if conf.ndim == 3:
            conf = conf[:, :, 0]

        binary = (conf >= _CONFIDENCE_THRESHOLD).astype(np.uint8)
        if binary.sum() == 0:
            return None

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (_MORPH_KERNEL_SIZE, _MORPH_KERNEL_SIZE)
        )
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        opened = cv2.morphologyEx(closed,  cv2.MORPH_OPEN,  kernel)

        if opened.sum() == 0:
            opened = closed
        if opened.sum() == 0:
            return None

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            opened, connectivity=8
        )
        if n_labels <= 1:
            return None

        foreground_sizes = stats[1:, cv2.CC_STAT_AREA]
        largest_label    = int(foreground_sizes.argmax()) + 1
        mask = (labels == largest_label)

        if mask.sum() < 100:
            return None

        return mask

    except Exception as exc:
        print(
            f"  [segmentation] WARNING: segmentation failed ({exc!r}); "
            f"falling back to geometric isolation."
        )
        return None


def is_model_available() -> bool:
    """Return True if the selfie segmenter model file is present on disk."""
    return os.path.exists(MODEL_PATH)
