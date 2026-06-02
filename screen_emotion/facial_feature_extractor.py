"""
facial_feature_extractor.py — Landmark-based facial feature extraction.

Uses MediaPipe FaceLandmarker (Tasks API, mediapipe >= 0.10) to extract
measurable geometric features from a cropped face image.  All measurements
are normalised relative to face dimensions so they are comparable across
images of different sizes.

The face_landmarker.task model file (~9 MB) is downloaded automatically to
models/face_landmarker.task on first use.  Once cached it is reused with no
further network access.

If mediapipe is not installed, or the model download fails, the extractor
silently disables itself and extract() always returns an empty dict — the
rest of the pipeline degrades gracefully to distribution-based XAI.
"""

from __future__ import annotations

import logging
import math
import os
import urllib.request

import numpy as np

_LOGGER = logging.getLogger("facial_feature_extractor")

# ── Model download ────────────────────────────────────────────────────────
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
# Stored relative to this file → <project>/models/face_landmarker.task
_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "face_landmarker.task"
)


def _ensure_model() -> str:
    """Return absolute path to the model file, downloading it if needed."""
    path = os.path.abspath(_MODEL_PATH)
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"[FacialFeatureExtractor] Downloading face landmark model to: {path}")
    try:
        urllib.request.urlretrieve(_MODEL_URL, path)
        print("[FacialFeatureExtractor] Model download complete.")
    except Exception as exc:
        if os.path.exists(path):
            os.remove(path)
        raise RuntimeError(f"Model download failed: {exc}") from exc
    return path


# ── MediaPipe Face Mesh landmark indices (canonical 468-point mesh) ───────

# Eye Aspect Ratio (EAR) — 6-point formula:
#   EAR = (||P2-P6|| + ||P3-P5||) / (2 × ||P1-P4||)
_L_EYE = [33, 160, 158, 133, 153, 144]    # P1..P6 for the left eye
_R_EYE = [362, 385, 387, 263, 373, 380]   # P1..P6 for the right eye

# Eyebrow: inner (medial / near nose), centre, outer (lateral / near temple)
_L_BROW_INNER  = 55
_L_BROW_CENTER = 52
_L_BROW_OUTER  = 46

_R_BROW_INNER  = 285
_R_BROW_CENTER = 282
_R_BROW_OUTER  = 276

# Mouth
_M_LEFT   = 61    # left outer corner
_M_RIGHT  = 291   # right outer corner
_M_TOP    = 13    # upper-lip centre (inner)
_M_BOTTOM = 14    # lower-lip centre (inner)

# Face reference points (normalisation)
_FOREHEAD = 10
_CHIN     = 152
_F_LEFT   = 234   # left cheek
_F_RIGHT  = 454   # right cheek


# ── Geometry helpers ──────────────────────────────────────────────────────

def _dist(a: tuple, b: tuple) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _ear(lms: list, indices: list) -> float:
    p = [lms[i] for i in indices]
    return (_dist(p[1], p[5]) + _dist(p[2], p[4])) / (2.0 * _dist(p[0], p[3]) + 1e-6)


# ── Extractor class ───────────────────────────────────────────────────────

class FacialFeatureExtractor:
    """
    Extracts geometric facial features from a single cropped face image
    using MediaPipe FaceLandmarker (Tasks API).

    Usage:
        extractor = FacialFeatureExtractor()
        features  = extractor.extract(face_image_rgb)   # → dict or {}
    """

    def __init__(self) -> None:
        self._landmarker = None
        self._available  = False
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision as mp_vision

            model_path   = _ensure_model()
            base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
            options      = mp_vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1,
                min_face_detection_confidence=0.3,
                min_face_presence_confidence=0.3,
            )
            self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
            self._available  = True
            _LOGGER.info("MediaPipe FaceLandmarker initialised.")
        except ImportError:
            _LOGGER.warning(
                "mediapipe not installed — facial-feature extraction disabled. "
                "Install with:  pip install mediapipe"
            )
        except Exception as exc:
            _LOGGER.warning("FacialFeatureExtractor init failed: %s", exc)

    def is_available(self) -> bool:
        return self._available

    def extract(self, face_image: np.ndarray) -> dict:
        """
        Run FaceLandmarker on `face_image` (uint8 RGB numpy array) and
        return a feature dict.

        Returns {} on any error — the caller treats this as 'features
        unavailable' and falls back to distribution-based reasoning.
        """
        if not self._available or self._landmarker is None:
            return {}
        if face_image is None or face_image.size == 0:
            return {}
        try:
            return self._run(face_image)
        except Exception as exc:
            _LOGGER.debug("Feature extraction error: %s", exc)
            return {}

    def _run(self, img: np.ndarray) -> dict:
        import mediapipe as mp

        # Normalise to uint8 RGB
        if img.dtype != np.uint8:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        if img.ndim == 2:
            import cv2
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 1:
            import cv2
            img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2RGB)

        h, w = img.shape[:2]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        result   = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            return {}

        raw = result.face_landmarks[0]   # list of NormalizedLandmark
        if len(raw) < 468:
            return {}

        # Convert normalised [0,1] coordinates to pixel coordinates
        lms = [(lm.x * w, lm.y * h) for lm in raw]

        # ── Reference distances ───────────────────────────────────────
        face_h = _dist(lms[_FOREHEAD], lms[_CHIN]) or 1.0
        face_w = _dist(lms[_F_LEFT],   lms[_F_RIGHT]) or 1.0

        # ── Eye Aspect Ratio ──────────────────────────────────────────
        left_ear  = _ear(lms, _L_EYE)
        right_ear = _ear(lms, _R_EYE)
        avg_ear   = (left_ear + right_ear) / 2.0

        # ── Eyebrow height ────────────────────────────────────────────
        # (eye-centre Y − brow-centre Y) / face_h
        # Positive = brow sits above the eye (raised); smaller = lowered.
        l_eye_cy    = (lms[_L_EYE[0]][1] + lms[_L_EYE[3]][1]) / 2.0
        r_eye_cy    = (lms[_R_EYE[0]][1] + lms[_R_EYE[3]][1]) / 2.0
        l_brow_lift = (l_eye_cy - lms[_L_BROW_CENTER][1]) / face_h
        r_brow_lift = (r_eye_cy - lms[_R_BROW_CENTER][1]) / face_h
        avg_brow_lift = (l_brow_lift + r_brow_lift) / 2.0

        # ── Brow inner-corner distance ────────────────────────────────
        brow_inner_dist = (
            _dist(lms[_L_BROW_INNER], lms[_R_BROW_INNER]) / face_w
        )

        # ── Oblique inner-corner raise (AU1, sad / fear indicator) ────
        # True when BOTH inner corners are higher (smaller Y) than
        # their outer counterparts.
        l_inner_raised = lms[_L_BROW_INNER][1] < lms[_L_BROW_OUTER][1]
        r_inner_raised = lms[_R_BROW_INNER][1] < lms[_R_BROW_OUTER][1]

        # ── Mouth corner angle ────────────────────────────────────────
        # (lip-centre Y − corners Y) / face_h
        # Positive → corners ABOVE centre = upturned (smile).
        # Negative → corners BELOW centre = downturned (frown).
        m_center_y      = (lms[_M_TOP][1] + lms[_M_BOTTOM][1]) / 2.0
        m_corners_y     = (lms[_M_LEFT][1] + lms[_M_RIGHT][1]) / 2.0
        mouth_corner_angle = (m_center_y - m_corners_y) / face_h

        # ── Mouth aspect ratio + width ────────────────────────────────
        mouth_h           = _dist(lms[_M_TOP],  lms[_M_BOTTOM])
        mouth_w           = _dist(lms[_M_LEFT], lms[_M_RIGHT])
        mar               = mouth_h / (mouth_w + 1e-6)
        mouth_width_ratio = mouth_w / face_w

        # ── Build feature dict ────────────────────────────────────────
        features: dict = {
            # ── Raw scalars (for debugging / downstream use) ───────────
            "ear":                round(avg_ear,            4),
            "brow_lift":          round(avg_brow_lift,      4),
            "brow_inner_dist":    round(brow_inner_dist,    4),
            "mouth_corner_angle": round(mouth_corner_angle, 4),
            "mar":                round(mar,                4),
            "mouth_width_ratio":  round(mouth_width_ratio,  4),

            # ── Eye openness ───────────────────────────────────────────
            # Narrowed < relaxed < widened.  Ranges intentionally overlap-free.
            "eyes_narrowed":        avg_ear < 0.20,
            "eyes_slightly_narrow": 0.20 <= avg_ear < 0.26,
            "eyes_relaxed":         0.26 <= avg_ear <= 0.33,   # normal / happy range
            "eyes_widened":         0.33 < avg_ear <= 0.37,
            "eyes_wide_open":       avg_ear > 0.37,

            # ── Eyebrow height ─────────────────────────────────────────
            # Four levels from lowered → slightly raised.
            "brows_lowered":        avg_brow_lift < 0.055,
            "brows_neutral":        0.055 <= avg_brow_lift <= 0.095,  # resting position
            "brows_slightly_raised": 0.095 < avg_brow_lift <= 0.11,
            "brows_raised":         avg_brow_lift > 0.11,

            # ── Brow inner-corner distance ─────────────────────────────
            "brows_contracted": brow_inner_dist < 0.14,
            "brows_relaxed":    brow_inner_dist > 0.17,   # widened threshold (was 0.22)

            # ── Oblique inner-corner raise (AU1 — fear / sad) ──────────
            "brow_inner_raised": l_inner_raised and r_inner_raised,

            # ── Mouth corner angle — three smile levels + frown ────────
            "smile_strong":       mouth_corner_angle > 0.050,
            "smile":              0.030 < mouth_corner_angle <= 0.050,
            "slight_smile":       0.010 < mouth_corner_angle <= 0.030,
            "mouth_neutral":      abs(mouth_corner_angle) <= 0.010,
            "mouth_corners_down": mouth_corner_angle < -0.015,

            # ── Mouth openness ─────────────────────────────────────────
            "mouth_open":          mar > 0.45,
            "mouth_slightly_open": 0.18 < mar <= 0.45,
            "mouth_closed_relaxed": 0.08 <= mar <= 0.22,   # lips together, no tension
            "mouth_compressed":    mar < 0.08,

            # ── Mouth width ────────────────────────────────────────────
            "mouth_wide": mouth_width_ratio > 0.48,   # lowered from 0.54
        }

        return features
