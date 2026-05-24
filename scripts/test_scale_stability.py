"""
test_scale_stability.py — בודק יציבות ניבוי בגדלים שונים של אותה תמונה.

הרעיון: לוקח תמונה אחת, יוצר ממנה גרסאות בגדלים שונים (סקייל 0.5, 0.7,
1.0, 1.5, 2.0) ומריץ את הצינור המלא של MoodCapture על כל גרסה.

אם הניבוי יציב (אותו רגש דומיננטי בכל הגדלים) — סימן שתיקון הגודל
הקבוע ב-`_prepare_image` עובד. אם לא — אנחנו עדיין רגישים לרזולוציה.

הרצה:
    python scripts/test_scale_stability.py --input path/to/image.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np


# Build the ensemble pipeline once (loading the fine-tuned model is slow).
_PIPELINE = {}


def _get_pipeline():
    if _PIPELINE:
        return _PIPELINE
    from screen_emotion.image_preprocessing import ImagePreprocessor
    from screen_emotion.face_detection import MTCNNFaceDetector
    from screen_emotion.emotion_predictor import EmotionPredictor
    from screen_emotion.finetuned_emotion_model import FinetunedEmotionModel
    from screen_emotion.ensemble_emotion_model import EnsembleEmotionModel

    _PIPELINE["pre"] = ImagePreprocessor()
    _PIPELINE["detector"] = MTCNNFaceDetector()
    _PIPELINE["df"] = EmotionPredictor()
    try:
        ft = FinetunedEmotionModel("models/finetuned_emotion.keras")
        _PIPELINE["ft"] = ft
        _PIPELINE["ens"] = EnsembleEmotionModel(_PIPELINE["df"], ft, weight_deepface=0.5)
    except Exception as exc:
        print(f"[warn] could not load fine-tuned model: {exc}")
    return _PIPELINE


def run_pipeline(rgb_image: np.ndarray, label: str) -> None:
    """Runs preprocess -> MTCNN -> crop -> all 3 predictors and prints results."""
    from screen_emotion.face_cropping import extract_padded_face_region

    p = _get_pipeline()
    preprocessed = p["pre"].process(rgb_image)
    faces = p["detector"].detect(preprocessed)
    if not faces:
        print(f"  [{label}] no face detected")
        return
    face = max(faces, key=lambda f: f.face_width * f.face_height)
    cropped = extract_padded_face_region(preprocessed, face)
    face_image = cropped.get_image()
    landmarks = cropped.get_landmarks()
    fw, fh = face.face_width, face.face_height

    df_dom, df_conf, _ = p["df"].predict(face_image, landmarks=landmarks)
    line = f"  [{label:>6s}]  bbox={fw:>3d}x{fh:<3d}  DF={df_dom.upper():>9s}({df_conf*100:>3.0f}%)"
    if "ft" in p:
        ft_dom, ft_conf, _ = p["ft"].predict(face_image, landmarks=landmarks)
        ens_dom, ens_conf, _ = p["ens"].predict(face_image, landmarks=landmarks)
        line += f"  FT={ft_dom.upper():>9s}({ft_conf*100:>3.0f}%)  ENSEMBLE={ens_dom.upper():>9s}({ens_conf*100:>3.0f}%)"
    print(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, type=Path)
    parser.add_argument("--scales", type=str, default="0.5,0.7,1.0,1.3,1.7,2.0",
                        help="Comma-separated list of resize factors.")
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"[error] image not found: {args.input}")

    bgr = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if bgr is None:
        sys.exit("[error] could not read image")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    print(f"[info] loaded {args.input.name}  shape={rgb.shape}")

    scales = [float(s) for s in args.scales.split(",") if s.strip()]
    print(f"[info] testing scales: {scales}\n")

    for s in scales:
        h = max(1, int(rgb.shape[0] * s))
        w = max(1, int(rgb.shape[1] * s))
        interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC
        scaled = cv2.resize(rgb, (w, h), interpolation=interp)
        run_pipeline(scaled, f"x{s:.2f}")


if __name__ == "__main__":
    main()
