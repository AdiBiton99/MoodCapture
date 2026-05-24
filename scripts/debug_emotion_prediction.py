"""
debug_emotion_prediction.py — דיאגנוסטיקה מלאה של צינור MoodCapture.

מריץ את **בדיוק** אותו צינור של main.py:
    raw → ImagePreprocessor (CLAHE) → MTCNN → face crop with padding → predictor

מציג את הפלט הגולמי של DeepFace (לפני ה-bias) ואת התוצאה הסופית
אחרי ה-bias, כדי שנדע מי "אחראי" על הסיווג השגוי.

הרצה:
    python scripts/debug_emotion_prediction.py --input path/to/image.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np


def emoji_bar(prob: float, width: int = 20) -> str:
    filled = int(round(prob * width))
    return "#" * filled + "." * (width - filled)


def print_distribution(label: str, probs: dict) -> None:
    dom = max(probs, key=probs.get)
    print(f"\n=== {label} ===")
    print(f"  winner: {dom.upper()} ({probs[dom]*100:.1f}%)")
    for emo, p in sorted(probs.items(), key=lambda kv: -kv[1]):
        marker = " <-- winner" if emo == dom else ""
        print(f"    {emo:>9s}  {p*100:5.1f}%  [{emoji_bar(p)}]{marker}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, type=Path)
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"[error] image not found: {args.input}")

    print(f"[1] Loading: {args.input}")
    bgr = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if bgr is None:
        sys.exit("[error] could not read image")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    print(f"    shape: {rgb.shape}")

    # ----- 2. Run the SAME preprocessing main.py runs -----
    from screen_emotion.image_preprocessing import ImagePreprocessor
    pre = ImagePreprocessor()
    preprocessed = pre.process(rgb)
    print(f"[2] CLAHE preprocess done -> {preprocessed.shape}")

    # ----- 3. Run MTCNN on the *preprocessed* image (same as main.py) -----
    from screen_emotion.face_detection import MTCNNFaceDetector
    detector = MTCNNFaceDetector()
    faces = detector.detect(preprocessed)
    if not faces:
        sys.exit("[error] No face detected")
    face = max(faces, key=lambda f: f.face_width * f.face_height)
    bx, by, bw, bh = face.get_bounding_box()
    print(f"[3] MTCNN found {len(faces)} face(s); largest=({bx},{by},{bw}x{bh}) conf={face.confidence:.2f}")

    # ----- 4. Crop with padding (same as main.py via face_cropping) -----
    from screen_emotion.face_cropping import extract_padded_face_region
    cropped = extract_padded_face_region(preprocessed, face)
    face_image = cropped.get_image()
    landmarks  = cropped.get_landmarks()
    print(f"[4] padded crop shape: {face_image.shape}")
    out_crop = Path("debug_face_crop.png")
    cv2.imwrite(str(out_crop), cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
    print(f"    wrote crop -> {out_crop}  <-- open this to see exactly what the model receives!")

    # ----- 5. DeepFace raw vs DeepFace with bias -----
    print("\n[5] Running DeepFace on the SAME crop...")
    from deepface import DeepFace
    result = DeepFace.analyze(
        face_image, actions=["emotion"], enforce_detection=False, silent=True
    )
    raw = result[0]["emotion"]
    s = sum(raw.values()) or 1.0
    raw_probs = {k.lower(): v / s for k, v in raw.items()}
    print_distribution("DeepFace RAW (no bias)", raw_probs)

    from screen_emotion.emotion_predictor import EmotionPredictor
    df = EmotionPredictor()
    df_dom, df_conf, df_probs = df.predict(face_image, landmarks=landmarks)
    print_distribution("DeepFace AFTER bias (what app uses)", df_probs)

    # ----- 6. Try the fine-tuned model — but tolerate load failure -----
    print("\n[6] Attempting fine-tuned model...")
    try:
        from screen_emotion.finetuned_emotion_model import FinetunedEmotionModel
        ft = FinetunedEmotionModel("models/finetuned_emotion.keras")
        ft_dom, ft_conf, ft_probs = ft.predict(face_image, landmarks=landmarks)
        print_distribution("Fine-tuned (MobileNetV2)", ft_probs)

        # ----- 7. Ensemble at multiple weights -----
        print("\n[7] Ensemble at multiple weights...")
        from screen_emotion.ensemble_emotion_model import EnsembleEmotionModel
        for w in (0.3, 0.5, 0.7):
            ens = EnsembleEmotionModel(df, ft, weight_deepface=w)
            e_dom, e_conf, e_probs = ens.predict(face_image, landmarks=landmarks)
            print_distribution(f"Ensemble (DeepFace weight={w})", e_probs)
    except Exception as exc:
        print(f"[warn] Fine-tuned model could not be loaded:")
        print(f"       {type(exc).__name__}: {exc}")
        print("       --> main.py would crash here too. The app is probably")
        print("       running in pure-DeepFace fallback mode.")


if __name__ == "__main__":
    main()
