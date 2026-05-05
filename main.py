"""
main.py — נקודת הכניסה הראשית של מערכת MoodCapture

    מסך → ScreenCapturer
         → ImagePreprocessor
         → MTCNNFaceDetector
         → FusionEmotionModel (DeepFace + MediaPipe) / EmotionPredictor (fallback)
         → MultiFaceEmotionAggregator
         → EmotionOverlay (UI)

הרצה:
    python main.py         ← UI עם כפתור צילום
    python main.py --once  ← ניתוח צילום מסך אחד בטרמינל

מצב Fusion:
    אם קיים models/fusion_model.pkl → נטען FusionEmotionModel
    אחרת → DeepFace בלבד (ברירת מחדל)
    לאמן מודל Fusion:
        python ml/build_fusion_dataset.py
        python ml/train_fusion_model.py
"""

import os
import sys
import argparse

from screen_emotion.image_preprocessing import ImagePreprocessor
from screen_emotion.face_detection import MTCNNFaceDetector
from screen_emotion.emotion_predictor import EmotionPredictor
from screen_emotion.multi_face_aggregator import MultiFaceEmotionAggregator
from screen_emotion.emotion_analysis_service import EmotionAnalysisService
from capture.screen_capture import capture_screen

_FUSION_MODEL_PATH = "models/fusion_model.pkl"


def build_analysis_service() -> EmotionAnalysisService:
    """
    יוצר EmotionAnalysisService עם כל הרכיבים מחוברים.

    בוחר אוטומטית בין Fusion ל-DeepFace:
        — אם models/fusion_model.pkl קיים → FusionEmotionModel
        — אחרת → EmotionPredictor (DeepFace בלבד)
    """
    preprocessor  = ImagePreprocessor()
    face_detector = MTCNNFaceDetector()
    aggregator    = MultiFaceEmotionAggregator()

    if os.path.exists(_FUSION_MODEL_PATH):
        from screen_emotion.fusion_emotion_model import FusionEmotionModel
        emotion_model = FusionEmotionModel(model_path=_FUSION_MODEL_PATH)
        print("מצב: Fusion (DeepFace + MediaPipe Geometric)")
    else:
        emotion_model = EmotionPredictor(model_path=None)
        print("מצב: DeepFace  (fusion_model.pkl לא נמצא — הרץ ml/train_fusion_model.py)")

    return EmotionAnalysisService(
        preprocessor=preprocessor,
        face_detector=face_detector,
        emotion_model=emotion_model,
        aggregator=aggregator,
    )


def run_once() -> None:
    """מצלם מסך אחד, מנתח, מדפיס לטרמינל (בלי UI)."""
    print("Building analysis pipeline...")
    service = build_analysis_service()

    print("Capturing screen...")
    screenshot = capture_screen()
    print(f"  Screenshot size: {screenshot.shape[1]}x{screenshot.shape[0]}")

    print("Analyzing emotions...")
    results = service.analyze(screenshot)

    faces = results.get("faces", [])
    if not faces:
        print("\nNo faces detected in the screenshot.")
        return

    print(f"\nDetected {len(faces)} face(s):")
    for i, face in enumerate(faces):
        emotion    = face.get("emotion", "unknown")
        confidence = face.get("confidence", 0.0)
        print(f"  Face {i + 1}: {emotion:>10}  ({confidence * 100:.1f}%)")

    final_emotion     = results.get("final_emotion", "unknown")
    final_confidence  = results.get("confidence", 0.0)
    print(f"\nFinal result: {final_emotion.upper()} ({final_confidence * 100:.1f}% confidence)")


def run_with_overlay() -> None:
    """מריץ UI (EmotionOverlay) עם כפתור צילום מסך ובחירת אזור."""
    from PyQt5.QtWidgets import QApplication
    from ui.overlay_ui import EmotionOverlay

    print("Building analysis pipeline...")
    service = build_analysis_service()

    app     = QApplication(sys.argv)
    overlay = EmotionOverlay()
    overlay.show()

    def _analyze_and_display(screenshot, region_offset=(0, 0)):
        try:
            print("Analyzing emotions...")
            results = service.analyze(screenshot)
            overlay.display_image(
                screenshot,
                results.get("faces", []),
                region_offset=region_offset,
            )
            overlay.update_results(results)
            n   = len(results.get("faces", []))
            emo = results.get("final_emotion", "none")
            print(f"Done: {n} face(s) | dominant: {emo}")
        except Exception as e:
            print(f"[ERROR] {e}")
        finally:
            overlay._reset_buttons()

    def on_capture():
        try:
            print("Capturing full screen...")
            screenshot = capture_screen()
            _analyze_and_display(screenshot, region_offset=(0, 0))
        except Exception as e:
            print(f"[ERROR] {e}")
            overlay._reset_buttons()

    def on_region():
        from ui.region_selector import RegionSelector
        from capture.screen_capture import ScreenCapturer

        try:
            overlay.hide()
            QApplication.processEvents()

            full_screenshot = capture_screen()
            region          = RegionSelector.select(full_screenshot)

            overlay.show()
            overlay.raise_()

            if region is None:
                print("Region selection cancelled.")
                overlay._reset_buttons()
                return

            x, y, w, h = region
            print(f"Selected region: ({x}, {y}, {w}x{h})")

            capturer = ScreenCapturer()
            cropped  = capturer.capture_region(x, y, w, h)
            _analyze_and_display(cropped, region_offset=(x, y))

        except Exception as e:
            print(f"[ERROR] {e}")
            overlay.show()
            overlay._reset_buttons()

    overlay.set_capture_callback(on_capture)
    overlay.set_region_callback(on_region)

    print("Overlay ready. Click Capture Screen to analyze.")
    sys.exit(app.exec_())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MoodCapture — ניתוח רגשות מתוכן מסך",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py        # UI עם כפתור צילום\n"
            "  python main.py --once # ניתוח אחד בטרמינל\n"
        ),
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Capture one screenshot, print results, and exit (no UI).",
    )

    args = parser.parse_args()

    if args.once:
        run_once()
    else:
        run_with_overlay()


if __name__ == "__main__":
    main()
