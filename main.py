"""
main.py — נקודת הכניסה הראשית של מערכת MoodCapture

    מסך → ScreenCapturer
         → ImagePreprocessor
         → MTCNNFaceDetector
         → EmotionPredictor (DeepFace) / FinetunedEmotionModel / EnsembleEmotionModel
         → MultiFaceEmotionAggregator
         → EmotionOverlay (UI)

הרצה:
    python main.py                                    # UI, DeepFace (default)
    python main.py --once                             # ניתוח חד-פעמי בטרמינל
    python main.py --model finetuned                  # רק המודל המכוונן
    python main.py --model ensemble                   # שילוב DeepFace + מכוונן
    python main.py --model ensemble --ensemble-weight 0.6
"""

import sys
import argparse

from screen_emotion.image_preprocessing import ImagePreprocessor
from screen_emotion.face_detection import MTCNNFaceDetector
from screen_emotion.emotion_predictor import EmotionPredictor
from screen_emotion.multi_face_aggregator import MultiFaceEmotionAggregator
from screen_emotion.emotion_analysis_service import EmotionAnalysisService
from capture.screen_capture import capture_screen


DEFAULT_FINETUNED_MODEL_PATH = "models/finetuned_emotion.keras"


def build_emotion_model(mode: str, ensemble_weight: float, finetuned_path: str):
    """בוחר את מודל הרגשות לפי mode."""
    if mode == "deepface":
        print("מצב: DeepFace")
        return EmotionPredictor(model_path=None)

    if mode == "finetuned":
        from screen_emotion.finetuned_emotion_model import FinetunedEmotionModel
        print("מצב: Fine-tuned (MobileNetV2)")
        return FinetunedEmotionModel(finetuned_path)

    if mode == "ensemble":
        from screen_emotion.finetuned_emotion_model import FinetunedEmotionModel
        from screen_emotion.ensemble_emotion_model import EnsembleEmotionModel
        print("מצב: Ensemble (DeepFace + Fine-tuned)")
        deepface = EmotionPredictor(model_path=None)
        finetuned = FinetunedEmotionModel(finetuned_path)
        return EnsembleEmotionModel(deepface, finetuned, weight_deepface=ensemble_weight)

    raise ValueError(f"Unknown model mode: {mode}")


def build_analysis_service(
    mode: str = "deepface",
    ensemble_weight: float = 0.5,
    finetuned_path: str = DEFAULT_FINETUNED_MODEL_PATH,
) -> EmotionAnalysisService:
    """יוצר EmotionAnalysisService עם המודל הנבחר."""
    preprocessor = ImagePreprocessor()
    face_detector = MTCNNFaceDetector()
    aggregator = MultiFaceEmotionAggregator()
    emotion_model = build_emotion_model(mode, ensemble_weight, finetuned_path)

    return EmotionAnalysisService(
        preprocessor=preprocessor,
        face_detector=face_detector,
        emotion_model=emotion_model,
        aggregator=aggregator,
    )


def run_once(mode: str, ensemble_weight: float, finetuned_path: str) -> None:
    """מצלם מסך אחד, מנתח, מדפיס לטרמינל (בלי UI)."""
    print("Building analysis pipeline...")
    service = build_analysis_service(mode, ensemble_weight, finetuned_path)

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


def run_with_overlay(mode: str, ensemble_weight: float, finetuned_path: str) -> None:
    """מריץ UI (EmotionOverlay) עם כפתור צילום מסך ובחירת אזור."""
    from PyQt5.QtWidgets import QApplication
    from ui.overlay_ui import EmotionOverlay

    print("Building analysis pipeline...")
    service = build_analysis_service(mode, ensemble_weight, finetuned_path)

    app     = QApplication(sys.argv)
    overlay = EmotionOverlay()
    overlay.show()

    def _analyze_and_display(screenshot, region_offset=(0, 0), reference_size=None):
        try:
            print("Analyzing emotions...")
            results = service.analyze(screenshot)
            overlay.display_image(
                screenshot,
                results.get("faces", []),
                region_offset=region_offset,
                reference_size=reference_size,
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
            _analyze_and_display(
                cropped,
                region_offset=(x, y),
                reference_size=(full_screenshot.shape[1], full_screenshot.shape[0]),
            )

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
            "  python main.py                                # UI, DeepFace (default)\n"
            "  python main.py --once                         # one-shot analysis\n"
            "  python main.py --model finetuned              # fine-tuned model only\n"
            "  python main.py --model ensemble               # DeepFace + fine-tuned\n"
            "  python main.py --model ensemble --ensemble-weight 0.6\n"
        ),
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Capture one screenshot, print results, and exit (no UI).",
    )
    parser.add_argument(
        "--model",
        choices=("deepface", "finetuned", "ensemble"),
        default="deepface",
        help="Which emotion model to use. Default: deepface.",
    )
    parser.add_argument(
        "--ensemble-weight",
        type=float,
        default=0.5,
        help="DeepFace weight in ensemble mode (0.0–1.0). Default: 0.5.",
    )
    parser.add_argument(
        "--finetuned-model-path",
        default=DEFAULT_FINETUNED_MODEL_PATH,
        help=f"Path to fine-tuned .keras model. Default: {DEFAULT_FINETUNED_MODEL_PATH}.",
    )

    args = parser.parse_args()

    if not 0.0 <= args.ensemble_weight <= 1.0:
        parser.error("--ensemble-weight must be between 0.0 and 1.0")

    if args.once:
        run_once(args.model, args.ensemble_weight, args.finetuned_model_path)
    else:
        run_with_overlay(args.model, args.ensemble_weight, args.finetuned_model_path)


if __name__ == "__main__":
    main()
