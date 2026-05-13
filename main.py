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
from screen_emotion.emotion_explanation_service import EmotionExplanationService
from capture.screen_capture import capture_screen


DEFAULT_FINETUNED_MODEL_PATH = "models/finetuned_emotion.keras"


def build_emotion_model(mode: str, ensemble_weight: float, finetuned_path: str):
    """בוחר את מודל הרגשות לפי mode."""
    if mode == "deepface":
        print("מצב: DeepFace")
        return EmotionPredictor(model_path=None)

    if mode == "finetuned":
        if not os.path.exists(finetuned_path):
            print(
                f"[WARNING] Fine-tuned model not found at: {finetuned_path}\n"
                f"  Falling back to DeepFace.\n"
                f"  To train the model once, run: python ml/train_finetune_model.py"
            )
            return EmotionPredictor(model_path=None)
        from screen_emotion.finetuned_emotion_model import FinetunedEmotionModel
        print("מצב: Fine-tuned (MobileNetV2)")
        return FinetunedEmotionModel(finetuned_path)

    if mode == "ensemble":
        if not os.path.exists(finetuned_path):
            print(
                f"[WARNING] Fine-tuned model not found at: {finetuned_path}\n"
                f"  Ensemble requires the fine-tuned model — falling back to DeepFace.\n"
                f"  To train the model once, run: python ml/train_finetune_model.py"
            )
            return EmotionPredictor(model_path=None)
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
    from PyQt5.QtCore import QThread, QTimer, pyqtSignal
    from ui.overlay_ui import EmotionOverlay

    print("Building analysis pipeline...")
    service = build_analysis_service(mode, ensemble_weight, finetuned_path)

    # ── Explainable AI Emotion Assistant ──────────────────────────────
    # Built once at startup. OpenAIService loads .env (if available) and
    # silently falls back to a local explanation when no key is configured
    # or the API call fails — this keeps the existing pipeline working
    # regardless of network or credentials.
    try:
        from services.openai_service import OpenAIService
        openai_service = OpenAIService()
        print(
            "Explainable AI: OpenAI enabled"
            if openai_service.is_available()
            else f"Explainable AI: local-only mode ({openai_service.status()})"
        )
    except Exception as e:
        print(f"Explainable AI: OpenAI client unavailable ({e}). Using local fallback.")
        openai_service = None

    explanation_service = EmotionExplanationService(openai_service=openai_service)

    app     = QApplication(sys.argv)
    overlay = EmotionOverlay()
    overlay.show()

    # Watchdog timeout — if no result for a given tab within this many ms,
    # force the deterministic local fallback. Protects against API hangs.
    _EXPLAIN_WATCHDOG_MS = 22_000

    # Per-analysis state. seq monotonically increases so stale results from
    # an older capture are dropped. `cache` holds the explanation text per
    # target_id (e.g. "overall", "face:0", "face:1", ...) so switching tabs
    # is instant and free after the first generation.
    _explain_state = {
        "seq":     0,
        "results": None,
        "cache":   {},     # target_id -> str
        "workers": [],     # keep refs so QThread isn't GC'd mid-run
    }

    def _face_context(results: dict, idx: int) -> dict:
        """Build the `context` argument for explain_face()."""
        return {
            "final_emotion":    results.get("final_emotion"),
            "final_confidence": float(results.get("confidence", 0.0) or 0.0),
            "faces_count":      len(results.get("faces", []) or []),
            "face_index":       idx,
        }

    def _explain_target(target_id: str, results: dict, *,
                        openai_service_override=None) -> str:
        """Synchronously generate the explanation for a given tab target."""
        svc = (EmotionExplanationService(openai_service=openai_service_override)
               if openai_service_override is not None
               else explanation_service)
        if target_id == "overall":
            return svc.explain(results)
        try:
            idx = int(target_id.split(":", 1)[1])
            face = results["faces"][idx]
        except (KeyError, IndexError, ValueError):
            return "The selected face is no longer available."
        return svc.explain_face(face, _face_context(results, idx))

    class _ExplainWorker(QThread):
        finished_text = pyqtSignal(int, str, str)  # seq, target_id, text
        failed         = pyqtSignal(int, str, str)  # seq, target_id, error

        def __init__(self, seq: int, target_id: str, results: dict):
            super().__init__()
            self._seq       = seq
            self._target_id = target_id
            self._results   = results

        def run(self) -> None:
            try:
                text = _explain_target(self._target_id, self._results)
            except Exception as exc:
                self.failed.emit(self._seq, self._target_id, str(exc))
                return
            self.finished_text.emit(self._seq, self._target_id, text or "")

    def _on_worker_done(seq: int, target_id: str, text: str) -> None:
        if seq != _explain_state["seq"]:
            return  # superseded by a newer capture
        if not text:
            return
        _explain_state["cache"][target_id] = text
        # Only refresh the body if the user is still viewing this tab.
        if overlay.get_active_explanation_tab() == target_id:
            overlay.update_explanation(text)

    def _on_worker_failed(seq: int, target_id: str, err: str) -> None:
        if seq != _explain_state["seq"]:
            return
        print(f"[ExplainAI] worker failed ({target_id}): {err}")
        # Use the deterministic local fallback so the user still sees a sentence.
        try:
            text = _explain_target(target_id, _explain_state["results"],
                                   openai_service_override=None)
        except Exception:
            text = "The explanation could not be generated for this selection."
        _explain_state["cache"][target_id] = text
        if overlay.get_active_explanation_tab() == target_id:
            overlay.update_explanation(text)

    def _watchdog_fire(seq: int, target_id: str) -> None:
        """Force the local fallback if nothing arrived in time for this tab."""
        if seq != _explain_state["seq"]:
            return
        if target_id in _explain_state["cache"]:
            return  # already delivered
        print(f"[ExplainAI] watchdog (seq={seq}, tab={target_id}) — local fallback")
        try:
            text = _explain_target(target_id, _explain_state["results"],
                                   openai_service_override=None)
        except Exception as exc:
            print(f"[ExplainAI] watchdog fallback failed: {exc}")
            text = "The explanation could not be generated in time."
        _explain_state["cache"][target_id] = text
        if overlay.get_active_explanation_tab() == target_id:
            overlay.update_explanation(text)

    def _request_target(target_id: str) -> None:
        """Spawn a worker for `target_id` (no-op if already cached)."""
        if target_id in _explain_state["cache"]:
            if overlay.get_active_explanation_tab() == target_id:
                overlay.update_explanation(_explain_state["cache"][target_id])
            return
        seq = _explain_state["seq"]
        results = _explain_state["results"]
        if results is None:
            return
        worker = _ExplainWorker(seq, target_id, results)
        worker.finished_text.connect(_on_worker_done)
        worker.failed.connect(_on_worker_failed)
        _explain_state["workers"].append(worker)
        worker.start()
        QTimer.singleShot(
            _EXPLAIN_WATCHDOG_MS,
            lambda s=seq, t=target_id: _watchdog_fire(s, t),
        )

    def _start_explanation(results: dict) -> None:
        """
        Connected to `overlay.analysis_completed` — runs in MAIN thread.
        Resets state and primes the card with a hint. No API call is made
        until the user clicks a face on the screen.
        """
        _explain_state["seq"]    += 1
        _explain_state["results"] = results
        _explain_state["cache"]   = {}
        _explain_state["workers"] = []
        overlay.prepare_explanation(results)

    def _on_face_requested(target_id: str) -> None:
        """
        User clicked a tab in the card. The card has already updated its
        chip + tab highlight on the UI side; we just need to refresh the body.
        """
        if _explain_state["results"] is None:
            return
        if target_id in _explain_state["cache"]:
            overlay.update_explanation(_explain_state["cache"][target_id])
        else:
            overlay.show_explanation_loading()
            _request_target(target_id)

    # Wire up: when the overlay finishes rendering analysis results in the
    # main thread, kick off the Explainable AI pipeline (also in main thread).
    overlay.analysis_completed.connect(_start_explanation)
    overlay.face_explanation_requested.connect(_on_face_requested)

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
            # `update_results` is queued to the main thread. After it renders,
            # EmotionOverlay emits `analysis_completed`, which fires
            # `_start_explanation` (connected above) IN THE MAIN THREAD —
            # the right place to spawn the QTimer watchdog.
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
