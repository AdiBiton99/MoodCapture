"""
emotion_analysis_service.py — צינור מלא מצילום מסך לניתוח רגשות.

תפקיד המודול:
    מקבל תמונה (צילום מסך) ומפעיל את השלבים בסדר:

        צילום מסך → ImagePreprocessor → MTCNNFaceDetector → חיתוך פנים עם שוליים
        → EmotionPredictor → MultiFaceEmotionAggregator → dict

    עקרון Dependency Injection:
        EmotionAnalysisService מקבל את כל הרכיבים בבנאי — ניתן להחליף מודל או מזהה פנים בנפרד.
"""

import numpy as np

from .face_cropping import CroppedFace, extract_padded_face_region
from .face_detection import DetectedFace


class EmotionAnalysisService:
    """
    מנהל את כל תהליך ניתוח הרגשות מהתחלה ועד הסוף.

    מקבל בבנאי את כל הרכיבים ומפעיל אותם בסדר הנכון
    כשמגיעה תמונה לניתוח דרך מתודת analyze().
    """

    def __init__(
        self,
        preprocessor,
        face_detector,
        emotion_model,
        aggregator,
    ):
        """
        יצירת שירות הניתוח עם כל הרכיבים הדרושים.

        פרמטרים:
            preprocessor   — אובייקט ImagePreprocessor עם מתודת process(image)
            face_detector  — לרוב MTCNNFaceDetector עם detect(image) → list[DetectedFace]
            emotion_model  — EmotionPredictor עם predict(...)
            aggregator     — MultiFaceEmotionAggregator עם aggregate(predictions)

        דוגמה לשימוש:
            service = EmotionAnalysisService(preprocessor, detector, model, aggregator)
            result  = service.analyze(screenshot_image)
        """
        self._preprocessor = preprocessor
        self._face_detector = face_detector
        self._emotion_model = emotion_model
        self._aggregator = aggregator

    # ------------------------------------------------------------------
    # מתודה ראשית — הכניסה הרשמית לצינור
    # ------------------------------------------------------------------

    def analyze(self, raw_image: np.ndarray) -> dict:
        """
        מנתח את כל הפנים שמופיעות בתמונה ומחזיר תוצאת רגש.

        קלט:  תמונה גולמית (numpy array RGB) — למשל צילום מסך

        פלט:  מילון עם התוצאה הכוללת:
              {
                "faces": [
                    {
                        "emotion":    "שמחה",
                        "confidence": 0.87,
                        "bbox":       (x, y, width, height)
                    },
                    ...
                ],
                "final_emotion": "שמחה",
                "confidence":    0.90
              }

        אם לא נמצאו פנים — מחזיר מבנה ריק עם הסבר.
        """
        # ═══ שלב 1: עיבוד מקדים ═══
        # resize + blur — מכין את התמונה לזיהוי פנים
        preprocessed_image = self._preprocessor.process(raw_image)
        bbox_scale_x, bbox_scale_y = self._analysis_to_raw_scales(
            raw_image, preprocessed_image
        )

        # ═══ שלב 2: זיהוי פנים ═══
        # MTCNN מחפש פנים ומחזיר רשימת DetectedFace
        detected_faces: list[DetectedFace] = self._face_detector.detect(preprocessed_image)

        # אם לא נמצאו פנים — אין מה להמשיך
        if not detected_faces:
            return self._build_empty_result("No faces detected in the image")

        # ═══ שלבים 3+4: חיתוך + ניתוח רגש לכל פנים ═══
        per_face_results = self._crop_and_analyze_each_face(preprocessed_image, detected_faces)

        # ═══ שלב 5: חישוב תוצאה כוללת ═══
        # שולחים את כל הניבויים לאגרגטור שיחשב תוצאה אחת מייצגת
        all_predictions = [result["prediction"] for result in per_face_results]
        final_emotion, final_confidence = self._aggregator.aggregate(all_predictions)

        return self._build_final_result(
            per_face_results,
            final_emotion,
            final_confidence,
            bbox_scale_x,
            bbox_scale_y,
        )

    # ------------------------------------------------------------------
    # מתודות עזר פנימיות
    # ------------------------------------------------------------------

    @staticmethod
    def _analysis_to_raw_scales(
        raw_image: np.ndarray,
        processed_image: np.ndarray,
    ) -> tuple[float, float]:
        raw_h, raw_w = raw_image.shape[:2]
        proc_h, proc_w = processed_image.shape[:2]
        scale_x = raw_w / proc_w if proc_w else 1.0
        scale_y = raw_h / proc_h if proc_h else 1.0
        return scale_x, scale_y

    @staticmethod
    def _scale_bbox(
        bbox: tuple[int, int, int, int],
        scale_x: float,
        scale_y: float,
    ) -> tuple[int, int, int, int]:
        x, y, width, height = bbox
        return (
            int(round(x * scale_x)),
            int(round(y * scale_y)),
            int(round(width * scale_x)),
            int(round(height * scale_y)),
        )

    def _crop_and_analyze_each_face(
        self,
        image: np.ndarray,
        detected_faces: list[DetectedFace],
    ) -> list[dict]:
        """
        עוברת על כל הפנים שנמצאו:
            1. חותכת כל פנים עם שוליים (extract_padded_face_region)
            2. שולחת את החיתוך למודל לניתוח רגש
            הנקודות מותאמות לקואורדינטות היחסיות בתמונה החתוכה.

        פלט: רשימת מילונים:
             [{ "bbox": (x,y,w,h), "prediction": ("שמחה", 0.87) }, ...]
        """
        results = []

        for detected_face in detected_faces:
            cropped_face: CroppedFace = extract_padded_face_region(image, detected_face)

            # --- שליחת תמונת הפנים החתוכה למודל ---
            # try/except: אם פנים אחת נכשלת, ממשיכים לשאר הפנים
            face_image = cropped_face.get_image()
            landmarks  = cropped_face.get_landmarks()  # נקודות עיניים לישור
            try:
                emotion, confidence, all_emotions = self._emotion_model.predict(face_image, landmarks=landmarks)
            except Exception:
                continue  # פנים כושלות — מדלגים עליהן בשקט

            results.append({
                "bbox":         detected_face.get_bounding_box(),
                "prediction":   (emotion, confidence),
                "all_emotions": all_emotions,  # {"happy": 0.72, "sad": 0.15, ...}
                "face_image":   face_image,
            })

        return results

    def _build_final_result(
        self,
        per_face_results: list[dict],
        final_emotion: str,
        final_confidence: float,
        bbox_scale_x: float = 1.0,
        bbox_scale_y: float = 1.0,
    ) -> dict:
        """
        בונה את מילון התוצאה הסופי שיחזור לקורא.

        פורמט:
            {
              "faces": [
                  { "emotion": "שמחה", "confidence": 0.87, "bbox": (x,y,w,h) },
                  ...
              ],
              "final_emotion": "שמחה",
              "confidence":    0.90
            }
        """
        faces_list = []
        for face_result in per_face_results:
            emotion, confidence = face_result["prediction"]
            faces_list.append({
                "emotion":      emotion,
                "confidence":   round(confidence, 4),
                "all_emotions": face_result.get("all_emotions", {}),
                "bbox":         self._scale_bbox(
                    face_result["bbox"], bbox_scale_x, bbox_scale_y
                ),
                "face_image":   face_result.get("face_image"),
            })

        return {
            "faces":         faces_list,
            "final_emotion": final_emotion,
            "confidence":    round(final_confidence, 4),
        }

    def _build_empty_result(self, reason: str) -> dict:
        """
        מחזיר מבנה תוצאה ריק כשלא נמצאו פנים.
        כולל הסבר קצר כדי שהממשק יוכל להציג הודעה מתאימה.
        """
        return {
            "faces":         [],
            "final_emotion": None,
            "confidence":    0.0,
            "message":       reason,
        }
