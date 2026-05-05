"""
emotion_predictor.py — ניבוי רגש מפנים: DeepFace (ברירת מחדל) או CNN מותאם.

שני מצבי עבודה:
    1. מצב DeepFace (ברירת מחדל):
         model = EmotionPredictor()
         — לא דורש אימון מקומי.

    2. מצב מודל מותאם אישית:
         model = EmotionPredictor(model_path="models/best_model.keras")
         — CNN שאומן על FER2013 (python ml/train_model.py)

שימוש:
    model = EmotionPredictor()
    emotion, confidence, all_emotions = model.predict(face_image)
"""

import cv2
import numpy as np


# שמות הרגשות לפי סדר FER2013 (לשימוש במודל המותאם אישית)
EMOTION_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# מיפוי שמות DeepFace לשמות הסטנדרטיים שלנו
EMOTION_MAP = {
    "angry":    "angry",
    "disgust":  "disgust",
    "fear":     "fear",
    "happy":    "happy",
    "sad":      "sad",
    "surprise": "surprise",
    "neutral":  "neutral",
}

# תיקון הטיה — מודלי FER2013 מנבאים יותר מדי "neutral"
# כי ~30% מדאטאסט האימון הם פנים ניטרליות.
# אנחנו מגדילים רגשות "שקטים" ומקטינים neutral.
EMOTION_BIAS = {
    "angry":    1.8,
    "disgust":  0.6,
    "fear":     1.0,
    "happy":    1.0,
    "sad":      1.5,
    "surprise": 1.1,
    "neutral":  0.55,
}


class EmotionPredictor:
    """
    מנבא רגש מתמונת פנים חתוכה (DeepFace או CNN מותאם).

    מצב DeepFace (ברירת מחדל):
        model = EmotionPredictor()

    מצב מודל מותאם אישית:
        model = EmotionPredictor(model_path="models/best_model.keras")
        — משתמש ב-CNN שאימנת בעצמך על FER2013.
        — כדי לאמן מודל חדש: python ml/train_model.py
    """

    def __init__(self, model_path: str = None):
        """
        טוען את מודל הרגשות.

        פרמטרים:
            model_path — None = השתמש ב-DeepFace (ברירת מחדל)
                         נתיב לקובץ = טען מודל מותאם אישית, למשל:
                         "models/best_model.keras"
        """
        self._use_custom = model_path is not None
        self._model_path = model_path

        if self._use_custom:
            self._load_custom_model(model_path)
        else:
            self._load_deepface()

    # ----------------------------------------------------------------
    # טעינת מודלים
    # ----------------------------------------------------------------

    def _load_deepface(self) -> None:
        """טוען את DeepFace ומריץ warmup להורדת משקלות."""
        print("טוען מודל DeepFace...")
        from deepface import DeepFace
        self._deepface = DeepFace
        self._warmup_deepface()
        print("מודל DeepFace נטען בהצלחה.")

    def _warmup_deepface(self) -> None:
        """מריץ ניבוי דמה כדי שהמשקלות יטענו לזיכרון."""
        try:
            dummy = np.full((48, 48, 3), 128, dtype=np.uint8)
            self._deepface.analyze(
                dummy,
                actions=["emotion"],
                enforce_detection=False,
                silent=True,
            )
        except Exception:
            pass

    def _load_custom_model(self, model_path: str) -> None:
        """
        טוען מודל CNN מותאם אישית שאומן עם ml/train_model.py.

        פרמטרים:
            model_path — נתיב לקובץ .keras, למשל "models/best_model.keras"
        """
        import os
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"לא נמצא מודל ב: {model_path}\n"
                f"כדי לאמן מודל חדש: python ml/train_model.py"
            )

        print(f"טוען מודל מותאם אישית מ: {model_path}")
        import tensorflow as tf
        self._custom_model = tf.keras.models.load_model(model_path)
        print("מודל מותאם אישית נטען בהצלחה.")

    # ----------------------------------------------------------------
    # ניבוי ראשי
    # ----------------------------------------------------------------

    def predict(self, face_image: np.ndarray, landmarks: dict = None) -> tuple:
        """
        מנבא רגש מתמונת פנים חתוכה.

        שלבים:
            1. יישור פנים לפי נקודות עיניים (אם סופק)
            2. הכנת התמונה לפורמט הדרוש
            3. ניבוי (DeepFace או מודל מותאם)
            4. תיקון הטיה + נרמול

        פרמטרים:
            face_image — numpy array של פנים חתוכות (RGB)
            landmarks  — dict אופציונלי עם "left_eye" / "right_eye" (x, y)
                         קואורדינטות יחסיות לתמונת הפנים.
                         אם סופק — הפנים יסובבו להיות ישרות לפני הניבוי.

        מחזיר:
            (emotion_name, confidence, all_emotions)
            למשל: ("happy", 0.92, {"happy": 0.92, "sad": 0.03, ...})
        """
        prepared = self._prepare_image(face_image, grayscale=self._use_custom)

        if landmarks:
            prepared = self._align_face(prepared, landmarks)

        if self._use_custom:
            return self._predict_custom(prepared)
        else:
            return self._predict_deepface(prepared)

    # ----------------------------------------------------------------
    # ניבוי DeepFace
    # ----------------------------------------------------------------

    def _predict_deepface(self, face_image: np.ndarray) -> tuple:
        """מנבא רגש עם DeepFace ומחיל תיקון הטיה."""
        result = self._deepface.analyze(
            face_image,
            actions=["emotion"],
            enforce_detection=False,
            silent=True,
        )

        emotions_raw = result[0]["emotion"]

        mapped = {
            EMOTION_MAP.get(k.lower(), k.lower()): v
            for k, v in emotions_raw.items()
        }

        return self._apply_bias_and_normalize(mapped)

    # ----------------------------------------------------------------
    # ניבוי מודל מותאם אישית
    # ----------------------------------------------------------------

    def _predict_custom(self, face_image: np.ndarray) -> tuple:
        """
        מנבא רגש עם המודל שאומן על FER2013.

        המודל מצפה לתמונה:
            — גווני אפור (1 ערוץ)
            — גודל 48×48
            — ערכים בטווח [0, 1]
        """
        import tensorflow as tf

        # שינוי גודל ל-48x48
        img = cv2.resize(face_image, (48, 48), interpolation=cv2.INTER_AREA)

        # נרמול לטווח [0, 1]
        img = img.astype(np.float32) / 255.0

        # הוספת מימדי batch ו-channel: (48, 48) → (1, 48, 48, 1)
        img = np.expand_dims(img, axis=-1)  # (48, 48, 1)
        img = np.expand_dims(img, axis=0)   # (1, 48, 48, 1)

        predictions = self._custom_model.predict(img, verbose=0)[0]

        raw_emotions = {
            name: float(prob)
            for name, prob in zip(EMOTION_NAMES, predictions)
        }

        return self._apply_bias_and_normalize(raw_emotions)

    # ----------------------------------------------------------------
    # עזרים משותפים
    # ----------------------------------------------------------------

    def _apply_bias_and_normalize(self, raw_emotions: dict) -> tuple:
        """
        מחיל תיקון הטיה ומנרמל לסכום 1.

        מחזיר: (dominant_emotion, confidence, all_emotions_dict)
        """
        corrected = {k: v * EMOTION_BIAS.get(k, 1.0) for k, v in raw_emotions.items()}

        total = sum(corrected.values()) or 1.0
        all_emotions = {k: round(v / total, 4) for k, v in corrected.items()}

        dominant   = max(all_emotions, key=all_emotions.get)
        confidence = all_emotions[dominant]
        return dominant, confidence, all_emotions

    @staticmethod
    def _prepare_image(face_image: np.ndarray, grayscale: bool = False) -> np.ndarray:
        """
        מכין תמונת פנים לניבוי:
            — המרת גווני אפור ל-RGB (או ההפך לפי מצב)
            — מוודא uint8
            — הגדלת תמונות קטנות מדי לפחות 96×96

        פרמטרים:
            grayscale — True = פלט בגווני אפור (למודל מותאם)
                        False = פלט RGB (ל-DeepFace)
        """
        img = face_image.copy()

        if grayscale:
            # המרה לגווני אפור אם צבעוני
            if img.ndim == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            elif img.ndim == 3 and img.shape[2] == 1:
                img = img[:, :, 0]
        else:
            # המרה ל-RGB אם גווני אפור
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.ndim == 3 and img.shape[2] == 1:
                img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2RGB)

        if img.dtype != np.uint8:
            img = (img * 255).clip(0, 255).astype(np.uint8)

        # הגדלת תמונות קטנות — משפר דיוק על פנים קטנות
        h, w = img.shape[:2]
        if h < 96 or w < 96:
            scale = max(96 / h, 96 / w)
            img   = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        return img

    @staticmethod
    def _align_face(face_image: np.ndarray, landmarks: dict) -> np.ndarray:
        """
        מסובב את תמונת הפנים כך שהעיניים יהיו אופקיות.

        למה זה עוזר:
            מודלים אומנו על פנים ישרות.
            ראש מוטה יכול לגרום לטעויות — למשל לקרוא לפנים שמחות "ניטרלי"
            אם הראש מוטה 20 מעלות.

        פרמטרים:
            face_image — תמונת פנים חתוכה (RGB או גווני אפור)
            landmarks  — {"left_eye": (x, y), "right_eye": (x, y), ...}
                         קואורדינטות יחסיות לתמונה
        """
        left_eye  = landmarks.get("left_eye")
        right_eye = landmarks.get("right_eye")

        if not left_eye or not right_eye:
            return face_image

        dx    = right_eye[0] - left_eye[0]
        dy    = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))

        # סיבוב קטן מ-2 מעלות — לא שווה את העלות
        if abs(angle) < 2.0:
            return face_image

        eye_center = (
            int((left_eye[0] + right_eye[0]) / 2),
            int((left_eye[1] + right_eye[1]) / 2),
        )
        h, w    = face_image.shape[:2]
        M       = cv2.getRotationMatrix2D(eye_center, angle, scale=1.0)
        aligned = cv2.warpAffine(face_image, M, (w, h), flags=cv2.INTER_CUBIC)
        return aligned
