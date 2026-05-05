"""
fusion_emotion_model.py — מודל פיוז'ן לניבוי רגשות.

רעיון:
    מחליף את EmotionPredictor בצינור הקיים (drop-in replacement).
    במקום להסתמך רק על DeepFace, משלב:
        1. הסתברויות DeepFace (7 ערכים)
        2. מאפיינים גיאומטריים מ-MediaPipe (~20 ערכים)

    הוקטור המשולב עובר דרך מסווג מאומן (MLPClassifier / LogisticRegression)
    שאומן על FER2013 עם ה-ml/train_fusion_model.py.

מצב fallback:
    אם המודל לא נטען (קובץ לא קיים) — חוזרים לתוצאת DeepFace בלבד.
    אם MediaPipe לא מוצא פנים — fallback לתוצאת DeepFace.

שימוש:
    model = FusionEmotionModel(model_path="models/fusion_model.pkl")
    emotion, confidence, all_emotions = model.predict(face_image)
"""

import os
import numpy as np

from .emotion_predictor import EmotionPredictor, EMOTION_NAMES, EMOTION_BIAS
from .geometric_emotion_features import GeometricEmotionFeatures, FEATURE_DIM

# גודל וקטור הכניסה למסווג:  7 הסתברויות DeepFace  +  20 מאפיינים גיאומטריים
DEEPFACE_DIM  = len(EMOTION_NAMES)  # 7
TOTAL_DIM     = DEEPFACE_DIM + FEATURE_DIM  # 27


class FusionEmotionModel:
    """
    מודל פיוז'ן: DeepFace + מאפיינים גיאומטריים → מסווג מאומן.

    משמש כ-drop-in replacement ל-EmotionPredictor.
    החתימה של predict() זהה לחלוטין:
        emotion, confidence, all_emotions = model.predict(face_image, landmarks=landmarks)

    פרמטרים:
        model_path         — נתיב ל-fusion_model.pkl (פלט של ml/train_fusion_model.py)
        fallback_to_deepface — True: אם המודל לא קיים / MediaPipe נכשל, חזור ל-DeepFace
    """

    def __init__(self, model_path: str, fallback_to_deepface: bool = True):
        self._fallback = fallback_to_deepface
        self._model_path = model_path

        # DeepFace תמיד נטען — משמש גם כמחלץ הסתברויות וגם כ-fallback
        print("טוען DeepFace (בסיס Fusion)...")
        self._deepface = EmotionPredictor(model_path=None)

        # MediaPipe לחילוץ מאפיינים גיאומטריים
        self._geo = GeometricEmotionFeatures()

        # מסווג מאומן + scaler
        self._classifier = None
        self._scaler      = None
        self._load_classifier(model_path)

    # ------------------------------------------------------------------
    # ממשק ציבורי — זהה לחתימת EmotionPredictor.predict()
    # ------------------------------------------------------------------

    def predict(self, face_image: np.ndarray, landmarks: dict = None) -> tuple:
        """
        מנבא רגש מתמונת פנים חתוכה.

        פרמטרים:
            face_image — numpy array, RGB, uint8
            landmarks  — dict עם "left_eye"/"right_eye" (אופציונלי, להעברה ל-DeepFace)

        מחזיר:
            (dominant_emotion, confidence, all_emotions_dict)
        """
        # --- שלב 1: הסתברויות DeepFace (7 ערכים) ---
        deepface_emotion, deepface_conf, deepface_all = self._deepface.predict(
            face_image, landmarks=landmarks
        )

        # --- fallback מיידי אם המסווג לא נטען ---
        if self._classifier is None:
            return deepface_emotion, deepface_conf, deepface_all

        # --- שלב 2: מאפיינים גיאומטריים (~20 ערכים) ---
        geo_features = self._geo.extract(face_image)
        if geo_features is None:
            # MediaPipe לא מצא פנים — fallback
            return deepface_emotion, deepface_conf, deepface_all

        # --- שלב 3: בניית וקטור משולב ---
        feature_vector = self._build_vector(deepface_all, geo_features)

        # --- שלב 4: נרמול + סיווג ---
        try:
            x_scaled = self._scaler.transform(feature_vector.reshape(1, -1))
            probs     = self._classifier.predict_proba(x_scaled)[0]
        except Exception:
            return deepface_emotion, deepface_conf, deepface_all

        # --- שלב 5: פורמט זהה ל-EmotionPredictor ---
        return self._format_output(probs)

    # ------------------------------------------------------------------
    # עזרים פנימיים
    # ------------------------------------------------------------------

    def _load_classifier(self, model_path: str) -> None:
        """טוען את המסווג ואת ה-scaler מקובץ .pkl."""
        if not os.path.exists(model_path):
            if self._fallback:
                print(
                    f"[FusionEmotionModel] מודל Fusion לא נמצא ב: {model_path}\n"
                    f"  → מצב fallback: ישתמש ב-DeepFace בלבד.\n"
                    f"  → לאמן מודל: python ml/train_fusion_model.py"
                )
            else:
                raise FileNotFoundError(
                    f"מודל Fusion לא נמצא ב: {model_path}\n"
                    f"כדי לאמן: python ml/train_fusion_model.py"
                )
            return

        try:
            import joblib
            bundle = joblib.load(model_path)
            self._classifier = bundle["model"]
            self._scaler      = bundle["scaler"]
            print(f"[FusionEmotionModel] מודל נטען בהצלחה מ: {model_path}")
        except Exception as e:
            print(f"[FusionEmotionModel] שגיאה בטעינת מודל: {e}  → fallback ל-DeepFace")

    @staticmethod
    def _build_vector(deepface_all: dict, geo_features: np.ndarray) -> np.ndarray:
        """
        בונה וקטור מאפיינים מאוחד.

        סדר קבוע — חייב להיות זהה לסדר שבו בנינו את ה-dataset באימון:
            [p_angry, p_disgust, p_fear, p_happy, p_neutral, p_sad, p_surprise,
             geo_0, geo_1, ..., geo_19]
        """
        deepface_vec = np.array(
            [deepface_all.get(name, 0.0) for name in EMOTION_NAMES],
            dtype=np.float32,
        )
        return np.concatenate([deepface_vec, geo_features])

    @staticmethod
    def _format_output(probs: np.ndarray) -> tuple:
        """
        ממיר מערך הסתברויות לפורמט הסטנדרטי:
            (dominant_emotion, confidence, all_emotions_dict)
        """
        all_emotions = {name: round(float(p), 4) for name, p in zip(EMOTION_NAMES, probs)}
        dominant     = max(all_emotions, key=all_emotions.get)
        confidence   = all_emotions[dominant]
        return dominant, confidence, all_emotions
