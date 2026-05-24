"""
ensemble_emotion_model.py — אנסמבל של DeepFace + מודל Fine-tuned.

תפקיד:
    מריץ שני מנבאים על אותה תמונת פנים, ומשלב את ההסתברויות שלהם
    בממוצע משוקלל. מחזיר תוצאה בפורמט זהה ל-EmotionPredictor:
        emotion, confidence, all_emotions = model.predict(face_image, landmarks=...)

הסבר השקלול:
    תמיד מבוצע ממוצע משוקלל אמיתי של ההסתברויות של שני המודלים,
    גם כשהם לא מסכימים. זה מונע מצב שבו מודל אחד שטועה בביטחון
    גבוה (למשל "fear 89%" על פנים מחייכות) דורס את המודל השני.
    הביאס (EMOTION_BIAS) **לא** מוחל שוב כאן — הוא כבר חל פעם
    אחת בתוך EmotionPredictor.predict() עבור DeepFace.

ערך w (weight_deepface) קובע כמה מסתמכים על DeepFace:
    1.0 → רק DeepFace
    0.0 → רק Fine-tuned
    0.5 → ממוצע שווה (ברירת מחדל)
"""

from __future__ import annotations

from .emotion_predictor import EMOTION_NAMES


class EnsembleEmotionModel:
    """
    מאחד שני מנבאים על ידי ממוצע משוקלל של ההסתברויות.

    דוגמה:
        deepface = EmotionPredictor(model_path=None)
        finetuned = FinetunedEmotionModel("models/finetuned_emotion.keras")
        model = EnsembleEmotionModel(deepface, finetuned, weight_deepface=0.5)
        emotion, confidence, all_emotions = model.predict(face_image, landmarks=landmarks)
    """

    def __init__(self, deepface_model, finetuned_model, weight_deepface: float = 0.5):
        if not 0.0 <= weight_deepface <= 1.0:
            raise ValueError(
                f"weight_deepface must be between 0.0 and 1.0, got {weight_deepface}"
            )

        self._deepface = deepface_model
        self._finetuned = finetuned_model
        self._w_df = float(weight_deepface)
        self._w_ft = 1.0 - self._w_df

        print(
            f"מצב Ensemble: DeepFace={self._w_df:.2f}, "
            f"Fine-tuned={self._w_ft:.2f}"
        )

    def predict(self, face_image: np.ndarray, landmarks: dict | None = None) -> tuple:
        """
        משלב את שני המנבאים בממוצע משוקלל אמיתי על כל ההסתברויות.

        תמיד עושים weighted-average — כך שמודל אחד שטועה בביטחון גבוה
        (למשל "fear 89%" על פנים מחייכות) **לא יכול לדרוס** את המודל
        השני. שני הקולות נשמעים, והסיווג נקבע על פי המשקלים.
        """
        emotion_df, confidence_df, df_probs = self._deepface.predict(
            face_image, landmarks=landmarks
        )
        emotion_ft, confidence_ft, ft_probs = self._finetuned.predict(
            face_image, landmarks=landmarks
        )

        if emotion_df != emotion_ft:
            print(
                f"[ensemble] disagreement: "
                f"DeepFace={emotion_df}({confidence_df:.2f}) "
                f"vs Fine-tuned={emotion_ft}({confidence_ft:.2f}) "
                f"-> averaging with w_df={self._w_df:.2f}"
            )

        combined = {
            name: self._w_df * float(df_probs.get(name, 0.0))
            + self._w_ft * float(ft_probs.get(name, 0.0))
            for name in EMOTION_NAMES
        }
        return self._finalize_distribution(combined)

    @staticmethod
    def _finalize_distribution(raw_emotions: dict[str, float]) -> tuple:
        """
        מחדד את ההתפלגות (sharpening קל) ומנרמל.

        חשוב: לא מחילים EMOTION_BIAS כאן. DeepFace.predict() כבר מחיל
        את הביאס פעם אחת על הפלט שלו, וה-fine-tuned מודל לא צריך אותו
        מלכתחילה. החלת הביאס שוב על הממוצע המשוקלל גרמה בעבר להגברה
        כפולה של 'angry' (×3.24 במקום ×1.8) — מה שהוביל לסיווגים שגויים
        של פנים מחייכות כ-'angry' או 'fear'.
        """
        sharpened = {
            name: float(probability) ** 1.25
            for name, probability in raw_emotions.items()
        }
        total = sum(sharpened.values()) or 1.0
        all_emotions = {
            name: round(value / total, 4) for name, value in sharpened.items()
        }
        dominant = max(all_emotions, key=all_emotions.get)
        return dominant, all_emotions[dominant], all_emotions
