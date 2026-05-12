"""
ensemble_emotion_model.py — אנסמבל של DeepFace + מודל Fine-tuned.

תפקיד:
    מריץ שני מנבאים על אותה תמונת פנים, ומשלב את ההסתברויות שלהם
    בממוצע משוקלל. מחזיר תוצאה בפורמט זהה ל-EmotionPredictor:
        emotion, confidence, all_emotions = model.predict(face_image, landmarks=...)

הסבר השקלול:
    כששני המודלים מסכימים על הרגש המנצח — ממוצע משוקלל של ההסתברויות,
    ואז חידוד קל + תיקון הטיה (כמו ב-DeepFace).
    כשהם לא מסכימים — נלקח המודל עם הביטחון הגבוה יותר, כדי שלא יידרס
    ניבוי חזק על ידי ממוצע שטוח.

ערך w (weight_deepface) קובע כמה מסתמכים על DeepFace:
    1.0 → רק DeepFace
    0.0 → רק Fine-tuned
    0.5 → ממוצע שווה (ברירת מחדל)
"""

from __future__ import annotations

from .emotion_predictor import EMOTION_BIAS, EMOTION_NAMES


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
        """משלב את שני המנבאים עם חידוד כשיש הסכמה, או בוחר את המודל החזק יותר."""
        emotion_df, confidence_df, df_probs = self._deepface.predict(
            face_image, landmarks=landmarks
        )
        emotion_ft, confidence_ft, ft_probs = self._finetuned.predict(
            face_image, landmarks=landmarks
        )

        if emotion_df != emotion_ft:
            if confidence_df >= confidence_ft:
                return emotion_df, confidence_df, df_probs
            return emotion_ft, confidence_ft, ft_probs

        combined = {
            name: self._w_df * float(df_probs.get(name, 0.0))
            + self._w_ft * float(ft_probs.get(name, 0.0))
            for name in EMOTION_NAMES
        }
        return self._finalize_distribution(combined)

    @staticmethod
    def _finalize_distribution(raw_emotions: dict[str, float]) -> tuple:
        """מחדד את ההתפלגות ומחיל תיקון הטיה לפני בחירת הרגש המנצח."""
        sharpened = {
            name: float(probability) ** 1.25
            for name, probability in raw_emotions.items()
        }
        corrected = {
            name: value * EMOTION_BIAS.get(name, 1.0)
            for name, value in sharpened.items()
        }

        total = sum(corrected.values()) or 1.0
        all_emotions = {
            name: round(value / total, 4) for name, value in corrected.items()
        }

        dominant = max(all_emotions, key=all_emotions.get)
        return dominant, all_emotions[dominant], all_emotions
