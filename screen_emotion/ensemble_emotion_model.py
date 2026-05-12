"""
ensemble_emotion_model.py — אנסמבל של DeepFace + מודל Fine-tuned.

תפקיד:
    מריץ שני מנבאים על אותה תמונת פנים, ומשלב את ההסתברויות שלהם
    בממוצע משוקלל. מחזיר תוצאה בפורמט זהה ל-EmotionPredictor:
        emotion, confidence, all_emotions = model.predict(face_image, landmarks=...)

הסבר השקלול:
    p_final[label] = w * p_deepface[label] + (1 - w) * p_finetuned[label]

ערך w (weight_deepface) קובע כמה מסתמכים על DeepFace:
    1.0 → רק DeepFace
    0.0 → רק Fine-tuned
    0.5 → ממוצע שווה (ברירת מחדל)
"""

from __future__ import annotations

import numpy as np

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
        """מריץ את שני המנבאים וממצע את ההסתברויות."""
        _, _, df_probs = self._deepface.predict(face_image, landmarks=landmarks)
        _, _, ft_probs = self._finetuned.predict(face_image, landmarks=landmarks)

        combined: dict[str, float] = {}
        for name in EMOTION_NAMES:
            p_df = float(df_probs.get(name, 0.0))
            p_ft = float(ft_probs.get(name, 0.0))
            combined[name] = self._w_df * p_df + self._w_ft * p_ft

        total = sum(combined.values()) or 1.0
        all_emotions = {name: round(value / total, 4) for name, value in combined.items()}

        dominant = max(all_emotions, key=all_emotions.get)
        confidence = all_emotions[dominant]
        return dominant, confidence, all_emotions
