"""
finetuned_emotion_model.py — מודל רגשות מבוסס MobileNetV2 לאחר fine-tuning על FER2013.

תפקיד:
    מודל CNN שאומן באופן עצמאי (אימון מקומי) על FER2013 דרך
    ml/train_finetune_model.py — backbone של MobileNetV2 מאומן מראש
    על ImageNet עם ראש סיווג ל-7 רגשות.

    מחליף את DeepFace כצנרת ניבוי, ושומר על אותה חתימה:
        emotion, confidence, all_emotions = model.predict(face_image, landmarks=...)

    שימוש כ-drop-in replacement ל-EmotionPredictor.

הערות:
    — קלט: RGB 96x96, נרמול MobileNetV2 (preprocess_input).
    — אין EMOTION_BIAS — הוא כויל במקור עבור DeepFace בלבד.
    — יישור פנים לפי עיניים זהה ל-EmotionPredictor.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from .emotion_predictor import EMOTION_NAMES


_INPUT_SIZE = 96


class FinetunedEmotionModel:
    """
    טוען מודל Keras מאומן (MobileNetV2 + ראש 7 רגשות) ומבצע ניבוי.

    דוגמה:
        model = FinetunedEmotionModel("models/finetuned_emotion.keras")
        emotion, confidence, all_emotions = model.predict(face_image, landmarks=landmarks)
    """

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Fine-tuned model not found at: {model_path}\n"
                f"Train it first with: python ml/train_finetune_model.py"
            )

        print(f"טוען מודל Fine-tuned מ: {model_path}")
        import tensorflow as tf
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        self._model = tf.keras.models.load_model(model_path)
        self._preprocess_input = preprocess_input
        print("מודל Fine-tuned נטען בהצלחה.")

    def predict(self, face_image: np.ndarray, landmarks: dict | None = None) -> tuple:
        """
        מנבא רגש מתמונת פנים חתוכה.

        פרמטרים:
            face_image — numpy array, RGB או גווני אפור, uint8
            landmarks  — אופציונלי, dict עם "left_eye"/"right_eye" ליישור

        מחזיר:
            (emotion, confidence, all_emotions)
        """
        prepared = self._prepare_image(face_image)

        if landmarks:
            prepared = self._align_face(prepared, landmarks)

        resized = cv2.resize(prepared, (_INPUT_SIZE, _INPUT_SIZE), interpolation=cv2.INTER_CUBIC)
        tensor = self._preprocess_input(resized.astype(np.float32))
        tensor = np.expand_dims(tensor, axis=0)

        predictions = self._model.predict(tensor, verbose=0)[0]

        all_emotions = {
            name: round(float(prob), 4)
            for name, prob in zip(EMOTION_NAMES, predictions)
        }

        dominant = max(all_emotions, key=all_emotions.get)
        confidence = all_emotions[dominant]
        return dominant, confidence, all_emotions

    @staticmethod
    def _prepare_image(face_image: np.ndarray) -> np.ndarray:
        """מוודא RGB uint8."""
        img = face_image.copy()

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2RGB)

        if img.dtype != np.uint8:
            img = (img * 255).clip(0, 255).astype(np.uint8)

        return img

    @staticmethod
    def _align_face(face_image: np.ndarray, landmarks: dict) -> np.ndarray:
        """מסובב את הפנים כך שהעיניים יהיו אופקיות (זהה ל-EmotionPredictor)."""
        left_eye = landmarks.get("left_eye")
        right_eye = landmarks.get("right_eye")

        if not left_eye or not right_eye:
            return face_image

        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))

        if abs(angle) < 2.0:
            return face_image

        eye_center = (
            int((left_eye[0] + right_eye[0]) / 2),
            int((left_eye[1] + right_eye[1]) / 2),
        )
        h, w = face_image.shape[:2]
        rotation = cv2.getRotationMatrix2D(eye_center, angle, scale=1.0)
        aligned = cv2.warpAffine(face_image, rotation, (w, h), flags=cv2.INTER_CUBIC)
        return aligned
