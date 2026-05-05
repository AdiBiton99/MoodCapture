"""
multi_face_aggregator.py — איחוד ניבויים מכמה פנים לתוצאה אחת

תפקיד הקובץ:
    כשיש יותר מפנים אחת בתמונה, שירות הניתוח מקבל כמה ניבויים.
    המאגד מחשב תוצאה אחת מייצגת.

    לדוגמה:
        Face 1: happy (0.92)
        Face 2: happy (0.75)
        Face 3: sad   (0.80)
        ─────────────────────
        Result: happy (0.84)  ← רוב הפנים שמחות, ממוצע על המנצחים

ממשק:
    aggregator = MultiFaceEmotionAggregator()
    emotion, confidence = aggregator.aggregate(predictions)
"""

from collections import Counter


class MultiFaceEmotionAggregator:
    """
    מחשב תוצאת רגש אחת מרשימת ניבויים של מספר פנים (הצבעה + ממוצע ביטחון).

    אסטרטגיה:
        בחירת הרגש הנפוץ ביותר (majority vote).
        הביטחון הסופי = ממוצע הביטחונות של כל הפנים שניבאו את הרגש המנצח.

    שימוש:
        aggregator = MultiFaceEmotionAggregator()
        result = aggregator.aggregate([("happy", 0.9), ("happy", 0.7), ("sad", 0.8)])
        # → ("happy", 0.80)
    """

    def aggregate(
        self,
        predictions: list[tuple[str, float]],
    ) -> tuple[str, float]:
        """
        מאחד רשימת ניבויים לתוצאה אחת.

        פרמטרים:
            predictions — רשימה של זוגות (emotion_name, confidence).
                          למשל: [("happy", 0.92), ("neutral", 0.67)]

        פלט:
            (emotion_name, confidence) — הרגש הדומיננטי + ביטחון ממוצע

        אם הרשימה ריקה — מחזיר (None, 0.0)
        """
        if not predictions:
            return None, 0.0

        # אם יש פנים אחת בלבד — אין מה לאחד
        if len(predictions) == 1:
            return predictions[0]

        # --- majority vote: איזה רגש הופיע הכי הרבה ---
        emotion_names = [emotion for emotion, _ in predictions]
        vote_counts   = Counter(emotion_names)
        dominant_emotion = vote_counts.most_common(1)[0][0]

        # --- ממוצע ביטחון רק עבור הפנים שניבאו את הרגש המנצח ---
        winning_confidences = [
            confidence
            for emotion, confidence in predictions
            if emotion == dominant_emotion
        ]
        average_confidence = sum(winning_confidences) / len(winning_confidences)

        return dominant_emotion, round(average_confidence, 4)
