"""
מודול זיהוי פנים — face_detection.py (MTCNN)

תפקיד המודול:
    לקבל תמונה ולמצוא בה פנים.
    לכל פנים שנמצאות — מחזיר אובייקט DetectedFace שמכיל:
        • מיקום הפנים בתמונה (x, y, רוחב, גובה)
        • ביטחון הזיהוי (0.0 עד 1.0)
        • 5 נקודות עיגון: עיניים, אף, פינות פה


    האלגוריתם בשימוש: MTCNN
    ——————————————————————————————————————————
    MTCNN = Multi-Task Cascaded Convolutional Networks
    מודל עצבי שמיועד במיוחד לזיהוי פנים עם דיוק גבוה,
    גם כשהפנים קטנות, מוטות, או בתאורה גרועה.
    חשוב: MTCNN מצפה לתמונה בפורמט RGB (לא BGR של OpenCV!)
"""

import cv2
import numpy as np
from mtcnn import MTCNN


# ============================
# קבועים
# ============================
MINIMUM_FACE_SIZE_PIXELS = 20        # פנים קטנות מ-20x20 פיקסלים — נפסלות
MINIMUM_DETECTION_CONFIDENCE = 0.60  # סף נמוך יותר — יזהה פנים עם תאורה חריגה/פילטרים
# סף קצה: פנים שנוגעים בקצה התמונה (פחות מ-N פיקסלים מהקצה)
# בדרך כלל פנים חצויים — האימוצי'ה לא אמינה ושני העיניים / הפה לא נראים במלואם
EDGE_MARGIN_PIXELS = 4
# סף "ניתוק" של פנים גדולים: אם הפנים גדולים אבל הם נוגעים בקצה,
# רוב הסיכויים שהמסגרת לא מכילה את כל הפנים — נפסל גם כן.
MIN_KEYPOINTS_INSIDE_RATIO = 1.0   # כל 5 נקודות העיגון חייבות להיות בתוך התמונה


# ============================
# מחלקת נתונים: פנים שנמצאו
# ============================

class DetectedFace:
    """
    מייצג פנים בודדות שנמצאו בתמונה.

    מכיל את כל המידע שצריך על מיקום הפנים:
    מיקום, גודל, ביטחון הזיהוי, ונקודות עיגון.

    מה זה נקודות עיגון (keypoints)?
        נקודות ספציפיות על הפנים — עין שמאל, עין ימין, אף, פינות פה.
        אנחנו משתמשים בהן כדי לחתוך ולישר את הפנים לפני ניתוח הרגש.
    """

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        confidence: float,
        keypoints: dict[str, tuple[int, int]],
    ):
        """
        יצירת אובייקט פנים שנמצאו.

        פרמטרים:
            x, y       — קואורדינטת הפינה השמאלית-עליונה של מסגרת הפנים
            width      — רוחב מסגרת הפנים בפיקסלים
            height     — גובה מסגרת הפנים בפיקסלים
            confidence — ביטחון הזיהוי בין 0.0 ל-1.0
            keypoints  — מילון: שם_נקודה → (x, y) בתמונה המקורית
        """
        self.face_x = x
        self.face_y = y
        self.face_width = width
        self.face_height = height
        self.confidence = confidence
        self.keypoints = keypoints   # למשל: {"left_eye": (120, 80), "nose": (130, 100), ...}

    # ------------------------------------------------------------------
    # מתודת מפעל — יוצרת DetectedFace מהפלט הגולמי של MTCNN
    # ------------------------------------------------------------------

    @staticmethod
    def from_mtcnn_result(mtcnn_face_dict: dict) -> "DetectedFace":
        """
        ממיר את המילון הגולמי שמחזיר MTCNN לאובייקט DetectedFace מסודר.

        MTCNN מחזיר:
            {
              "box": [x, y, w, h],
              "confidence": 0.99,
              "keypoints": {
                  "left_eye": (x, y),
                  "right_eye": (x, y),
                  "nose": (x, y),
                  "mouth_left": (x, y),
                  "mouth_right": (x, y)
              }
            }

        אנחנו ממירים לאובייקט DetectedFace עם שמות שדות ברורים.
        """
        x, y, width, height = mtcnn_face_dict["box"]
        raw_keypoints = mtcnn_face_dict["keypoints"]

        # בנייה מחדש של מילון הנקודות בפורמט אחיד
        keypoints = {
            "left_eye":    tuple(raw_keypoints["left_eye"]),
            "right_eye":   tuple(raw_keypoints["right_eye"]),
            "nose":        tuple(raw_keypoints["nose"]),
            "mouth_left":  tuple(raw_keypoints["mouth_left"]),
            "mouth_right": tuple(raw_keypoints["mouth_right"]),
        }

        return DetectedFace(
            x=x,
            y=y,
            width=width,
            height=height,
            confidence=mtcnn_face_dict["confidence"],
            keypoints=keypoints,
        )

    # ------------------------------------------------------------------
    # מתודות נוחות לגישה לשדות
    # ------------------------------------------------------------------

    def get_bounding_box(self) -> tuple[int, int, int, int]:
        """מחזיר את מסגרת הפנים כ-(x, y, רוחב, גובה)."""
        return (self.face_x, self.face_y, self.face_width, self.face_height)

    def get_top_left_corner(self) -> tuple[int, int]:
        """מחזיר את הפינה השמאלית-עליונה של מסגרת הפנים."""
        return (self.face_x, self.face_y)

    def get_bottom_right_corner(self) -> tuple[int, int]:
        """מחזיר את הפינה הימנית-תחתונה של מסגרת הפנים."""
        return (self.face_x + self.face_width, self.face_y + self.face_height)

    def __repr__(self) -> str:
        """ייצוג טקסטואלי לצרכי debugging."""
        return (
            f"DetectedFace("
            f"x={self.face_x}, y={self.face_y}, "
            f"w={self.face_width}, h={self.face_height}, "
            f"confidence={self.confidence:.2f})"  # English repr for debugging
        )


# ============================
# מחלקת זיהוי הפנים הראשית
# ============================

class MTCNNFaceDetector:
    """
    מזהה פנים בתמונה באמצעות MTCNN ומחזיר רשימת אובייקטי DetectedFace.

    משתמש באלגוריתם MTCNN.
    מסנן אוטומטית פנים קטנות מדי ופנים עם ביטחון זיהוי נמוך מדי.
    """

    def __init__(
        self,
        minimum_face_size: int = MINIMUM_FACE_SIZE_PIXELS,
        minimum_confidence: float = MINIMUM_DETECTION_CONFIDENCE,
    ):
        """
        יצירת מזהה פנים.

        פרמטרים:
            minimum_face_size   — גודל פנים מינימלי בפיקסלים (רוחב וגובה)
            minimum_confidence  — ביטחון זיהוי מינימלי (0.0–1.0)
        """
        # טעינת המודל — זה לוקח כמה שניות בפעם הראשונה
        self._mtcnn_detector = MTCNN()
        self.minimum_face_size = minimum_face_size
        self.minimum_confidence = minimum_confidence

    # ------------------------------------------------------------------
    # מתודה ראשית
    # ------------------------------------------------------------------

    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        """
        מחפש פנים בתמונה ומחזיר רשימת פנים שעברו סינון.

        קלט:  תמונה בפורמט RGB (numpy uint8) — חשוב! לא BGR.
        פלט:  רשימת אובייקטי DetectedFace. רשימה ריקה אם אין פנים.

        הסינון שמתבצע:
            1. פנים קטנות מדי — נפסלות
            2. פנים עם ביטחון זיהוי נמוך — נפסלות
            3. פנים חצויים (נוגעים בקצה התמונה) — נפסלות, כי
               האימוצי'ה לא אמינה כשרק חצי מהפנים נראה.
        """
        # הצינור שלנו מספק תמונה ב-RGB (מ-screen_capture).
        # MTCNN מצפה ל-RGB — אין צורך בהמרה.
        # ⚠️ אם קוראים ל-detect() עם תמונת OpenCV (BGR), יש להמיר לפני הקריאה.
        raw_detections = self._mtcnn_detector.detect_faces(image)

        img_h, img_w = image.shape[:2]

        # המרה לאובייקטים + סינון
        valid_faces = []
        for raw_face in raw_detections:
            detected_face = DetectedFace.from_mtcnn_result(raw_face)

            if self._is_face_valid(detected_face, img_w, img_h):
                valid_faces.append(detected_face)
            else:
                print(
                    f"[face_detect] dropped face "
                    f"bbox=({detected_face.face_x},{detected_face.face_y},"
                    f"{detected_face.face_width}x{detected_face.face_height}) "
                    f"conf={detected_face.confidence:.2f} "
                    f"(failed size/confidence/edge check)"
                )

        return valid_faces

    # ------------------------------------------------------------------
    # מתודות עזר פנימיות
    # ------------------------------------------------------------------

    def _is_face_valid(self, face: DetectedFace,
                       img_w: int, img_h: int) -> bool:
        """
        בודק שהפנים עומדות בכל התנאים:
            1. גדולות מספיק (לפחות minimum_face_size פיקסלים)
            2. ביטחון הזיהוי גבוה מספיק
            3. לא חצויים — כל המסגרת + כל נקודות העיגון בתוך התמונה
        """
        return (
            self._is_face_large_enough(face)
            and self._is_detection_confident_enough(face)
            and self._is_face_fully_inside(face, img_w, img_h)
        )

    def _is_face_fully_inside(self, face: DetectedFace,
                              img_w: int, img_h: int) -> bool:
        """
        בודק שהפנים לא חצויים — המסגרת לא נוגעת בקצה התמונה
        וכל נקודות העיגון (עיניים, אף, פינות פה) בתוך התמונה.

        למה זה חשוב?
            פנים שחצי מהם מחוץ למסך → הזיהוי הרגשי לא אמין
            (חסר חצי מהפה / עין אחת). עדיף לוותר עליהם בכלל.
        """
        # מסגרת חייבת להיות עם מרווח מהקצה
        x = face.face_x
        y = face.face_y
        w = face.face_width
        h = face.face_height
        if x < EDGE_MARGIN_PIXELS:
            return False
        if y < EDGE_MARGIN_PIXELS:
            return False
        if x + w > img_w - EDGE_MARGIN_PIXELS:
            return False
        if y + h > img_h - EDGE_MARGIN_PIXELS:
            return False

        # כל נקודות העיגון חייבות להיות בתוך התמונה
        for kp_x, kp_y in face.keypoints.values():
            if kp_x < 0 or kp_x >= img_w or kp_y < 0 or kp_y >= img_h:
                return False

        return True

    def _is_face_large_enough(self, face: DetectedFace) -> bool:
        """
        בודק שהפנים גדולות מספיק בשני הצירים.
        פנים קטנות מ-64x64 פיקסלים — לא מספיק פרטים לזיהוי רגש.
        """
        return (
            face.face_width >= self.minimum_face_size
            and face.face_height >= self.minimum_face_size
        )

    def _is_detection_confident_enough(self, face: DetectedFace) -> bool:
        """
        בודק שהביטחון של MTCNN בזיהוי גבוה מספיק.
        ביטחון נמוך = אולי אלו לא פנים בכלל, רק רקע שנראה כמו פנים.
        """
        return face.confidence >= self.minimum_confidence
