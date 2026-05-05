"""
מודול חיתוך אזור פנים עם שוליים — face_cropping.py

תפקיד המודול:
    לקבל תמונה מלאה + מידע על פנים שנמצאו,
    ולחתוך את אזור הפנים עם שוליים (margin) כדי שלא נחתוך בדיוק על קצה הפנים.

    מה לקחנו מהקוד המשותף (Cropped_Face.py):
        ✔ מחלקת CroppedFace — שומרת את תמונת הפנים + נקודות עיגון יחסיות
        ✔ extract_padded_face_region() — חיתוך עם שוליים יחסיים + תיקון נקודות

    למה שוליים?
        אם נחתוך בדיוק על מסגרת הפנים, נפספס הקשר (מצח, לחיים, סנטר).
        שוליים יחסיים לגודל הפנים נותנים למודל רגשות הקשר מלא יותר.

    למה נקודות עיגון יחסיות?
        לאחר החיתוך, הפיקסל (120, 80) בתמונה המקורית
        הופך ל-(120 - x_start, 80 - y_start) בתמונה החתוכה.
        אנחנו מתקנים זאת כדי שהנקודות יצביעו על המיקום הנכון בתמונה החתוכה.
"""

import numpy as np

from .face_detection import DetectedFace


# ============================
# קבוע: אחוז שוליים ברירת מחדל
# ============================
DEFAULT_MARGIN_PERCENTAGE = 0.20  # 20% שוליים — מספיק הקשר בלי רקע מיותר שמבלבל את המודל


# ============================
# מחלקת נתונים: פנים חתוכות
# ============================

class CroppedFace:
    """
    מייצג תמונת פנים שחותכה מהתמונה הגדולה, עם נקודות העיגון שלה.

    למה לא רק numpy array?
        כי אנחנו רוצים לשמור גם את נקודות העיגון (עיניים, אף, פה)
        המותאמות לתמונה החתוכה — נשתמש בהן בעתיד לישור הפנים.

    שדות:
        face_image  — תמונת הפנים החתוכה (numpy array, RGB)
        landmarks   — נקודות עיגון יחסיות לתמונה החתוכה
                      למשל: {"left_eye": (30, 20), "nose": (40, 45), ...}
    """

    def __init__(
        self,
        face_image: np.ndarray,
        landmarks: dict[str, tuple[int, int]],
    ):
        """
        יצירת אובייקט פנים חתוכות.

        פרמטרים:
            face_image — תמונת הפנים החתוכה (numpy array)
            landmarks  — נקודות עיגון מותאמות לתמונה החתוכה
        """
        self.face_image = face_image
        self.landmarks = landmarks   # קואורדינטות יחסיות (לא לתמונה המקורית!)

    def get_image(self) -> np.ndarray:
        """מחזיר את תמונת הפנים החתוכה."""
        return self.face_image

    def get_landmarks(self) -> dict[str, tuple[int, int]]:
        """מחזיר את נקודות העיגון היחסיות לתמונה החתוכה."""
        return self.landmarks

    def get_image_dimensions(self) -> tuple[int, int]:
        """מחזיר (גובה, רוחב) של תמונת הפנים החתוכה."""
        height, width = self.face_image.shape[:2]
        return (height, width)

    def __repr__(self) -> str:
        """ייצוג טקסטואלי לצרכי debugging."""
        height, width = self.get_image_dimensions()
        return f"CroppedFace(size={width}x{height}, landmarks={list(self.landmarks.keys())})"  # English repr for debugging


# ============================
# פונקציית החיתוך הראשית
# ============================

def extract_padded_face_region(
    image: np.ndarray,
    detected_face: DetectedFace,
    margin_percentage: float = DEFAULT_MARGIN_PERCENTAGE,
) -> CroppedFace:
    """
    חותך את אזור הפנים מהתמונה עם שוליים נוספים.

    השלבים:
        1. חישוב שוליים — לפי margin_percentage מגודל הפנים בכל כיוון
        2. חישוב קואורדינטות החיתוך + הגבלה לגבולות התמונה
        3. ביצוע החיתוך עם .copy() — כדי לא להחזיק את כל התמונה בזיכרון
        4. תיקון נקודות העיגון — הפיכה לקואורדינטות יחסיות לתמונה החתוכה

    פרמטרים:
        image             — התמונה המלאה (numpy array, RGB)
        detected_face     — אובייקט DetectedFace עם מיקום ונקודות עיגון
        margin_percentage — חלק יחסי מגודל הפנים לשוליים (ברירת מחדל: 0.20)

    פלט: אובייקט CroppedFace עם תמונת הפנים ונקודות עיגון יחסיות
    """
    image_height, image_width = image.shape[:2]

    # --- שלב 1: חישוב כמה פיקסלים של שוליים להוסיף ---
    # 40% מרוחב הפנים = שוליים בציר X
    # 40% מגובה הפנים = שוליים בציר Y
    margin_x = int(detected_face.face_width * margin_percentage)
    margin_y = int(detected_face.face_height * margin_percentage)

    # --- שלב 2: חישוב קואורדינטות החיתוך עם שוליים ---
    # max/min מבטיחים שלא נחרוג מגבולות התמונה
    x_start = max(0,            detected_face.face_x - margin_x)
    y_start = max(0,            detected_face.face_y - margin_y)
    x_end   = min(image_width,  detected_face.face_x + detected_face.face_width  + margin_x)
    y_end   = min(image_height, detected_face.face_y + detected_face.face_height + margin_y)

    # --- שלב 3: ביצוע החיתוך ---
    # .copy() חשוב! בלעדיו, numpy שומר הפניה לתמונה המקורית כולה בזיכרון
    face_image = image[y_start:y_end, x_start:x_end].copy()

    # --- שלב 4: תיקון נקודות עיגון לקואורדינטות יחסיות ---
    relative_landmarks = _convert_landmarks_to_relative_coordinates(
        detected_face.keypoints,
        x_offset=x_start,
        y_offset=y_start,
    )

    return CroppedFace(face_image=face_image, landmarks=relative_landmarks)


# ============================
# פונקציית עזר
# ============================

def _convert_landmarks_to_relative_coordinates(
    original_landmarks: dict[str, tuple[int, int]],
    x_offset: int,
    y_offset: int,
) -> dict[str, tuple[int, int]]:
    """
    ממיר נקודות עיגון מקואורדינטות של התמונה המקורית
    לקואורדינטות יחסיות לתמונה החתוכה.

    לדוגמה:
        נקודת "left_eye" = (120, 80) בתמונה המקורית
        x_offset = 90, y_offset = 60
        → נקודה יחסית = (120-90, 80-60) = (30, 20)

    פרמטרים:
        original_landmarks — מילון: שם_נקודה → (x, y) בתמונה המקורית
        x_offset           — עמודה שממנה התחלנו לחתוך
        y_offset           — שורה שממנה התחלנו לחתוך

    פלט: מילון: שם_נקודה → (x, y) יחסי לתמונה החתוכה
    """
    if not original_landmarks:
        return {}

    relative_landmarks = {}
    for point_name, (original_x, original_y) in original_landmarks.items():
        relative_x = original_x - x_offset
        relative_y = original_y - y_offset
        relative_landmarks[point_name] = (relative_x, relative_y)

    return relative_landmarks
