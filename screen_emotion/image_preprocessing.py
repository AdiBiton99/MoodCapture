"""
מודול עיבוד מקדים של תמונה לפני זיהוי פנים — image_preprocessing.py

תפקיד המודול:
    לקבל תמונה גולמית (למשל: צילום מסך) ולהכין אותה
    לפני שמחפשים בה פנים.

    השלבים:
    1. שינוי גודל — כדי שהתמונה לא תהיה גדולה מדי ותאט את התוכנה
    2. טשטוש גאוסיאני — מסיר "רעש" ופרטים שמבלבלים את אלגוריתם הזיהוי
    3. נרמול ערכי הפיקסלים — מכניס את הערכים לטווח [0,1] שנוח למודלים של למידת מכונה

    חשוב לגבי נרמול:
        ברירת המחדל היא normalize=False!
        הסיבה: MTCNN מצפה לתמונה בפורמט uint8 (ערכים 0–255).
        אם ננרמל לפני זיהוי הפנים, MTCNN יכשל.
        הנרמול מתבצע בתוך מודל הרגשות עצמו — לא כאן.
"""

import cv2
import numpy as np


# ============================
# קבועים לברירות מחדל
# ============================
DEFAULT_MAX_WIDTH = 1920   # רוחב מקסימלי — שומרים על רזולוציה גבוהה לזיהוי פנים קטנות
DEFAULT_MAX_HEIGHT = 1080  # גובה מקסימלי
DEFAULT_BLUR_KERNEL = (1, 1)  # טשטוש מינימלי — גרעין (1,1) = ללא טשטוש בפועל


class ImagePreprocessor:
    """
    אחראי להכנת התמונה לפני זיהוי הפנים.

    קיבלנו תמונה גולמית — מחזירים תמונה נקייה ומוכנה לניתוח.
    """

    def __init__(
        self,
        max_width: int = DEFAULT_MAX_WIDTH,
        max_height: int = DEFAULT_MAX_HEIGHT,
        blur_kernel: tuple = DEFAULT_BLUR_KERNEL,
        normalize: bool = False,  # False! MTCNN צריך uint8 — אל תנרמל לפניו
    ):
        """
        יצירת מעבד תמונה חדש.

        פרמטרים:
            max_width   — רוחב מקסימלי מותר לתמונה (פיקסלים)
            max_height  — גובה מקסימלי מותר לתמונה (פיקסלים)
            blur_kernel — גודל גרעין הטשטוש, למשל (3,3) או (5,5)
            normalize   — האם לנרמל את ערכי הפיקסלים ל-[0,1]
                          ⚠️ השאר False אם MTCNN מגיע אחרי (ברירת מחדל)
        """
        self.max_width = max_width
        self.max_height = max_height
        self.blur_kernel = blur_kernel
        self.normalize = normalize

    # ------------------------------------------------------------------
    # מתודה ראשית — נקרא לה מבחוץ
    # ------------------------------------------------------------------

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        מעבד את התמונה בשלושה שלבים:
            1. שינוי גודל אם התמונה גדולה מדי
            2. נרמול היסטוגרמה — עוזר עם פילטרים צבעוניים חזקים
            3. טשטוש קל להסרת רעש
            4. נרמול ערכי הפיקסלים (אופציונלי)

        קלט:  תמונה מסוג numpy array (RGB)
        פלט:  תמונה מעובדת מסוג numpy array
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid image: image is None or empty. Cannot preprocess.")

        # --- שלב 1: שינוי גודל ---
        image = self._resize_if_too_large(image)

        # --- שלב 2: CLAHE — נרמול היסטוגרמה אדפטיבי ---
        # מתקן תאורה לא אחידה ופילטרי צבע חזקים (כחול/כתום)
        image = self._apply_clahe(image)

        # --- שלב 3: טשטוש גאוסיאני ---
        image = self._apply_gaussian_blur(image)

        # --- שלב 4: נרמול (אם הוגדר) ---
        if self.normalize:
            image = self._normalize_pixel_values(image)

        return image

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        CLAHE (Contrast Limited Adaptive Histogram Equalization) — נרמול ניגודיות אדפטיבי.
        עוזר ל-MTCNN לזהות פנים עם תאורה דרמטית, פילטרים כחולים/כתומים וכו'.
        """
        try:
            # המרה ל-LAB: L=בהירות, A+B=צבע
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)

            # CLAHE רק על ערוץ הבהירות — לא משנה את הצבע
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_ch = clahe.apply(l_ch)

            lab = cv2.merge((l_ch, a_ch, b_ch))
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        except Exception:
            return image  # אם נכשל — מחזיר את המקורית

    # ------------------------------------------------------------------
    # מתודות עזר פנימיות — לא נקרא להן מבחוץ
    # ------------------------------------------------------------------

    def _resize_if_too_large(self, image: np.ndarray) -> np.ndarray:
        """
        בודק אם התמונה גדולה מהמקסימום המותר.
        אם כן — מקטין אותה תוך שמירה על יחס גובה-רוחב.
        אם לא — מחזיר את התמונה כמו שהיא.
        """
        height, width = image.shape[:2]

        # חישוב יחסי ההקטנה הדרושים בכל ציר
        width_ratio = self.max_width / width
        height_ratio = self.max_height / height

        # בוחרים את היחס הקטן יותר כדי שהתמונה תתאים בשני הצירים
        scale_ratio = min(width_ratio, height_ratio)

        # אם התמונה קטנה מספיק — אין מה לשנות
        if scale_ratio >= 1.0:
            return image

        new_width = int(width * scale_ratio)
        new_height = int(height * scale_ratio)

        resized_image = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA,  # INTER_AREA מתאים להקטנה
        )
        return resized_image

    def _apply_gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        """
        טשטוש גאוסיאני להפחתת רעש.
        כאשר blur_kernel=(1,1) השלב מדולג בפועל — חשוב לצילומי מסך עם פנים קטנות,
        כי טשטוש פוגע ביכולת MTCNN לזהות פנים קטנות.
        """
        if self.blur_kernel == (1, 1):
            return image  # אין טשטוש
        blurred_image = cv2.GaussianBlur(image, self.blur_kernel, sigmaX=0)
        return blurred_image

    def _normalize_pixel_values(self, image: np.ndarray) -> np.ndarray:
        """
        ממיר ערכי פיקסלים מטווח [0, 255] לטווח [0.0, 1.0].
        מודלים של למידת מכונה מעדיפים ערכים בטווח זה.
        """
        normalized_image = image.astype(np.float32) / 255.0
        return normalized_image
