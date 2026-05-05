"""
mesh_texture_features.py — מאפייני טקסטורה מ-LBP על אזורי פנים.

רעיון:
    Local Binary Patterns (LBP) — מתאר טקסטורה מקומית ללא צורך בלמידה.
    אנחנו חותכים אזורים ספציפיים מהפנים (לחיים, מצח, אזור פה)
    ומחשבים היסטוגרמת LBP מנורמלת לכל אזור.

    מאפייני LBP לוכדים:
        — גבינות עור (disgust, כעס — שינויי טקסטורה)
        — קמטים בגבות (ריכוז, כעס)
        — חלקות לחיים (שמחה — פנים מרוחקות)

פלט: numpy array בצורה (TEXTURE_DIM,)
    ברירת מחדל: 3 אזורים × 26 bins = 78 ערכים

שימוש:
    extractor = MeshTextureFeatures()
    features  = extractor.extract(face_image_rgb)   # None אם אזור ריק
    
שילוב עם Fusion (Phase 3):
    כדי לשלב, הגדל את TOTAL_DIM ב-fusion_emotion_model.py:
        TOTAL_DIM = DEEPFACE_DIM + FEATURE_DIM + TEXTURE_DIM
    ועדכן את build_fusion_dataset.py לשמור גם את וקטור הטקסטורה.
"""

import numpy as np
import cv2

# גודל היסטוגרמת LBP לכל אזור — 2^8 ערכים אפשריים, אך uniform LBP = 59 bins
# אנחנו משתמשים ב-26 bins (radius=1, n_points=8, uniform) לביצועים טובים
_LBP_BINS    = 26
_NUM_REGIONS = 3  # לחיה שמאלית, לחיה ימנית, מצח

# גודל וקטור הטקסטורה — לשימוש חיצוני
TEXTURE_DIM = _LBP_BINS * _NUM_REGIONS  # 78


class MeshTextureFeatures:
    """
    מחלקה לחילוץ מאפייני טקסטורה LBP מאזורים ספציפיים בפנים.

    שימוש עצמאי:
        extractor = MeshTextureFeatures()
        features  = extractor.extract(face_image)   # (78,) או None

    שילוב ב-FusionEmotionModel (Phase 3):
        הוסף self._texture = MeshTextureFeatures() ל-__init__
        והוסף texture_features = self._texture.extract(face_image) ל-predict()
        ואז שלב ב-build_vector: concat([deepface_vec, geo_features, texture_features])
    """

    def extract(self, face_image: np.ndarray) -> np.ndarray | None:
        """
        מחלץ וקטור טקסטורה מהפנים.

        פרמטרים:
            face_image — תמונת פנים חתוכה, RGB, uint8

        מחזיר:
            np.ndarray בצורה (TEXTURE_DIM,) — float32 מנורמל
            None — אם אחד האזורים ריק (תמונה קטנה מדי)
        """
        gray = self._to_grayscale(face_image)
        h, w = gray.shape

        # --- חלוקה לאזורים לפי יחסים קבועים ---
        # המיקום לפי ידע אנתרופומטרי (ללא MediaPipe — מהיר יותר)
        regions = self._get_regions(h, w)

        histograms = []
        for (y1, y2, x1, x2) in regions:
            patch = gray[y1:y2, x1:x2]
            if patch.size == 0 or patch.shape[0] < 8 or patch.shape[1] < 8:
                return None
            hist = self._lbp_histogram(patch)
            histograms.append(hist)

        return np.concatenate(histograms).astype(np.float32)

    # ------------------------------------------------------------------
    # אזורים ופרמטרים
    # ------------------------------------------------------------------

    @staticmethod
    def _get_regions(h: int, w: int) -> list[tuple[int, int, int, int]]:
        """
        מחזיר רשימת אזורים (y1, y2, x1, x2) לפי גודל הפנים.

        אזורים:
            לחיה שמאלית  — 50–75% גובה, 0–30% רוחב
            לחיה ימנית   — 50–75% גובה, 70–100% רוחב
            מצח           — 0–30% גובה, 20–80% רוחב
        """
        return [
            (int(h * 0.50), int(h * 0.75), int(w * 0.00), int(w * 0.30)),  # לחיה שמאלית
            (int(h * 0.50), int(h * 0.75), int(w * 0.70), int(w * 1.00)),  # לחיה ימנית
            (int(h * 0.00), int(h * 0.30), int(w * 0.20), int(w * 0.80)),  # מצח
        ]

    @staticmethod
    def _lbp_histogram(patch: np.ndarray) -> np.ndarray:
        """
        מחשב היסטוגרמת LBP מנורמלת לאזור נתון.

        מימוש מינימלי של uniform LBP (radius=1, 8 neighbors):
            — לכל פיקסל: משווה ל-8 שכנים ובונה קוד בינארי
            — uniform patterns: רצפים עם לא יותר מ-2 מעברי 0/1
              ← 58 patterns unique + 1 bin לכל השאר = 59 bins
            — אנחנו משתמשים ב-26 bins (CLAHE תחילה → histogram נדחס לאזור relevant)

        מחזיר:
            numpy array (26,) מנורמל (סכום = 1)
        """
        # הגדלה ל-64x64 לעקביות
        patch_resized = cv2.resize(patch, (64, 64), interpolation=cv2.INTER_AREA)

        # שיפור ניגוד לפני LBP
        patch_eq = cv2.equalizeHist(patch_resized)

        lbp_map = _compute_lbp(patch_eq)

        # histogram עם 26 bins על טווח 0-255
        hist, _ = np.histogram(lbp_map.ravel(), bins=_LBP_BINS, range=(0, 256))
        hist     = hist.astype(np.float32)

        total = hist.sum()
        if total > 0:
            hist /= total

        return hist

    # ------------------------------------------------------------------
    # עזרים
    # ------------------------------------------------------------------

    @staticmethod
    def _to_grayscale(image: np.ndarray) -> np.ndarray:
        """ממיר לגווני אפור."""
        if image.ndim == 2:
            return image
        if image.ndim == 3 and image.shape[2] == 1:
            return image[:, :, 0]
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def _compute_lbp(gray: np.ndarray) -> np.ndarray:
    """
    מחשב מפת LBP (Local Binary Pattern) בסיסית על תמונה בגווני אפור.

    לכל פיקסל (מלבד הגבולות):
        משווה ל-8 שכנים בכיוון השעון → קוד בינארי 8 סיביות → ערך 0–255

    ללא תלות ב-scikit-image — שמירת requirements נקיים.
    """
    h, w    = gray.shape
    lbp_map = np.zeros((h, w), dtype=np.uint8)

    # הזזות לכל שכן בכיוון השעון (dy, dx)
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, 1),
                 (1,  1),  (1,  0),  (1, -1), (0, -1)]

    center = gray[1:-1, 1:-1].astype(np.int16)

    code = np.zeros_like(center, dtype=np.uint8)
    for bit, (dy, dx) in enumerate(neighbors):
        neighbor = gray[1+dy:h-1+dy, 1+dx:w-1+dx].astype(np.int16)
        code     |= (np.uint8(neighbor >= center) << bit)

    lbp_map[1:-1, 1:-1] = code
    return lbp_map
