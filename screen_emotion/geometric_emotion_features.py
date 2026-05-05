"""
geometric_emotion_features.py — חילוץ מאפיינים גיאומטריים מפנים.

הרעיון:
    MediaPipe מחזיר 468 נקודות ציון על הפנים.
    אנחנו ממירים אותן למדדים גיאומטריים קומפקטיים (~20 ערכים)
    שמייצגים ביטויי פנים — פתיחת עיניים, פה, זווית חיוך, גבות.

פלט: numpy array בצורה (FEATURE_DIM,) — כל הערכים מנורמלים ל-[0, 1] בערך.

שימוש:
    extractor = GeometricEmotionFeatures()
    features  = extractor.extract(face_image_rgb)  # None אם לא נמצאו פנים
"""

import numpy as np
import cv2

# ---------------------------------------------------------------------------
# נקודות ציון לפי אינדקס קנוני של MediaPipe 468 (FaceMesh)
# ---------------------------------------------------------------------------

# Eye Aspect Ratio — ריווח עיניים לרוחב ולאורך
_LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
_RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]

# Mouth — קצוות ותחתית/עליונה של שפתיים
_MOUTH_IDX = [61, 291, 13, 14]   # שמאל, ימין, עליון, תחתון

# גבות
_LEFT_BROW_IDX  = [70, 63, 105, 66, 107]   # נקודות גבה שמאלית
_RIGHT_BROW_IDX = [336, 296, 334, 293, 300] # נקודות גבה ימנית

# בסיס עיניים (לחישוב הרמת גבות ביחס לעיניים)
_LEFT_EYE_TOP_IDX  = 386  # נקודת קצה עליונה של עין שמאלית
_RIGHT_EYE_TOP_IDX = 159  # נקודת קצה עליונה של עין ימנית

# אף + פה (מרחק לנרמול)
_NOSE_TIP_IDX  = 1
_CHIN_IDX      = 152

# קמטי אף (disgust)
_NOSE_LEFT_IDX  = 49
_NOSE_RIGHT_IDX = 279

# קצוות שפתיים פנימיים (lip compression)
_INNER_LIP_TOP_IDX    = 13
_INNER_LIP_BOTTOM_IDX = 14

# מספר המאפיינים שהמחלקה מחזירה — לשימוש חיצוני (FusionEmotionModel, build_fusion_dataset)
FEATURE_DIM = 20


import os as _os

# נתיב למודל face_landmarker.task — ביחס לשורש הפרויקט
_MODEL_PATH = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
    "models", "face_landmarker.task",
)


class GeometricEmotionFeatures:
    """
    מחלקה לחילוץ מאפיינים גיאומטריים מתמונת פנים.

    משתמשת ב-MediaPipe Tasks API (גרסה 0.10+) עם מודל face_landmarker.task.

    שימוש:
        extractor = GeometricEmotionFeatures()
        features  = extractor.extract(face_rgb)    # np.ndarray (20,) או None
    """

    def __init__(self, model_path: str = _MODEL_PATH):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        if not _os.path.exists(model_path):
            raise FileNotFoundError(
                f"מודל FaceLandmarker לא נמצא ב: {model_path}\n"
                f"הורד אותו עם:\n"
                f"  Invoke-WebRequest -Uri https://storage.googleapis.com/mediapipe-models/"
                f"face_landmarker/face_landmarker/float16/1/face_landmarker.task "
                f"-OutFile {model_path}"
            )

        self._mp = mp
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self._detector = mp_vision.FaceLandmarker.create_from_options(options)

    # ------------------------------------------------------------------
    # ממשק ציבורי
    # ------------------------------------------------------------------

    def extract(self, face_image: np.ndarray) -> np.ndarray | None:
        """
        מחלץ וקטור מאפיינים גיאומטריים מתמונת פנים.

        פרמטרים:
            face_image — תמונת פנים חתוכה, פורמט RGB, uint8

        מחזיר:
            np.ndarray בצורה (FEATURE_DIM,) — ערכים float32
            None — אם MediaPipe לא מצא פנים בתמונה
        """
        image = self._ensure_rgb_uint8(face_image)
        landmarks = self._run_mediapipe(image)
        if landmarks is None:
            return None

        h, w = image.shape[:2]
        pts  = self._landmarks_to_array(landmarks, w, h)

        face_height = self._face_height_norm(pts)
        if face_height < 1e-6:
            return None

        features = np.array([
            self._ear(pts, _LEFT_EYE_IDX),                     # 0  EAR עין שמאלית
            self._ear(pts, _RIGHT_EYE_IDX),                    # 1  EAR עין ימנית
            self._mar(pts),                                    # 2  MAR (פתיחת פה)
            self._mouth_width(pts, face_height),               # 3  רוחב פה מנורמל
            self._smile_angle(pts),                            # 4  זווית חיוך (cos)
            self._left_brow_raise(pts, face_height),           # 5  הרמת גבה שמאלית
            self._right_brow_raise(pts, face_height),          # 6  הרמת גבה ימנית
            self._brow_furrow(pts, face_height),               # 7  קיפול גבות (כעס)
            self._lip_compression(pts, face_height),           # 8  דחיסת שפתיים
            self._nose_to_mouth_dist(pts, face_height),        # 9  מרחק אף-פה
            self._nose_wrinkle(pts, face_height),              # 10 קמט אף
            self._eye_asymmetry(pts),                          # 11 אסימטריה עיניים
            self._brow_asymmetry(pts, face_height),            # 12 אסימטריה גבות
            self._mouth_corner_height(pts, face_height),       # 13 גובה קצוות פה
            self._left_eye_openness(pts, face_height),         # 14 פתיחות עין שמאלית
            self._right_eye_openness(pts, face_height),        # 15 פתיחות עין ימנית
            self._mean_ear(pts),                               # 16 EAR ממוצע
            self._mouth_area(pts, face_height),                # 17 שטח פה מנורמל
            self._brow_to_eye_left(pts, face_height),          # 18 מרחק גבה-עין שמאל
            self._brow_to_eye_right(pts, face_height),         # 19 מרחק גבה-עין ימין
        ], dtype=np.float32)

        features = np.clip(features, 0.0, 1.0)
        return features

    def close(self):
        """שחרר משאבי MediaPipe."""
        self._detector.close()

    # ------------------------------------------------------------------
    # הפעלת MediaPipe
    # ------------------------------------------------------------------

    def _run_mediapipe(self, image: np.ndarray):
        """מריץ FaceLandmarker ומחזיר landmarks או None."""
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=image,
        )
        result = self._detector.detect(mp_image)
        if not result.face_landmarks:
            return None
        return result.face_landmarks[0]  # רשימת NormalizedLandmark

    @staticmethod
    def _landmarks_to_array(landmarks, w: int, h: int) -> np.ndarray:
        """ממיר landmarks (normalized coords) למערך פיקסל (N, 2)."""
        pts = np.array(
            [(lm.x * w, lm.y * h) for lm in landmarks],
            dtype=np.float32,
        )
        return pts

    # ------------------------------------------------------------------
    # מאפיינים גיאומטריים
    # ------------------------------------------------------------------

    @staticmethod
    def _dist(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    def _face_height_norm(self, pts: np.ndarray) -> float:
        """גובה הפנים — משמש לנרמול כל המרחקים."""
        return self._dist(pts[_NOSE_TIP_IDX], pts[_CHIN_IDX]) + 1e-6

    def _ear(self, pts: np.ndarray, eye_idx: list) -> float:
        """Eye Aspect Ratio — ריווח אנכי / אופקי של העין."""
        p = pts[eye_idx]
        vertical   = (self._dist(p[1], p[5]) + self._dist(p[2], p[4])) / 2.0
        horizontal = self._dist(p[0], p[3])
        return vertical / (horizontal + 1e-6)

    def _mean_ear(self, pts: np.ndarray) -> float:
        return (self._ear(pts, _LEFT_EYE_IDX) + self._ear(pts, _RIGHT_EYE_IDX)) / 2.0

    def _mar(self, pts: np.ndarray) -> float:
        """Mouth Aspect Ratio — פתיחת פה אנכית / אופקית."""
        mouth = pts[_MOUTH_IDX]
        vertical   = self._dist(mouth[2], mouth[3])
        horizontal = self._dist(mouth[0], mouth[1]) + 1e-6
        return vertical / horizontal

    def _mouth_width(self, pts: np.ndarray, face_h: float) -> float:
        """רוחב פה מנורמל לפי גובה הפנים."""
        return self._dist(pts[_MOUTH_IDX[0]], pts[_MOUTH_IDX[1]]) / face_h

    def _smile_angle(self, pts: np.ndarray) -> float:
        """
        זווית חיוך: cos של הזווית שנוצרת בין קצוות הפה לנקודת האמצע.
        ערך גבוה = חיוך רחב; ערך נמוך / שלילי = עצב.
        ממיר לטווח [0, 1] ע"י (cos+1)/2.
        """
        left  = pts[_MOUTH_IDX[0]]
        right = pts[_MOUTH_IDX[1]]
        dy    = right[1] - left[1]
        dx    = right[0] - left[0] + 1e-6
        angle = np.arctan2(dy, dx)
        return float((np.cos(angle) + 1.0) / 2.0)

    def _left_brow_raise(self, pts: np.ndarray, face_h: float) -> float:
        """מרחק ממוצע בין גבה שמאלית לגבול עליון העין שמאלית."""
        brow_y = float(np.mean(pts[_LEFT_BROW_IDX][:, 1]))
        eye_y  = float(pts[_LEFT_EYE_TOP_IDX][1])
        return max(0.0, (eye_y - brow_y) / face_h)

    def _right_brow_raise(self, pts: np.ndarray, face_h: float) -> float:
        """מרחק ממוצע בין גבה ימנית לגבול עליון העין הימנית."""
        brow_y = float(np.mean(pts[_RIGHT_BROW_IDX][:, 1]))
        eye_y  = float(pts[_RIGHT_EYE_TOP_IDX][1])
        return max(0.0, (eye_y - brow_y) / face_h)

    def _brow_furrow(self, pts: np.ndarray, face_h: float) -> float:
        """
        קיפול גבות — מרחק אופקי בין מרכזי הגבות מנורמל.
        ערך קטן = גבות קרובות (כעס/ריכוז); ערך גדול = גבות רחוקות.
        """
        left_center  = float(np.mean(pts[_LEFT_BROW_IDX][:, 0]))
        right_center = float(np.mean(pts[_RIGHT_BROW_IDX][:, 0]))
        return abs(right_center - left_center) / face_h

    def _lip_compression(self, pts: np.ndarray, face_h: float) -> float:
        """דחיסת שפתיים — פה סגור בלחץ (כעס, עצב). ערך קטן = שפתיים לחוצות."""
        gap = self._dist(pts[_INNER_LIP_TOP_IDX], pts[_INNER_LIP_BOTTOM_IDX])
        return gap / face_h

    def _nose_to_mouth_dist(self, pts: np.ndarray, face_h: float) -> float:
        """מרחק קצה אף לקצה עליון פה — קטן בגועל (disgust)."""
        return self._dist(pts[_NOSE_TIP_IDX], pts[_INNER_LIP_TOP_IDX]) / face_h

    def _nose_wrinkle(self, pts: np.ndarray, face_h: float) -> float:
        """
        קמט אף — קרבה בין כנפי האף (disgust, כעס).
        ערך קטן = כנפיים קרובות (עיוות אף).
        """
        return self._dist(pts[_NOSE_LEFT_IDX], pts[_NOSE_RIGHT_IDX]) / face_h

    def _eye_asymmetry(self, pts: np.ndarray) -> float:
        """אסימטריה בין EAR שתי העיניים — |EAR_left - EAR_right| מנורמל."""
        diff = abs(self._ear(pts, _LEFT_EYE_IDX) - self._ear(pts, _RIGHT_EYE_IDX))
        return min(1.0, diff * 5.0)

    def _brow_asymmetry(self, pts: np.ndarray, face_h: float) -> float:
        """אסימטריה בהרמת גבות — הבדל בגובה בין שתי הגבות."""
        diff = abs(self._left_brow_raise(pts, face_h) - self._right_brow_raise(pts, face_h))
        return min(1.0, diff * 5.0)

    def _mouth_corner_height(self, pts: np.ndarray, face_h: float) -> float:
        """
        גובה ממוצע של קצוות הפה ביחס למרכז הפה.
        ערך גבוה = חיוך (קצוות למעלה); ערך נמוך = עצב (קצוות למטה).
        """
        mouth_center_y  = float(np.mean(pts[_MOUTH_IDX][:, 1]))
        corners_y       = (pts[_MOUTH_IDX[0]][1] + pts[_MOUTH_IDX[1]][1]) / 2.0
        relative        = (mouth_center_y - corners_y) / face_h  # חיובי = חיוך
        return float(np.clip((relative + 0.1) / 0.2, 0.0, 1.0))

    def _left_eye_openness(self, pts: np.ndarray, face_h: float) -> float:
        """פתיחות עין שמאלית — EAR * (face_h / רוחב עין)."""
        p = pts[_LEFT_EYE_IDX]
        return self._dist(p[1], p[5]) / face_h

    def _right_eye_openness(self, pts: np.ndarray, face_h: float) -> float:
        """פתיחות עין ימנית."""
        p = pts[_RIGHT_EYE_IDX]
        return self._dist(p[1], p[5]) / face_h

    def _mouth_area(self, pts: np.ndarray, face_h: float) -> float:
        """שטח פה מנורמל — MAR * רוחב פה."""
        mar = self._mar(pts)
        w   = self._mouth_width(pts, face_h)
        return min(1.0, mar * w * 5.0)

    def _brow_to_eye_left(self, pts: np.ndarray, face_h: float) -> float:
        """מרחק גבה-עין שמאלית — מדד ישיר להרמת גבה."""
        return self._left_brow_raise(pts, face_h)

    def _brow_to_eye_right(self, pts: np.ndarray, face_h: float) -> float:
        """מרחק גבה-עין ימנית."""
        return self._right_brow_raise(pts, face_h)

    # ------------------------------------------------------------------
    # עזרים
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_rgb_uint8(image: np.ndarray) -> np.ndarray:
        """מוודא שהתמונה היא RGB uint8."""
        img = image.copy()
        if img.dtype != np.uint8:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2RGB)
        return img
