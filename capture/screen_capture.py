"""
screen_capture.py — לכידת מסך

תפקיד הקובץ:
    מצלם את תוכן המסך ומחזיר אותו כ-numpy array.
    משמש כקלט הראשון לצינור הניתוח:

        capture_screen() → preprocess → face_detector → emotion_model

ספרייה בשימוש: mss
    mss היא ספרייה מהירה וחוצת פלטפורמות ללכידת מסך.
    מהירה יותר מ-PIL ומ-pyautogui.

התקנה:
    pip install mss

הרצה לבדיקה:
    python capture/screen_capture.py
"""

import numpy as np
import mss


class ScreenCapturer:
    """
    אחראי על לכידת תמונת המסך.

    מחזיר numpy array מוכן לעיבוד ע"י שאר המערכת.

    שימוש:
        capturer   = ScreenCapturer()
        screenshot = capturer.capture()          # לכידת המסך הראשי
        screenshot = capturer.capture(monitor=2) # לכידת מסך שני
    """

    def capture(self, monitor: int = 1) -> np.ndarray:
        """
        מצלם את המסך ומחזיר תמונה בפורמט RGB.

        פרמטרים:
            monitor — מספר המסך לצילום.
                      1 = מסך ראשי (ברירת מחדל)
                      0 = כל המסכים יחד (virtual screen)

        פלט:
            numpy array בצורה (height, width, 3), dtype uint8, פורמט RGB

        הערה:
            mss מחזיר תמונה ב-BGRA (4 ערוצים).
            אנחנו ממירים ל-RGB (3 ערוצים) כי MTCNN ושאר המערכת מצפים ל-RGB.
        """
        with mss.mss() as sct:
            # קבלת גבולות המסך המבוקש
            monitor_bounds = sct.monitors[monitor]

            # לכידת המסך — מחזיר תמונה בפורמט BGRA
            raw_screenshot = sct.grab(monitor_bounds)

            # המרה ל-numpy array
            bgra_image = np.array(raw_screenshot)

            # המרה מ-BGRA ל-RGB:
            # [:, :, 2] = ערוץ R (באינדקס 2 ב-BGRA)
            # [:, :, 1] = ערוץ G
            # [:, :, 0] = ערוץ B
            rgb_image = bgra_image[:, :, [2, 1, 0]]

        return rgb_image

    def capture_region(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        """
        מצלם אזור ספציפי במסך (לא את כל המסך).

        שימושי כשרוצים לנתח חלק מסוים — למשל חלון שיחת וידאו בלבד.

        פרמטרים:
            x, y          — קואורדינטת הפינה השמאלית-עליונה
            width, height — גודל האזור לצילום

        פלט:
            numpy array RGB של האזור המבוקש
        """
        region = {"left": x, "top": y, "width": width, "height": height}

        with mss.mss() as sct:
            raw_screenshot = sct.grab(region)
            bgra_image     = np.array(raw_screenshot)
            rgb_image      = bgra_image[:, :, [2, 1, 0]]

        return rgb_image


# ============================
# פונקציה ישירה — לשימוש מהיר
# ============================

def capture_screen(monitor: int = 1) -> np.ndarray:
    """
    פונקציה עוטפת פשוטה ללכידת המסך.

    שימוש:
        from capture.screen_capture import capture_screen
        image = capture_screen()

    פרמטרים:
        monitor — מספר המסך (ברירת מחדל: 1 = מסך ראשי)

    פלט:
        numpy array RGB
    """
    return ScreenCapturer().capture(monitor)


# ============================
# בדיקה עצמאית
# ============================

if __name__ == "__main__":
    print("Capturing screen...")
    img = capture_screen()
    print(f"Shape: {img.shape}")       # (height, width, 3)
    print(f"dtype: {img.dtype}")       # uint8
    print(f"Range: [{img.min()}, {img.max()}]")  # [0, 255]
    print("Screen capture OK!")
