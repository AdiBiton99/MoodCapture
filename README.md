# מערכת ניתוח רגשות מבוססת תוכן מסך
**פרויקט גמר — עדי ביטון ותמירה אוחנה**  
מנחה: עמית בן בסט

---

## מבנה הפרויקט

```
PROJECT1/
├── main.py                    ← נקודת כניסה (UI / --once, --mode deepface|fusion)
├── requirements.txt
├── screen_emotion/            ← לוגיקת ניתוח (חבילה ראשית)
│   ├── image_preprocessing.py      ImagePreprocessor
│   ├── face_detection.py           MTCNNFaceDetector, DetectedFace
│   ├── face_cropping.py            CroppedFace, extract_padded_face_region
│   ├── emotion_predictor.py        EmotionPredictor (DeepFace / CNN מותאם)
│   ├── fusion_emotion_model.py     FusionEmotionModel (DeepFace + mesh/LBP)
│   ├── multi_face_aggregator.py    MultiFaceEmotionAggregator
│   ├── emotion_analysis_service.py EmotionAnalysisService (צינור מלא)
│   ├── mediapipe_face_mesh.py      MediaPipeFaceMeshDetector
│   ├── geometric_emotion_features.py  14 פיצ'רים גיאומטריים (אימון RF)
│   └── mesh_texture_features.py       פיצ'רי Fusion (mesh + LBP)
├── capture/                   ← צילום מסך / אזור
├── ui/                        ← PyQt5 (overlay, בחירת אזור)
├── ml/                        ← סקריפטים לאימון / בניית CSV
└── training/                  ← convert_fer2013.py
```

---

## צינור (Pipeline)

```
צילום מסך → ImagePreprocessor → MTCNNFaceDetector → חיתוך עם שוליים
    → EmotionPredictor (או FusionEmotionModel) → MultiFaceEmotionAggregator → UI
```

---

## התקנה והרצה

### דרישות מקדימות

- **Python 3.10–3.12** (נבדק עם Python 3.10.11). אין להשתמש ב-3.13+ — `tensorflow` ו-`mediapipe` עדיין לא תומכים.
- **Git** (להורדת הקוד).
- **Windows / macOS / Linux** — הפרויקט נבדק ב-Windows 11 + PowerShell.

#### בדיקת גרסת Python במחשב

ב-Windows, בדקי דרך ה-Python launcher (`py`) ולא דרך `python` — ב-Windows 11 הפקודה `python` מצביעה כברירת מחדל ל-stub של Microsoft Store שלא ירוץ:

```powershell
py -0           # מציג את כל גרסאות Python המותקנות
py -3.10 --version
```

אם אין Python 3.10–3.12 מותקן, להוריד מ-[python.org/downloads](https://www.python.org/downloads/).

---

### התקנה ב-Windows (PowerShell)

```powershell
git clone https://github.com/AdiBiton99/MoodCapture.git
cd MoodCapture

py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

> אם `Activate.ps1` נכשל עם שגיאת **ExecutionPolicy**, להריץ פעם אחת:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```

> כדי שעברית תוצג נכון בקונסול:
> ```powershell
> chcp 65001
> $env:PYTHONUTF8 = "1"
> ```

---

### התקנה ב-Linux / macOS

```bash
git clone https://github.com/AdiBiton99/MoodCapture.git
cd MoodCapture

python3.10 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

### הרצה

לפני כל הרצה — לוודא שה-venv פעיל (תופיע התווית `(.venv)` בתחילת השורה).

```bash
python main.py                 # DeepFace + ממשק UI עם כפתור צילום ובחירת אזור
python main.py --once          # ניתוח חד-פעמי בטרמינל ללא UI
python main.py --mode fusion   # דורש models/fusion_model.pkl לאחר אימון
```

> בהפעלה ראשונה, DeepFace יוריד מודלים מאומנים (~100MB) לתיקייה `~/.deepface/`. ההורדה הזו רצה פעם אחת.

---

### פתרון בעיות נפוצות

| בעיה | פתרון |
|---|---|
| `python` רק מדפיס "Python" ויוצא | זה ה-stub של Microsoft Store. להשתמש ב-`py -3.10` במקום, או לכבות אותו ב-Settings → Apps → "App execution aliases" |
| `Activate.ps1 cannot be loaded because running scripts is disabled` | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` |
| `ModuleNotFoundError` כלשהו אחרי התקנה | לוודא שה-venv פעיל (`.\.venv\Scripts\Activate.ps1`) ושהפקודה `python` מצביעה ל-`.venv\Scripts\python.exe` |
| עברית בקונסול מופיעה כ-`???` | `chcp 65001` ו-`$env:PYTHONUTF8 = "1"` (או להפעיל UTF-8 גלובלי ב-Windows Settings) |
| התקנת `tensorflow` נכשלת | ודאי שאת על Python 3.10–3.12 ועל מערכת 64-bit |

---

## FER2013 ו-MTCNN

| | FER2013 | MTCNN |
|---|---|---|
| **תפקיד** | אימון מודל CNN / הכנת דאטה | זיהוי פנים בצילום מסך |
| **שימוש בפרויקט** | `training/convert_fer2013.py`, `ml/train_model.py` | `screen_emotion/face_detection.py` |
