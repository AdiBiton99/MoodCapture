# מערכת ניתוח רגשות מבוססת תוכן מסך
**פרויקט גמר — עדי ביטון ותמירה אוחנה**  
מנחה: עמית בן בסט

---

## מבנה הפרויקט

```
MoodCapture/
├── main.py                         ← נקודת כניסה (UI / --once, --model)
├── requirements.txt
├── models/
│   └── finetuned_emotion.keras     ← מודל מאומן (לא ב-git; נוצר באימון)
├── data/fer2013/                   ← דאטה לאימון/הערכה (לא ב-git)
├── reports/                        ← תוצאות הערכה ב-JSON
├── screen_emotion/                 ← לוגיקת ניתוח
│   ├── image_preprocessing.py
│   ├── face_detection.py
│   ├── face_cropping.py
│   ├── emotion_predictor.py        ← DeepFace
│   ├── finetuned_emotion_model.py  ← MobileNetV2 מאומן על FER2013
│   ├── ensemble_emotion_model.py   ← שילוב DeepFace + מכוונן
│   ├── multi_face_aggregator.py
│   ├── emotion_analysis_service.py
│   └── evaluation_metrics.py
├── capture/                        ← צילום מסך / אזור
├── ui/                             ← PyQt5 (overlay, בחירת אזור)
└── ml/
    ├── prepare_fer2013.py
    ├── train_finetune_model.py
    ├── evaluate_emotion_model.py
    └── summarize_evaluations.py
```

---

## צינור (Pipeline)

```
צילום מסך → ImagePreprocessor → MTCNNFaceDetector → חיתוך עם שוליים
    → מודל רגש (DeepFace / Fine-tuned / Ensemble) → MultiFaceEmotionAggregator → UI
```

---

## התקנה והרצה

### דרישות מקדימות

- **Python 3.10–3.12** (נבדק עם Python 3.10.11). אין להשתמש ב-3.13+ — `tensorflow` עדיין לא תומך.
- **Git** (להורדת הקוד).
- **Windows / macOS / Linux** — הפרויקט נבדק ב-Windows 11 + PowerShell.

#### בדיקת גרסת Python במחשב

ב-Windows, בדקי דרך ה-Python launcher (`py`) ולא דרך `python` — ב-Windows 11 הפקודה `python` מצביעה כברירת מחדל ל-stub של Microsoft Store שלא ירוץ:

```powershell
py -0
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
python -m pip install -r requirements.txt
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
python -m pip install -r requirements.txt
```

---

### הכנת דאטה ומודל (למצבי Fine-tuned / Ensemble)

לפני הרצה עם `--model finetuned` או `--model ensemble` צריך:

1. **FER2013** — לשים את `fer2013.csv` ב-`data/fer2013/_download/fer2013/fer2013.csv` (למשל מ-[Kaggle FER2013](https://www.kaggle.com/datasets/msambare/fer2013)).
2. **המרה לתיקיות תמונות**:
   ```powershell
   python ml/prepare_fer2013.py
   ```
   נוצרות `data/fer2013/train/` ו-`data/fer2013/test/`.
3. **אימון** (אם אין `models/finetuned_emotion.keras`):
   ```powershell
   python ml/train_finetune_model.py
   ```

למצב **DeepFace בלבד** אין צורך בדאטה או במודל מקומי.

---

### הרצת המערכת

לפני כל הרצה — לוודא שה-venv פעיל (תופיע התווית `(.venv)` בתחילת השורה).

**ממשק (ברירת מחדל: DeepFace):**

```powershell
python main.py
```

**מצבי מודל:**

```powershell
python main.py --model deepface
python main.py --model finetuned
python main.py --model ensemble --ensemble-weight 0.5
```

**ניתוח חד-פעמי בטרמינל (בלי UI):**

```powershell
python main.py --once --model ensemble --ensemble-weight 0.5
```

**פרמטרים שימושיים:**

| פרמטר | ברירת מחדל | תיאור |
|---|---|---|
| `--model` | `deepface` | `deepface`, `finetuned`, `ensemble` |
| `--ensemble-weight` | `0.5` | משקל DeepFace ב-Ensemble (0.0–1.0) |
| `--finetuned-model-path` | `models/finetuned_emotion.keras` | נתיב למודל המכוונן |
| `--once` | כבוי | צילום מסך אחד והדפסה לטרמינל |

ב-UI: כפתור צילום מסך מלא, או בחירת אזור על המסך.

> בהפעלה ראשונה עם DeepFace, הספרייה מורידה מודלים (~100MB) ל-`~/.deepface/`. זה קורה פעם אחת.

---

### אימון והערכה (אופציונלי)

```powershell
python ml/evaluate_emotion_model.py --dataset-dir data/fer2013 --split test --mode deepface
python ml/evaluate_emotion_model.py --dataset-dir data/fer2013 --split test --mode finetuned
python ml/evaluate_emotion_model.py --dataset-dir data/fer2013 --split test --mode ensemble --ensemble-weight 0.5
python ml/summarize_evaluations.py
```

---

### פתרון בעיות נפוצות

| בעיה | פתרון |
|---|---|
| `python` רק מדפיס "Python" ויוצא | זה ה-stub של Microsoft Store. להשתמש ב-`py -3.10` במקום, או לכבות ב-Settings → Apps → "App execution aliases" |
| `Activate.ps1 cannot be loaded because running scripts is disabled` | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` |
| `Fatal error in launcher` / `pip` מחפש נתיב venv ישן | למחוק `.venv`, ליצור מחדש עם `py -3.10 -m venv .venv`, ולהתקין עם `python -m pip install -r requirements.txt` |
| `ModuleNotFoundError` אחרי התקנה | לוודא שה-venv פעיל וש-`python` מצביע ל-`.venv\Scripts\python.exe` |
| עברית בקונסול מופיעה כ-`???` | `chcp 65001` ו-`$env:PYTHONUTF8 = "1"` |
| התקנת `tensorflow` נכשלת | Python 3.10–3.12, מערכת 64-bit |
| שגיאה על `finetuned_emotion.keras` | להריץ `python ml/train_finetune_model.py` אחרי `prepare_fer2013.py` |

---

## FER2013 ו-MTCNN

| | FER2013 | MTCNN |
|---|---|---|
| **תפקיד** | אימון והערכת מודל MobileNetV2 | זיהוי פנים בצילום מסך |
| **שימוש בפרויקט** | `ml/prepare_fer2013.py`, `ml/train_finetune_model.py`, `ml/evaluate_emotion_model.py` | `screen_emotion/face_detection.py` |
