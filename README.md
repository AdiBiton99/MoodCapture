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

```bash
pip install -r requirements.txt
python main.py                 # DeepFace + ממשק
python main.py --once          # ניתוח חד-פעמי בטרמינל
python main.py --mode fusion   # דורש models/fusion_model.pkl לאחר אימון
```

---

## FER2013 ו-MTCNN

| | FER2013 | MTCNN |
|---|---|---|
| **תפקיד** | אימון מודל CNN / הכנת דאטה | זיהוי פנים בצילום מסך |
| **שימוש בפרויקט** | `training/convert_fer2013.py`, `ml/train_model.py` | `screen_emotion/face_detection.py` |
