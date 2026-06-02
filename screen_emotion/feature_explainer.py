"""
feature_explainer.py — Maps extracted facial landmark features to
natural-language explanations.

All signals emitted by this module are grounded in measured landmark
geometry from FacialFeatureExtractor.  Nothing is hardcoded per emotion
label — every string appears ONLY when the corresponding boolean flag was
actually set True by the extractor on the real face image.

Public API:
    build_feature_signals(features, emotion)
        → list[tuple[str, str]]    (label, tooltip) pairs for UI chips

    build_feature_explanation(emotion, confidence, features, top_emotions)
        → str    complete explanation text combining feature evidence
                 (primary) and probability-distribution reasoning (secondary)
"""

from __future__ import annotations


# ── Which boolean flags are relevant to each predicted emotion ─────────────
#
# For each emotion the list covers:
#   • Distinctive positive indicators  (features strongly associated with it)
#   • Neutral / resting indicators     (features consistent with it, e.g. relaxed
#                                       brows for happy, neutral mouth for neutral)
#
# Only flags appearing in this list AND measured True will be shown.
# Order matters: most diagnostically important listed first.
_EMOTION_RELEVANT: dict[str, list[str]] = {
    "happy": [
        "smile_strong",          # broad, clearly raised corners
        "smile",                 # raised corners
        "slight_smile",          # mildly raised corners
        "mouth_wide",            # widened mouth span
        "eyes_slightly_narrow",  # cheek raise compresses lower lids
        "eyes_relaxed",          # open, relaxed eyelids — consistent with joy
        "brows_neutral",         # no tension — brows at rest
        "brows_relaxed",         # no contraction in brow region
        "mouth_closed_relaxed",  # lips together without tension
    ],
    "angry": [
        "brows_lowered",
        "brows_contracted",
        "eyes_narrowed",
        "eyes_slightly_narrow",
        "mouth_compressed",
        "mouth_corners_down",
    ],
    "sad": [
        "brow_inner_raised",
        "mouth_corners_down",
        "brows_lowered",
        "eyes_slightly_narrow",
        "eyes_narrowed",
        "mouth_closed_relaxed",
    ],
    "surprise": [
        "brows_raised",
        "brows_slightly_raised",
        "eyes_wide_open",
        "eyes_widened",
        "mouth_open",
        "mouth_slightly_open",
    ],
    "fear": [
        "brows_raised",
        "brows_slightly_raised",
        "brow_inner_raised",
        "brows_contracted",
        "eyes_wide_open",
        "eyes_widened",
        "mouth_slightly_open",
        "mouth_open",
    ],
    "disgust": [
        "brows_lowered",
        "brows_contracted",
        "eyes_slightly_narrow",
        "mouth_corners_down",
        "mouth_compressed",
        "eyes_narrowed",
    ],
    "neutral": [
        "brows_neutral",
        "brows_relaxed",
        "eyes_relaxed",
        "mouth_neutral",
        "mouth_closed_relaxed",
    ],
}


# ── Human-readable chip label for each boolean flag ────────────────────────
_FEATURE_LABELS: dict[str, str] = {
    # Eyebrow height
    "brows_raised":          "Raised eyebrows",
    "brows_slightly_raised": "Slightly raised eyebrows",
    "brows_neutral":         "Balanced eyebrow position",
    "brows_lowered":         "Lowered eyebrows",
    # Brow contraction
    "brows_contracted":      "Eyebrows drawn together",
    "brows_relaxed":         "Relaxed, non-furrowed eyebrows",
    "brow_inner_raised":     "Inner eyebrow corners raised (oblique shape)",
    # Eye openness
    "eyes_wide_open":        "Eyes opened very wide",
    "eyes_widened":          "Noticeably widened eyes",
    "eyes_relaxed":          "Relaxed eye openness",
    "eyes_slightly_narrow":  "Slightly narrowed eyes",
    "eyes_narrowed":         "Distinctly narrowed eyes",
    # Mouth corners
    "smile_strong":          "Clearly raised mouth corners (broad smile)",
    "smile":                 "Raised mouth corners (smile)",
    "slight_smile":          "Slightly upturned mouth corners",
    "mouth_neutral":         "Neutral mouth position",
    "mouth_corners_down":    "Downturned mouth corners",
    # Mouth openness
    "mouth_open":            "Open mouth",
    "mouth_slightly_open":   "Slightly parted lips",
    "mouth_closed_relaxed":  "Relaxed, closed lips",
    "mouth_compressed":      "Compressed / tightly closed lips",
    # Mouth width
    "mouth_wide":            "Widened mouth",
}


# ── Educational tooltip shown on chip hover ────────────────────────────────
_FEATURE_TIPS: dict[str, str] = {
    "brows_raised": (
        "The brow centres are measurably higher above the eye centres than "
        "at rest — frontalis muscle activity detected."
    ),
    "brows_slightly_raised": (
        "The brow centres are slightly elevated above the resting position — "
        "mild frontalis activity, consistent with alertness or mild surprise."
    ),
    "brows_neutral": (
        "The brow centres are at a balanced, resting height relative to the "
        "eye centres — no upward or downward pull detected."
    ),
    "brows_lowered": (
        "The brow centres are measurably closer to the eye centres than at "
        "rest — downward brow pull (depressor supercilii / corrugator)."
    ),
    "brows_contracted": (
        "The inner brow corners are unusually close together, suggesting "
        "corrugator supercilii muscle contraction."
    ),
    "brows_relaxed": (
        "The inner brow corners show no contraction — the brow region "
        "appears at rest with no visible furrowing."
    ),
    "brow_inner_raised": (
        "Both inner brow corners sit higher than their outer counterparts, "
        "forming an oblique (AU1) raise typical of sadness or fear."
    ),
    "eyes_wide_open": (
        "Eye Aspect Ratio is substantially above neutral — maximal eyelid "
        "retraction consistent with strong arousal (surprise / fear)."
    ),
    "eyes_widened": (
        "Eye Aspect Ratio is above the neutral range, indicating levator "
        "palpebrae superioris activity (eyelid retraction)."
    ),
    "eyes_relaxed": (
        "Eye Aspect Ratio is within the normal resting range — neither "
        "widened nor narrowed.  This is consistent with a calm or happy state."
    ),
    "eyes_slightly_narrow": (
        "Eye Aspect Ratio is slightly below neutral — mild lid constriction, "
        "often from cheek raising (happiness) or mild tension."
    ),
    "eyes_narrowed": (
        "Eye Aspect Ratio is distinctly below neutral — strong orbicularis "
        "oculi contraction producing marked lid constriction."
    ),
    "smile_strong": (
        "Mouth corners sit well above the horizontal midline of the lips — "
        "strong zygomaticus major activity (AU12), indicating a broad smile."
    ),
    "smile": (
        "Mouth corners sit noticeably above the horizontal midline — "
        "zygomaticus major activity (AU12) detected."
    ),
    "slight_smile": (
        "Mouth corners show a mild upward deviation above the lip midline — "
        "a subtle or partial smile (AU12 at low intensity)."
    ),
    "mouth_neutral": (
        "Mouth corners are at the lip midline — neither upturned nor "
        "downturned.  Consistent with a neutral or resting expression."
    ),
    "mouth_corners_down": (
        "Mouth corners fall below the lip midline — depressor anguli oris "
        "activity (AU15), typical of sadness or disgust."
    ),
    "mouth_open": (
        "Mouth Aspect Ratio is well above neutral — the jaw is substantially "
        "lowered (AU26/AU27), consistent with surprise or fear."
    ),
    "mouth_slightly_open": (
        "Mouth Aspect Ratio indicates partial jaw opening — lips are parted "
        "but not widely open."
    ),
    "mouth_closed_relaxed": (
        "Lips are together without compression — Mouth Aspect Ratio and "
        "corner angle indicate a relaxed, neutral lip posture."
    ),
    "mouth_compressed": (
        "Mouth Aspect Ratio is below neutral with a narrow horizontal span "
        "— orbicularis oris contraction creating pursed or pressed lips."
    ),
    "mouth_wide": (
        "The mouth horizontal span relative to face width is above the "
        "neutral range, consistent with a broad expression or smile."
    ),
}


# ── Public functions ───────────────────────────────────────────────────────

def build_feature_signals(
    features: dict,
    emotion: str,
) -> list[tuple[str, str]]:
    """
    Return a list of (label, tooltip) tuples for features that are:
        1. Measured True in `features` (set by FacialFeatureExtractor)
        2. Listed as relevant for the predicted `emotion`

    Returns [] if no relevant features were detected or features is empty.
    """
    if not features:
        return []
    relevant = _EMOTION_RELEVANT.get((emotion or "").lower(), [])
    signals: list[tuple[str, str]] = []
    for key in relevant:
        if features.get(key) is True:
            label = _FEATURE_LABELS.get(key)
            tip   = _FEATURE_TIPS.get(key, "")
            if label:
                signals.append((label, tip))
    return signals


def build_feature_explanation(
    emotion: str,
    confidence: float,
    features: dict,
    top_emotions: list,
) -> str:
    """
    Build a complete natural-language explanation combining:
        1. Primary   — feature-based evidence from measured landmarks.
        2. Secondary — probability-distribution reasoning.

    Parameters:
        emotion       — predicted emotion label (e.g. "happy")
        confidence    — prediction confidence in [0, 1]
        features      — dict from FacialFeatureExtractor.extract()
        top_emotions  — list of {"name": str, "score": float} dicts,
                        top-3 probabilities from the model

    Always returns a non-empty string.
    """
    emotion_l = (emotion or "unknown").lower()
    conf_pct  = f"{confidence * 100:.0f}%"

    if confidence >= 0.75:
        conf_desc = "high"
    elif confidence >= 0.50:
        conf_desc = "moderate"
    else:
        conf_desc = "low"

    # ── Sentence 1: classification summary ────────────────────────────
    s1 = (
        f'The model predicted "{emotion_l}" with {conf_desc} confidence '
        f"({conf_pct})."
    )

    # ── Sentence 2: feature evidence from landmarks ────────────────────
    signal_pairs  = build_feature_signals(features, emotion_l)
    signal_labels = [label for label, _ in signal_pairs]

    if signal_labels:
        # Format as a natural list
        if len(signal_labels) == 1:
            feature_list = signal_labels[0].lower()
            s2 = (
                f"The measured facial landmarks show {feature_list}. "
                f"Only one strong facial signal was detected, "
                f"so the explanation is limited."
            )
        elif len(signal_labels) == 2:
            feature_list = (
                f"{signal_labels[0].lower()} and {signal_labels[1].lower()}"
            )
            s2 = (
                f"This prediction is supported by {feature_list} "
                f"detected in the facial landmarks."
            )
        else:
            joined = (
                ", ".join(s.lower() for s in signal_labels[:-1])
                + f", and {signal_labels[-1].lower()}"
            )
            s2 = (
                f"This prediction is supported by {joined} "
                f"measured in the facial landmarks."
            )
    elif features:
        # Features extracted but none matched the emotion's profile.
        s2 = (
            "No individual landmark feature clearly aligned with the "
            "predicted emotion — the classification was driven by the "
            "overall facial geometry rather than a single dominant signal."
        )
    else:
        # MediaPipe unavailable.
        s2 = (
            "Facial landmark extraction was unavailable, so this "
            "explanation is based on the probability distribution only."
        )

    # ── Sentence 3: probability distribution reasoning ────────────────
    s3 = _distribution_reasoning(emotion_l, confidence, top_emotions)

    parts = [s1, s2, s3]
    return " ".join(p for p in parts if p).strip()


# ── Internal helpers ───────────────────────────────────────────────────────

def _distribution_reasoning(
    emotion_l: str,
    confidence: float,
    top_emotions: list,
) -> str:
    top1 = top_emotions[0] if len(top_emotions) > 0 else None
    top2 = top_emotions[1] if len(top_emotions) > 1 else None
    top3 = top_emotions[2] if len(top_emotions) > 2 else None

    if not (top1 and top2):
        return ""

    top1_score = float(top1.get("score", confidence))
    top2_score = float(top2.get("score", 0.0))
    top3_score = float(top3.get("score", 0.0)) if top3 else 0.0
    gap        = top1_score - top2_score

    three_way = (
        top3 is not None
        and top2_score >= top1_score - 0.10
        and top3_score >= top1_score - 0.15
    )
    if three_way:
        return (
            f"The probability distribution shows uncertainty: "
            f'"{top1["name"]}" ({top1_score * 100:.0f}%), '
            f'"{top2["name"]}" ({top2_score * 100:.0f}%), and '
            f'"{top3["name"]}" ({top3_score * 100:.0f}%) all received '
            f"comparable scores."
        )
    if gap < 0.10:
        return (
            f'The small gap between "{emotion_l}" ({top1_score * 100:.0f}%) '
            f'and "{top2["name"]}" ({top2_score * 100:.0f}%) indicates '
            f"some ambiguity in the overall expression signal."
        )
    if gap > 0.25:
        return (
            f'"{emotion_l.capitalize()}" ({top1_score * 100:.0f}%) clearly '
            f'dominated over "{top2["name"]}" ({top2_score * 100:.0f}%), '
            f"supporting a reliable prediction."
        )
    return (
        f'"{top2["name"].capitalize()}" ({top2_score * 100:.0f}%) was the '
        f"next-ranked emotion, which the model considered but ranked "
        f"below {emotion_l}."
    )
