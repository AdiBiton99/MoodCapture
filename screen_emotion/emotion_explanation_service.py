"""
emotion_explanation_service.py — Explainable AI Emotion Assistant.

Consumes the structured output of `EmotionAnalysisService.analyze(...)` and
produces a short, natural-language explanation of WHY the final emotion was
selected. The explanation is grounded in the actual prediction numbers — no
hallucinated emotions or invented facts.

Two execution modes:
    1. OpenAI mode  — calls the chat API via `OpenAIService` with a strict
                      system prompt and a compact JSON payload built from
                      the analysis result.
    2. Local mode   — deterministic template that mentions the dominant
                      emotion, runner-up, confidence band, and per-face
                      disagreement if any. Used whenever OpenAI is
                      unavailable or fails.

This module is intentionally synchronous; the caller is expected to invoke
`explain(...)` from a background thread (e.g. a QThread) so the UI stays
responsive.

Typical use:
    svc = EmotionExplanationService(openai_service=OpenAIService())
    text = svc.explain(analysis_result)
"""

from __future__ import annotations

import logging
from typing import Any, Optional


_LOGGER = logging.getLogger("emotion_explanation_service")


# Confidence bands used for both the OpenAI payload hint and the local template.
_HIGH_CONF = 0.75
_LOW_CONF  = 0.50


SYSTEM_PROMPT = (
    "You are a professional emotion analysis assistant that explains the "
    "predictions of a facial-emotion-inference model.\n"
    "\n"
    "You will receive a JSON object describing the model's output. Your job "
    "is to write a short, clear, academic-sounding explanation (2-4 sentences) "
    "of WHY the final emotion was selected.\n"
    "\n"
    "STRICT RULES — you MUST obey all of them:\n"
    "  * Stay grounded in the JSON. Do not invent emotions that are not present.\n"
    "  * Do not claim certainty when confidence is low; explicitly acknowledge "
    "    uncertainty when the confidence is below 0.50, and mention the "
    "    competing emotion.\n"
    "  * Never diagnose mental health, mood disorders, or personality traits.\n"
    "  * Never describe what the person is thinking or feeling internally. "
    "    Instead, say: \"the model detected signals associated with X\" or "
    "    \"the faces showed features consistent with X\".\n"
    "  * If multiple faces disagree, mention the disagreement and note that "
    "    the final emotion reflects the dominant pattern.\n"
    "  * Do not use bullet points or markdown. Plain prose only.\n"
    "  * Do not greet the user or add closing remarks. Output only the "
    "    explanation itself.\n"
)


SYSTEM_PROMPT_FACE = (
    "You are a professional facial-expression analyst writing an EDUCATIONAL "
    "explanation for an academic emotion-recognition project.\n"
    "\n"
    "You will receive:\n"
    "  1. An IMAGE of a single cropped face from the screenshot.\n"
    "  2. A JSON object with the model's prediction for THIS face: its "
    "predicted emotion, confidence, top emotion probabilities, and context "
    "about the overall aggregated result.\n"
    "\n"
    "Your job: write 3-5 sentences explaining WHY the face you see was "
    "classified as its predicted emotion, by DESCRIBING THE VISIBLE FACIAL "
    "FEATURES in the image and connecting them to the emotion. The reader "
    "should LEARN from this explanation — they should walk away understanding "
    "which features are associated with the predicted emotion.\n"
    "\n"
    "What to describe (use what you actually see):\n"
    "  * Mouth shape — smile width, lip position, teeth visibility, corner "
    "    direction (upturned / downturned / neutral).\n"
    "  * Eyes — openness, narrowing, crow's-feet wrinkles, gaze direction.\n"
    "  * Eyebrows — raised, lowered, furrowed, neutral.\n"
    "  * Cheeks — raised (cheek raiser), flat, sunken.\n"
    "  * Forehead — wrinkled, smooth.\n"
    "  * Overall facial muscle tone — relaxed vs. tense.\n"
    "\n"
    "STRICT RULES — you MUST obey all of them:\n"
    "  * Refer to the subject as \"this face\" or \"the face\". DO NOT use "
    "    labels like \"Face 1\", \"Face N\", \"face_index\", numbering of "
    "    any kind, or technical identifiers.\n"
    "  * Describe ONLY features you can actually see in the image. Do not "
    "    invent details. If a feature is hidden or unclear, do not mention it.\n"
    "  * Connect the visible features to the predicted emotion explicitly. "
    "    Example: \"The upturned corners of the mouth and raised cheeks are "
    "    classic indicators of happiness, which explains the model's high "
    "    confidence.\"\n"
    "  * If the face's confidence is below 0.50, point out the visual "
    "    ambiguity and mention the strongest competing emotion from the JSON.\n"
    "  * If this face's emotion DIFFERS from the overall final_emotion, "
    "    briefly note that this face stood out from the rest of the group.\n"
    "  * Do NOT diagnose mental health, mood disorders, or personality.\n"
    "  * Do NOT describe what the person is thinking or feeling internally. "
    "    Describe the EXPRESSION MECHANICS only.\n"
    "  * Do NOT identify the person, guess age, gender, ethnicity, or any "
    "    personal attribute.\n"
    "  * Plain prose, no markdown, no bullets, no greetings, no closing.\n"
    "  * 3-5 sentences total.\n"
)


class EmotionExplanationService:
    """
    Generates a natural-language explanation for an emotion analysis result.

    The service is safe to construct without an OpenAI key — in that case
    it operates in local-only mode and `explain(...)` will always return
    a deterministic template-based explanation.
    """

    def __init__(self, openai_service: Any = None) -> None:
        """
        Parameters:
            openai_service — an OpenAIService instance (or anything exposing
                             `is_available()` and `generate(system, user)`).
                             May be None — local fallback is always used.
        """
        self._openai = openai_service

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(self, analysis_result: dict) -> tuple[str, bool, list]:
        """
        Build an explanation from an analysis-service result dict.

        Returns (text, visual_signals_available, signals).
        The overall explanation never sends a face image, so
        visual_signals_available is always False and signals is [].

        Always returns a non-empty string. Order of attempts:
            1. OpenAI (if service is available)
            2. Local deterministic template
        """
        payload = self._build_payload(analysis_result)

        # Empty-result short-circuit — don't call the API for "no faces".
        if payload["faces_count"] == 0:
            return (
                "No faces were detected in the captured image, so no "
                "emotion explanation can be produced for this frame.",
                False,
                [],
            )

        if self._openai is not None and self._openai.is_available():
            try:
                text = self._openai.generate(SYSTEM_PROMPT, payload)
                if text:
                    return text, False, []
            except Exception as exc:
                _LOGGER.warning("OpenAI explanation failed; falling back: %s", exc)

        return self._local_fallback(payload), False, []

    def explain_face(self, face_data: dict, context: dict) -> tuple[str, bool, list]:
        """
        Build an explanation focused on a SINGLE face out of the analysis.

        Returns (text, visual_signals_available, signals).

        Priority order:
            1. MediaPipe landmark features (if face_data["features"] is non-empty)
               → feature-based explanation grounded in measured geometry
               → visual_signals_available = True
               → signals = list of (label, tooltip) tuples for UI chips
            2. OpenAI Vision (if available + face_image present)
               → visual_signals_available = (image was sent)
               → signals = []  (plain prose from GPT, no chip list)
            3. Local distribution-based fallback
               → visual_signals_available = False
               → signals = []

        Parameters:
            face_data — one entry from `analysis_result["faces"]`. Required
                        fields: `emotion`, `confidence`, `all_emotions`.
                        Optional: `face_image` (np.ndarray), `features` (dict).
            context   — overall result context.
        """
        features = face_data.get("features") or {}

        # ── Path 1: MediaPipe landmark features ──────────────────────────
        if features:
            from screen_emotion.feature_explainer import (
                build_feature_explanation,
                build_feature_signals,
            )
            payload      = self._build_face_payload(face_data, context)
            top_emotions = payload["face"].get("top_emotions") or []
            emotion      = face_data.get("emotion", "unknown")
            confidence   = float(face_data.get("confidence", 0.0))

            text    = build_feature_explanation(emotion, confidence, features, top_emotions)
            signals = build_feature_signals(features, (emotion or "").lower())
            # Only mark visual signals as available when we actually have chips to show.
            # If features were extracted but none matched the predicted emotion, the
            # explanation text already explains this — no chip section should appear.
            return text, bool(signals), signals

        # ── Path 2: OpenAI Vision ─────────────────────────────────────────
        payload     = self._build_face_payload(face_data, context)
        image_bytes = self._encode_face_image_png(face_data.get("face_image"))

        if self._openai is not None and self._openai.is_available():
            try:
                text = self._openai.generate(
                    SYSTEM_PROMPT_FACE,
                    payload,
                    image_bytes=image_bytes,
                    max_tokens=400,
                )
                if text:
                    visual_signals_available = image_bytes is not None
                    return text, visual_signals_available, []
            except Exception as exc:
                _LOGGER.warning("OpenAI face explanation failed; falling back: %s", exc)

        # ── Path 3: Local distribution fallback ───────────────────────────
        return self._local_fallback_face(payload), False, []

    # ------------------------------------------------------------------
    # Image encoding helper
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_face_image_png(face_image) -> Optional[bytes]:
        """
        Convert a numpy face crop (RGB) to PNG bytes suitable for the
        OpenAI vision API. Returns None on any failure or if the input is
        unusable. The image is downscaled to a max side of 384 px to keep
        token cost minimal — that's still plenty of detail for a face crop.
        """
        if face_image is None:
            return None
        try:
            import cv2
            import numpy as np
        except Exception:
            return None
        try:
            img = face_image
            if not isinstance(img, np.ndarray):
                return None
            if img.size == 0:
                return None
            if img.dtype != np.uint8:
                img = (img * 255).clip(0, 255).astype(np.uint8)
            h, w = img.shape[:2]
            max_dim = 384
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                img = cv2.resize(
                    img, (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            # The pipeline stores faces in RGB; cv2.imencode wants BGR.
            if img.ndim == 3 and img.shape[2] == 3:
                to_encode = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif img.ndim == 2:
                to_encode = img
            else:
                return None
            ok, buf = cv2.imencode(".png", to_encode)
            if not ok:
                return None
            return bytes(buf)
        except Exception as exc:
            _LOGGER.debug("Face image PNG encoding failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Payload construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_payload(analysis_result: dict) -> dict:
        """
        Reduce the full analysis dict to a compact, OpenAI-friendly payload.

        Only emotion-related fields are kept — no images, no bounding boxes,
        no PII. Per-face probabilities are trimmed to the top 3 to keep the
        prompt cheap and focused.
        """
        faces_in = analysis_result.get("faces") or []
        faces_out = []
        for idx, face in enumerate(faces_in):
            all_emotions = face.get("all_emotions") or {}
            top3 = sorted(all_emotions.items(), key=lambda kv: kv[1], reverse=True)[:3]
            faces_out.append({
                "face_index": idx,
                "emotion":    face.get("emotion", "unknown"),
                "confidence": round(float(face.get("confidence", 0.0)), 4),
                "top_emotions": [
                    {"name": name, "score": round(float(score), 4)}
                    for name, score in top3
                ],
            })

        final_emotion    = analysis_result.get("final_emotion")
        final_confidence = float(analysis_result.get("confidence", 0.0) or 0.0)

        return {
            "final_emotion":   final_emotion,
            "final_confidence": round(final_confidence, 4),
            "confidence_band":
                "high" if final_confidence >= _HIGH_CONF
                else "low" if final_confidence < _LOW_CONF
                else "medium",
            "faces_count": len(faces_out),
            "faces":       faces_out,
        }

    # ------------------------------------------------------------------
    # Local fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _local_fallback(payload: dict) -> str:
        """
        Distribution-aware explanation for the overall analysis result.
        Derives reasoning from probability scores — never invents visual features.
        """
        final_emotion = payload.get("final_emotion") or "unknown"
        final_conf    = float(payload.get("final_confidence", 0.0))
        band          = payload.get("confidence_band", "medium")
        faces         = payload.get("faces") or []
        n             = len(faces)

        if n == 0 or not final_emotion:
            return (
                "No faces were detected in the captured image, so no "
                "emotion explanation can be produced for this frame."
            )

        # --- Extract top-3 from the dominant face's distribution ------
        dominant_face = next(
            (f for f in faces if f.get("emotion") == final_emotion),
            faces[0],
        )
        top_list   = dominant_face.get("top_emotions") or []
        top1       = top_list[0] if len(top_list) > 0 else None
        top2       = top_list[1] if len(top_list) > 1 else None
        top3       = top_list[2] if len(top_list) > 2 else None
        top1_score = float(top1["score"]) if top1 else final_conf
        top2_score = float(top2["score"]) if top2 else 0.0
        top3_score = float(top3["score"]) if top3 else 0.0
        gap_1_2    = top1_score - top2_score
        conf_pct   = f"{final_conf * 100:.0f}%"

        # --- Sentence 1: result + confidence level --------------------
        if band == "high":
            s_result = (
                f'The model selected "{final_emotion}" as the dominant emotion '
                f"with high confidence ({conf_pct})."
            )
        elif band == "medium":
            s_result = (
                f'The model selected "{final_emotion}" with moderate confidence '
                f"({conf_pct}), indicating a recognisable but not overwhelming signal."
            )
        else:
            s_result = (
                f'The model assigned "{final_emotion}" with low confidence '
                f"({conf_pct}), so this prediction should be treated with caution."
            )

        # --- Sentence 2: distribution reasoning ----------------------
        s_dist = ""
        if top2:
            three_way = (
                top3 is not None
                and top2_score >= top1_score - 0.10
                and top3_score >= top1_score - 0.15
            )
            if three_way:
                s_dist = (
                    f'The model shows uncertainty: "{top1["name"]}" ({top1_score * 100:.0f}%), '
                    f'"{top2["name"]}" ({top2_score * 100:.0f}%), and '
                    f'"{top3["name"]}" ({top3_score * 100:.0f}%) all received comparable scores, '
                    f"indicating the expression carries mixed signals."
                )
            elif gap_1_2 < 0.10:
                s_dist = (
                    f'The small gap between "{final_emotion}" ({top1_score * 100:.0f}%) and '
                    f'"{top2["name"]}" ({top2_score * 100:.0f}%) indicates ambiguity '
                    f"in the facial expression signal."
                )
            elif gap_1_2 > 0.25:
                s_dist = (
                    f'The prediction is considered reliable because "{final_emotion}" '
                    f"({top1_score * 100:.0f}%) clearly exceeds all alternatives — "
                    f'the next-highest emotion, "{top2["name"]}", scored only '
                    f"{top2_score * 100:.0f}%. This large gap supports a strong prediction."
                )
            else:
                s_dist = (
                    f'"{top2["name"]}" ({top2_score * 100:.0f}%) was the second-ranked '
                    f'emotion, which the model considered but ranked below "{final_emotion}".'
                )

        # --- Sentence 3: multi-face agreement ------------------------
        s_faces = ""
        if n > 1:
            votes    = [f.get("emotion") for f in faces]
            winners  = [v for v in votes if v == final_emotion]
            agree_n  = len(winners)
            disagree = n - agree_n
            if agree_n == n:
                s_faces = (
                    f'All {n} detected faces agreed on "{final_emotion}", '
                    f"reinforcing the final decision."
                )
            elif agree_n >= n / 2:
                s_faces = (
                    f"Of the {n} detected faces, {agree_n} aligned with "
                    f'"{final_emotion}" while {disagree} showed different signals; '
                    f"the final emotion reflects the dominant pattern."
                )
            else:
                s_faces = (
                    f"The {n} detected faces produced mixed signals; "
                    f'"{final_emotion}" was chosen as the most frequently observed emotion.'
                )

        parts = [s_result, s_dist, s_faces]
        return " ".join(p for p in parts if p).strip()

    # ------------------------------------------------------------------
    # Per-face payload + fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _build_face_payload(face_data: dict, context: dict) -> dict:
        """
        Reduce a single face entry to a compact, OpenAI-friendly payload.

        Only emotion-related fields are kept — no images, no bounding boxes.
        Probabilities are trimmed to the top 3.
        """
        all_emotions = face_data.get("all_emotions") or {}
        top3 = sorted(all_emotions.items(), key=lambda kv: kv[1], reverse=True)[:3]
        confidence = float(face_data.get("confidence", 0.0) or 0.0)

        face_index = context.get("face_index")
        if face_index is None:
            face_index = face_data.get("face_index", 0)

        face_block = {
            "face_index":  int(face_index),
            "emotion":     face_data.get("emotion", "unknown"),
            "confidence":  round(confidence, 4),
            "top_emotions": [
                {"name": name, "score": round(float(score), 4)}
                for name, score in top3
            ],
        }

        final_emotion    = context.get("final_emotion")
        final_confidence = float(context.get("final_confidence", 0.0) or 0.0)
        faces_count      = int(context.get("faces_count", 1) or 1)

        return {
            "face": face_block,
            "context": {
                "final_emotion":    final_emotion,
                "final_confidence": round(final_confidence, 4),
                "faces_count":      faces_count,
                "agrees_with_overall":
                    face_block["emotion"] == final_emotion,
                "confidence_band":
                    "high" if confidence >= _HIGH_CONF
                    else "low" if confidence < _LOW_CONF
                    else "medium",
            },
        }

    @staticmethod
    def _local_fallback_face(payload: dict) -> str:
        """
        Distribution-aware fallback for a single-face explanation.
        Used when OpenAI Vision is unavailable or fails.
        Derives reasoning from probability scores only — never invents
        or infers any visual facial features.
        """
        face    = payload.get("face") or {}
        context = payload.get("context") or {}

        emotion  = (face.get("emotion") or "unknown").lower()
        conf     = float(face.get("confidence", 0.0))
        conf_pct = f"{conf * 100:.0f}%"
        band     = context.get("confidence_band", "medium")

        top_list   = face.get("top_emotions") or []
        top1       = top_list[0] if len(top_list) > 0 else None
        top2       = top_list[1] if len(top_list) > 1 else None
        top3       = top_list[2] if len(top_list) > 2 else None
        top1_score = float(top1["score"]) if top1 else conf
        top2_score = float(top2["score"]) if top2 else 0.0
        top3_score = float(top3["score"]) if top3 else 0.0
        gap_1_2    = top1_score - top2_score

        # --- Sentence 1: result + confidence level --------------------
        if band == "high":
            s_result = (
                f'The model classified this face as "{emotion}" '
                f"with high confidence ({conf_pct})."
            )
        elif band == "medium":
            s_result = (
                f'The model classified this face as "{emotion}" '
                f"with moderate confidence ({conf_pct})."
            )
        else:
            s_result = (
                f'The model classified this face as "{emotion}" with a low '
                f"confidence of {conf_pct}, so this prediction should be treated as uncertain."
            )

        # --- Sentence 2: distribution reasoning ----------------------
        s_dist = ""
        if top2:
            three_way = (
                top3 is not None
                and top2_score >= top1_score - 0.10
                and top3_score >= top1_score - 0.15
            )
            if three_way:
                s_dist = (
                    f"The model is uncertain because several emotions received similar "
                    f'scores: "{top1["name"]}" at {top1_score * 100:.0f}%, '
                    f'"{top2["name"]}" at {top2_score * 100:.0f}%, and '
                    f'"{top3["name"]}" at {top3_score * 100:.0f}%. '
                    f"This spread indicates the facial expression carries ambiguous signals."
                )
            elif gap_1_2 < 0.10:
                s_dist = (
                    f'The small difference between "{emotion}" ({top1_score * 100:.0f}%) and '
                    f'"{top2["name"]}" ({top2_score * 100:.0f}%) indicates ambiguity '
                    f"in the expression signal."
                )
            elif gap_1_2 > 0.25:
                s_dist = (
                    f'The large gap between "{emotion}" ({top1_score * 100:.0f}%) and '
                    f'"{top2["name"]}" ({top2_score * 100:.0f}%) suggests a clear '
                    f"emotional signal, supporting a reliable prediction."
                )
            else:
                s_dist = (
                    f'"{top2["name"]}" ({top2_score * 100:.0f}%) was the second-ranked '
                    f"emotion, which the model considered but did not select."
                )

        # --- Sentence 3: offline note ---------------------------------
        s_offline = (
            "Because OpenAI Vision is unavailable, this explanation is based "
            "solely on the model's probability distribution, not on visible facial features."
        )

        # --- Sentence 4: group context --------------------------------
        s_group = ""
        final_emotion = context.get("final_emotion")
        agrees        = context.get("agrees_with_overall", True)
        faces_count   = int(context.get("faces_count", 1) or 1)
        if faces_count > 1 and final_emotion and not agrees:
            s_group = (
                f'Note that the overall result for the image was "{final_emotion}", '
                f"so this face showed a different emotional signal from the rest of the group."
            )

        parts = [s_result, s_dist, s_offline, s_group]
        return " ".join(p for p in parts if p).strip()
