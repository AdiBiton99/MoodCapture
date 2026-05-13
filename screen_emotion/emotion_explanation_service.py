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


# Used in the local fallback (no image, no API): each emotion's TYPICAL
# facial features. Keeps the fallback educational rather than abstract.
EMOTION_FEATURE_HINTS = {
    "happy":    "an upturned mouth (smile), raised cheeks, and often crinkling "
                "around the eyes",
    "sad":      "downturned mouth corners, drooping eyelids, and inner brows "
                "drawn up or together",
    "angry":    "lowered, furrowed brows, tightened lips, and tense facial muscles",
    "neutral":  "relaxed facial muscles with no strong expression in the mouth, "
                "brows, or eyes",
    "surprise": "raised eyebrows, widened eyes, and a parted (open) mouth",
    "fear":     "raised and drawn-together eyebrows, widened eyes, and a "
                "horizontally stretched mouth",
    "disgust":  "a wrinkled nose, raised upper lip, and lowered brows",
}


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

    def explain(self, analysis_result: dict) -> str:
        """
        Build an explanation from an analysis-service result dict.

        Always returns a non-empty string. Order of attempts:
            1. OpenAI (if service is available)
            2. Local deterministic template

        The input dict is expected to follow the shape produced by
        EmotionAnalysisService.analyze(...).
        """
        payload = self._build_payload(analysis_result)

        # Empty-result short-circuit — don't call the API for "no faces".
        if payload["faces_count"] == 0:
            return (
                "No faces were detected in the captured image, so no "
                "emotion explanation can be produced for this frame."
            )

        if self._openai is not None and self._openai.is_available():
            try:
                text = self._openai.generate(SYSTEM_PROMPT, payload)
                if text:
                    return text
            except Exception as exc:
                _LOGGER.warning("OpenAI explanation failed; falling back: %s", exc)

        return self._local_fallback(payload)

    def explain_face(self, face_data: dict, context: dict) -> str:
        """
        Build an explanation focused on a SINGLE face out of the analysis.

        When an OpenAI vision-capable model is available, the CROPPED FACE
        IMAGE from `face_data["face_image"]` is sent alongside the JSON so
        the model can describe the actual visible features (mouth, eyes,
        eyebrows, etc.) that support the prediction. This makes the
        explanation educational and concrete rather than abstract.

        Parameters:
            face_data — one entry from `analysis_result["faces"]`. Required
                        fields: `emotion`, `confidence`, `all_emotions`.
                        Optional: `face_image` (np.ndarray) — enables vision.
            context   — overall result context, typically:
                            {
                              "final_emotion":    str,
                              "final_confidence": float,
                              "faces_count":      int,
                              "face_index":       int,
                            }

        Always returns a non-empty string. Order of attempts:
            1. OpenAI with image (if available + face_image present)
            2. OpenAI text-only (if image encoding failed)
            3. Local deterministic template (educational with feature hints)
        """
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
                    return text
            except Exception as exc:
                _LOGGER.warning("OpenAI face explanation failed; falling back: %s", exc)

        return self._local_fallback_face(payload)

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
        Deterministic, hand-written explanation. Always grounded in numbers
        from `payload`. Used whenever the OpenAI call is unavailable or fails.
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

        # --- per-face agreement ---------------------------------------
        votes = [f.get("emotion") for f in faces]
        winners = [v for v in votes if v == final_emotion]
        agreement = len(winners) / n if n else 0.0

        # --- runner-up emotion across the dominant face ---------------
        dominant_face = next(
            (f for f in faces if f.get("emotion") == final_emotion),
            faces[0],
        )
        top_list = dominant_face.get("top_emotions") or []
        runner_up: Optional[dict] = None
        for item in top_list:
            if item.get("name") != final_emotion:
                runner_up = item
                break

        # --- compose sentences ----------------------------------------
        conf_pct = f"{final_conf * 100:.0f}%"

        if band == "high":
            sentence_main = (
                f"The system selected \"{final_emotion}\" because the detected "
                f"face(s) showed signals consistent with {final_emotion} at a "
                f"high confidence of {conf_pct}."
            )
        elif band == "medium":
            sentence_main = (
                f"The system selected \"{final_emotion}\" with a moderate "
                f"confidence of {conf_pct}, indicating that the dominant "
                f"signal was {final_emotion} but not overwhelmingly so."
            )
        else:
            sentence_main = (
                f"The prediction \"{final_emotion}\" was assigned with a "
                f"relatively low confidence of {conf_pct}, so the result "
                f"should be treated as uncertain."
            )

        sentence_runner = ""
        if runner_up and runner_up.get("score", 0.0) > 0.05:
            sentence_runner = (
                f" The next strongest signal was \"{runner_up['name']}\" "
                f"at {runner_up['score'] * 100:.0f}%, which the model "
                f"considered but did not select."
            )

        sentence_faces = ""
        if n > 1:
            if agreement >= 0.99:
                sentence_faces = (
                    f" All {n} detected faces agreed on {final_emotion}, "
                    f"which reinforced the final decision."
                )
            elif agreement >= 0.5:
                disagreeing = n - len(winners)
                sentence_faces = (
                    f" Of the {n} detected faces, {len(winners)} aligned with "
                    f"{final_emotion} while {disagreeing} showed different "
                    f"signals; the final emotion reflects the dominant pattern."
                )
            else:
                sentence_faces = (
                    f" The {n} detected faces produced mixed signals, and "
                    f"\"{final_emotion}\" was chosen as the most frequently "
                    f"observed emotion across them."
                )

        return (sentence_main + sentence_runner + sentence_faces).strip()

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
        Deterministic, educational fallback for a single-face explanation.
        Used when OpenAI is unavailable or fails. Cannot see the actual
        face, so it instead describes the TYPICAL features associated with
        the predicted emotion — useful for the reader to learn from.
        """
        face       = payload.get("face") or {}
        context    = payload.get("context") or {}
        face_label = "This face"   # No "Face N" labels — see SYSTEM_PROMPT_FACE.

        emotion    = (face.get("emotion") or "unknown").lower()
        conf       = float(face.get("confidence", 0.0))
        conf_pct   = f"{conf * 100:.0f}%"
        band       = context.get("confidence_band", "medium")

        feature_hint = EMOTION_FEATURE_HINTS.get(emotion)

        top_emotions = face.get("top_emotions") or []
        runner_up: Optional[dict] = None
        for item in top_emotions:
            if item.get("name") != face.get("emotion"):
                runner_up = item
                break

        # --- main sentence: ties the predicted emotion to TYPICAL features ---
        if feature_hint:
            if band == "high":
                main = (
                    f"{face_label} was classified as \"{emotion}\" with a high "
                    f"confidence of {conf_pct}. {emotion.capitalize()} is "
                    f"typically expressed through {feature_hint}, and the "
                    f"model identified these features strongly in this face."
                )
            elif band == "medium":
                main = (
                    f"{face_label} was classified as \"{emotion}\" with a "
                    f"moderate confidence of {conf_pct}. {emotion.capitalize()} "
                    f"is typically expressed through {feature_hint}, and the "
                    f"model detected these features here, although not "
                    f"overwhelmingly."
                )
            else:
                main = (
                    f"{face_label} was classified as \"{emotion}\" with a "
                    f"relatively low confidence of {conf_pct}, so this "
                    f"prediction should be treated as uncertain. "
                    f"{emotion.capitalize()} is typically expressed through "
                    f"{feature_hint}, but in this face the signal was weak."
                )
        else:
            # Unknown emotion — fall back to a generic phrasing.
            main = (
                f"{face_label} was classified as \"{emotion}\" with a "
                f"confidence of {conf_pct}, based on signals the model "
                f"associates with that emotion."
            )

        runner_sentence = ""
        if runner_up and runner_up.get("score", 0.0) > 0.05:
            runner_name = runner_up["name"]
            runner_pct  = f"{runner_up['score'] * 100:.0f}%"
            runner_hint = EMOTION_FEATURE_HINTS.get(runner_name.lower())
            if runner_hint:
                runner_sentence = (
                    f" A secondary signal of \"{runner_name}\" "
                    f"({runner_pct}) was also detected — {runner_name} "
                    f"is associated with {runner_hint}, hinting that subtle "
                    f"elements of this expression were present too."
                )
            else:
                runner_sentence = (
                    f" The next strongest signal for this face was "
                    f"\"{runner_name}\" at {runner_pct}."
                )

        group_sentence = ""
        final_emotion = context.get("final_emotion")
        agrees        = context.get("agrees_with_overall", True)
        faces_count   = int(context.get("faces_count", 1) or 1)
        if faces_count > 1 and final_emotion and not agrees:
            group_sentence = (
                f" Note that the overall result for the image was "
                f"\"{final_emotion}\", so this face stood out from "
                f"the rest of the group."
            )

        return (main + runner_sentence + group_sentence).strip()
