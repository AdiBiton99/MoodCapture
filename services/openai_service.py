"""
openai_service.py — thin, fault-tolerant wrapper around the OpenAI Chat API.

Design goals:
    * NEVER crash the host app — every failure mode returns None and is logged.
    * Optional dependencies — `openai` and `python-dotenv` are loaded lazily.
      If either package is missing, `is_available()` returns False and the
      caller falls back to local logic.
    * Stateless — a single instance can be reused across requests; each call
      is independent and safe to run from a background thread.

Public API:
    svc = OpenAIService()
    if svc.is_available():
        text = svc.generate(system_prompt, user_payload)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional


_LOGGER = logging.getLogger("openai_service")

DEFAULT_MODEL       = "gpt-4o-mini"
DEFAULT_MAX_TOKENS  = 350
DEFAULT_TEMPERATURE = 0.3
DEFAULT_TIMEOUT_S   = 20.0


class OpenAIService:
    """
    Optional OpenAI client. Construct once at startup; reuse across calls.

    The constructor never raises:
        * If `python-dotenv` is installed, `.env` is loaded.
        * If `openai` is installed AND OPENAI_API_KEY is present, a client is built.
        * Otherwise the service is marked unavailable and `generate()` returns None.
    """

    def __init__(
        self,
        api_key:  Optional[str] = None,
        model:    Optional[str] = None,
        timeout:  float         = DEFAULT_TIMEOUT_S,
    ) -> None:
        self._client: Any  = None
        self._model:  str  = model or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL
        self._timeout      = timeout
        self._available    = False
        self._reason       = "not initialized"

        self._load_dotenv_if_present()

        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            self._reason = "OPENAI_API_KEY is not set"
            _LOGGER.info("OpenAIService disabled: %s. Local fallback will be used.", self._reason)
            return

        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:  # ImportError or anything weird
            self._reason = f"openai package not installed ({exc})"
            _LOGGER.info("OpenAIService disabled: %s. Local fallback will be used.", self._reason)
            return

        try:
            self._client    = OpenAI(api_key=key, timeout=timeout)
            self._available = True
            self._reason    = "ok"
            _LOGGER.info("OpenAIService initialized (model=%s).", self._model)
        except Exception as exc:
            self._reason = f"client init failed: {exc}"
            _LOGGER.warning("OpenAIService disabled: %s", self._reason)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """True iff OpenAI calls will be attempted (key + package + client present)."""
        return self._available

    def status(self) -> str:
        """Human-readable reason string — useful for logging / UI hints."""
        return self._reason

    def generate(
        self,
        system_prompt: str,
        user_payload:  Any,
        *,
        model:       Optional[str] = None,
        max_tokens:  int   = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        image_bytes: Optional[bytes] = None,
        image_mime:  str   = "image/png",
    ) -> Optional[str]:
        """
        Call the chat completions endpoint and return the assistant text.

        `user_payload` may be a string or a JSON-serializable object;
        objects are serialized with `json.dumps(...)` before being sent.

        If `image_bytes` is provided, the user message is sent as a multi-part
        content array (text + image) — requires a vision-capable model
        (e.g. gpt-4o-mini, gpt-4o). The image is base64-inlined.

        Returns None on any failure (network, auth, rate-limit, parse error).
        The caller is expected to provide a local fallback.
        """
        if not self._available or self._client is None:
            return None

        if not isinstance(user_payload, str):
            try:
                text_part = json.dumps(user_payload, ensure_ascii=False, default=str)
            except Exception as exc:
                _LOGGER.warning("Failed to serialize user payload: %s", exc)
                return None
        else:
            text_part = user_payload

        if image_bytes:
            try:
                import base64
                b64 = base64.b64encode(image_bytes).decode("ascii")
            except Exception as exc:
                _LOGGER.warning("Failed to base64-encode image: %s", exc)
                return None
            user_content: Any = [
                {"type": "text", "text": text_part},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image_mime};base64,{b64}",
                        "detail": "low",   # keeps token cost minimal
                    },
                },
            ]
        else:
            user_content = text_part

        try:
            response = self._client.chat.completions.create(
                model=model or self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_content},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            _LOGGER.warning("OpenAI request failed: %s", exc)
            return None

        try:
            text = response.choices[0].message.content
        except Exception as exc:
            _LOGGER.warning("OpenAI response had unexpected shape: %s", exc)
            return None

        if not text or not text.strip():
            _LOGGER.info("OpenAI returned an empty completion.")
            return None

        return text.strip()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _load_dotenv_if_present() -> None:
        """Best-effort .env loader — silent no-op if python-dotenv is missing."""
        try:
            from dotenv import load_dotenv  # type: ignore
        except Exception:
            return
        try:
            load_dotenv(override=False)
        except Exception as exc:
            _LOGGER.debug("dotenv load_dotenv() raised: %s", exc)
