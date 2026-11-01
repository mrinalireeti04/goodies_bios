"""Gemini API client wrapper for FORGE TRACE.

Encapsulates all communication with the Google Generative AI API, including
file uploads, content generation, retry / rate-limit handling, and response
parsing.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from backend.inference.ps3.config import ForgeTraceConfig
from backend.inference.ps3.exceptions import (
    FileUploadError,
    GeminiAPIError,
    GeminiRateLimitError,
    GeminiResponseParseError,
)
from backend.inference.ps3.logging_config import get_logger

logger = get_logger("gemini_client")

# Exceptions that justify automatic retry
_RETRYABLE = (GeminiRateLimitError,)


class GeminiClient:
    """Wrapper around the modern ``google-genai`` SDK.

    Responsibilities:
    * Configure the SDK client with API key.
    * Upload video files and wait for processing.
    * Send analysis requests with structured-JSON output.
    * Retry on transient / rate-limit errors.
    * Parse JSON from model responses.
    """

    def __init__(self, config: ForgeTraceConfig) -> None:
        self._config = config
        self._client = genai.Client(api_key=config.gemini_api_key)
        logger.info("GeminiClient initialised (model=%s)", config.gemini_model)

    # ------------------------------------------------------------------
    # File upload
    # ------------------------------------------------------------------

    def upload_video(
        self,
        video_path: str,
        mime_type: str = "video/mp4",
        poll_interval: float = 5.0,
        max_wait: float = 600.0,
    ) -> Any:
        """Upload a video file and block until processing completes.

        Returns the ``genai.types.File`` object.
        """
        logger.info("Uploading video segment")
        try:
            video_file = self._client.files.upload(
                file=video_path,
                config={"mime_type": mime_type},
            )
        except Exception as exc:
            logger.exception("Gemini upload failed for local segment %s", video_path)
            raise FileUploadError(
                f"Gemini video upload failed: {type(exc).__name__}"
            ) from exc

        try:
            waited = 0.0
            while video_file.state.name == "PROCESSING":
                if waited >= max_wait:
                    raise FileUploadError(
                        f"Gemini file processing timed out after {max_wait}s"
                    )
                logger.debug("Waiting for Gemini video processing (%.0fs elapsed)", waited)
                time.sleep(poll_interval)
                waited += poll_interval
                video_file = self._client.files.get(name=video_file.name)

            if video_file.state.name == "FAILED":
                raise FileUploadError("Gemini could not process the uploaded video")

            logger.info("Gemini video upload ready (state=%s)", video_file.state.name)
            return video_file
        except Exception:
            try:
                self._client.files.delete(name=video_file.name)
            except Exception:
                logger.warning("Could not clean up failed Gemini upload")
            raise

    def delete_file(self, file_ref: Any) -> None:
        """Best-effort deletion of an uploaded file."""
        try:
            self._client.files.delete(name=file_ref.name)
            logger.debug("Deleted uploaded file: %s", file_ref.name)
        except Exception:
            logger.warning("Could not delete uploaded file: %s", file_ref.name)

    # ------------------------------------------------------------------
    # Content generation
    # ------------------------------------------------------------------

    def analyze_video(
        self,
        video_file: Any,
        prompt: str,
        json_mode: bool = True,
    ) -> str:
        """Send a prompt + video to Gemini and return the text response.

        When *json_mode* is ``True`` the model is asked to respond in JSON.
        """
        gen_config: Dict[str, Any] = {
            "temperature": 0.2,
        }
        if json_mode:
            gen_config["response_mime_type"] = "application/json"

        return self._generate_with_retry(
            contents=[video_file, prompt],
            generation_config=gen_config,
        )

    def analyze_frames(
        self,
        frame_parts: List[Any],
        prompt: str,
        json_mode: bool = True,
    ) -> str:
        """Send multiple image parts + a prompt to Gemini.

        Used for temporal refinement with sampled frames.
        """
        gen_config: Dict[str, Any] = {
            "temperature": 0.1,
        }
        if json_mode:
            gen_config["response_mime_type"] = "application/json"

        contents = list(frame_parts) + [prompt]
        return self._generate_with_retry(
            contents=contents,
            generation_config=gen_config,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_with_retry(
        self,
        contents: list,
        generation_config: dict,
    ) -> str:
        """Wrapper that retries on transient failures."""
        max_attempts = self._config.max_retries + 1

        @retry(
            retry=retry_if_exception_type(_RETRYABLE),
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(
                multiplier=self._config.retry_base_wait_s,
                min=2,
                max=60,
            ),
            reraise=True,
        )
        def _call():
            try:
                response = self._client.models.generate_content(
                    model=self._config.gemini_model,
                    contents=contents,
                    config=generation_config,
                )
                if not response.text:
                    raise GeminiAPIError("Empty response from Gemini")
                return response.text

            except GeminiRateLimitError:
                raise
            except GeminiAPIError:
                raise
            except Exception as exc:
                exc_str = str(exc).lower()
                if "429" in exc_str or "rate" in exc_str or "quota" in exc_str:
                    raise GeminiRateLimitError() from exc
                if "500" in exc_str or "503" in exc_str or "unavailable" in exc_str:
                    raise GeminiRateLimitError(
                        f"Transient server error: {exc}"
                    ) from exc
                raise GeminiAPIError(f"Gemini API call failed: {exc}") from exc

        return _call()

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def parse_json_response(text: str) -> Any:
        """Parse a JSON response, tolerating markdown fences.

        Raises :class:`GeminiResponseParseError` on failure.
        """
        # Strip markdown code fences if present
        cleaned = text.strip()
        fence_match = re.match(
            r"```(?:json)?\s*\n?(.*?)```",
            cleaned,
            re.DOTALL,
        )
        if fence_match:
            cleaned = fence_match.group(1).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise GeminiResponseParseError(
                f"Failed to parse JSON from Gemini response: {exc}\n"
                f"Raw text (first 500 chars): {text[:500]}"
            ) from exc
