"""Custom exceptions for FORGE TRACE Task Intelligence module."""


class ForgeTraceError(Exception):
    """Base exception for all FORGE TRACE errors."""
    pass


class VideoValidationError(ForgeTraceError):
    """Raised when video input fails validation."""
    pass


class VideoReadError(ForgeTraceError):
    """Raised when a video file cannot be read or decoded."""
    pass


class GeminiAPIError(ForgeTraceError):
    """Raised when the Gemini API returns an error."""

    def __init__(self, message: str, status_code: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class GeminiRateLimitError(GeminiAPIError):
    """Raised when the Gemini API rate limit is exceeded."""

    def __init__(self, message: str = "Gemini API rate limit exceeded"):
        super().__init__(message, status_code=429, retryable=True)


class GeminiResponseParseError(ForgeTraceError):
    """Raised when a Gemini API response cannot be parsed."""
    pass


class FileUploadError(ForgeTraceError):
    """Raised when a file cannot be uploaded to the Gemini Files API."""
    pass


class SegmentAnalysisError(ForgeTraceError):
    """Raised when analysis of a video segment fails."""

    def __init__(self, message: str, segment_index: int | None = None):
        super().__init__(message)
        self.segment_index = segment_index


class TemporalRefinementError(ForgeTraceError):
    """Raised when temporal refinement of an action fails."""
    pass


class TimelineValidationError(ForgeTraceError):
    """Raised when the final timeline fails validation."""
    pass


class EnrichmentError(ForgeTraceError):
    """Raised when enrichment with external module data fails (non-fatal)."""
    pass


class ConfigurationError(ForgeTraceError):
    """Raised when configuration is missing or invalid."""
    pass
