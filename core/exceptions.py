"""Custom exception hierarchy for AI Story Generator Pro.

All project exceptions inherit from StoryGeneratorError. Categorized into
API, Pipeline, Config, and Export groups with structured data fields
(step name, retry_after, recoverability).
"""


class StoryGeneratorError(Exception):
    """Base exception for all project errors.

    All custom exceptions in the project inherit from this class,
    enabling unified exception handling at the top level.
    """

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


# ── API Errors ──────────────────────────────────────────────────────────


class APIError(StoryGeneratorError):
    """Base exception for all API-related errors.

    Attributes:
        provider: The API provider that produced the error.
        model: The model that was being used, if known.
    """

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        super().__init__(message)


class APIConnectionError(APIError):
    """Raised when unable to connect to an API provider.

    Covers network errors, DNS failures, and connection timeouts.
    """


class APIRateLimitError(APIError):
    """Raised when an API provider returns a 429 rate-limit response.

    Attributes:
        retry_after: Seconds to wait before retrying, if provided by the API.
    """

    def __init__(
        self,
        message: str,
        retry_after: float | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        self.retry_after = retry_after
        super().__init__(message, provider=provider, model=model)


class APIAuthError(APIError):
    """Raised when API authentication fails (invalid or missing key)."""


class APIResponseError(APIError):
    """Raised when an API returns an unexpected or malformed response.

    Attributes:
        status_code: HTTP status code, if available.
        response_body: Raw response body for debugging.
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_body: str | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.response_body = response_body
        super().__init__(message, provider=provider, model=model)


# ── Pipeline Errors ─────────────────────────────────────────────────────


class PipelineError(StoryGeneratorError):
    """Base exception for pipeline execution errors."""


class StepError(PipelineError):
    """Raised when a pipeline step fails.

    Attributes:
        step_name: Name of the step that failed.
        recoverable: Whether the pipeline can continue past this failure.
    """

    def __init__(
        self,
        message: str,
        step_name: str,
        recoverable: bool = True,
    ) -> None:
        self.step_name = step_name
        self.recoverable = recoverable
        super().__init__(message)


class EvaluationError(PipelineError):
    """Raised when story evaluation fails or produces invalid results."""


class StateError(PipelineError):
    """Raised when pipeline state is invalid, missing, or corrupted."""


# ── Config Errors ───────────────────────────────────────────────────────


class ConfigError(StoryGeneratorError):
    """Raised when configuration is invalid, missing, or cannot be loaded."""


# ── Prompt Errors ───────────────────────────────────────────────────────


class PromptTemplateError(StoryGeneratorError):
    """Raised when a prompt template is missing, malformed, or cannot be rendered.

    Attributes:
        template_name: The template that caused the error.
    """

    def __init__(
        self,
        message: str,
        template_name: str | None = None,
    ) -> None:
        self.template_name = template_name
        super().__init__(message)


# ── Export Errors ───────────────────────────────────────────────────────


class ExportError(StoryGeneratorError):
    """Raised when exporting a story to TXT, SSML, or report format fails.

    Attributes:
        export_format: The format that failed (e.g., "txt", "ssml", "json").
    """

    def __init__(
        self,
        message: str,
        export_format: str | None = None,
    ) -> None:
        self.export_format = export_format
        super().__init__(message)
