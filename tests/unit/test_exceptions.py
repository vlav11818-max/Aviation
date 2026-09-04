"""Unit tests for core.exceptions.

Tests: all exception classes instantiate correctly, structured data
fields accessible (StepError.step_name, APIRateLimitError.retry_after,
etc.), inheritance hierarchy correct (APIConnectionError is APIError
is StoryGeneratorError).
"""

from __future__ import annotations

import pytest

from core.exceptions import (
    APIAuthError,
    APIConnectionError,
    APIError,
    APIRateLimitError,
    APIResponseError,
    ConfigError,
    EvaluationError,
    ExportError,
    PipelineError,
    PromptTemplateError,
    StateError,
    StepError,
    StoryGeneratorError,
)


# ── Tests: base exception ─────────────────────────────────────────────


class TestStoryGeneratorError:
    """Tests for the root exception class."""

    def test_instantiate(self) -> None:
        """Root exception should be instantiable with a message."""
        exc = StoryGeneratorError("test error")
        assert str(exc) == "test error"

    def test_is_exception(self) -> None:
        """Root exception should be a subclass of Exception."""
        assert issubclass(StoryGeneratorError, Exception)


# ── Tests: API exceptions ─────────────────────────────────────────────


class TestAPIExceptions:
    """Tests for the API exception hierarchy."""

    def test_api_error_inherits(self) -> None:
        """APIError should inherit from StoryGeneratorError."""
        assert issubclass(APIError, StoryGeneratorError)

    def test_api_error_fields(self) -> None:
        """APIError should have provider and model fields."""
        exc = APIError("fail", provider="openai", model="gpt-4o")
        assert exc.provider == "openai"
        assert exc.model == "gpt-4o"
        assert str(exc) == "fail"

    def test_api_error_optional_fields(self) -> None:
        """APIError should work with default None fields."""
        exc = APIError("fail")
        assert exc.provider is None
        assert exc.model is None

    def test_connection_error_inherits(self) -> None:
        """APIConnectionError should inherit from APIError."""
        assert issubclass(APIConnectionError, APIError)
        assert issubclass(APIConnectionError, StoryGeneratorError)

    def test_connection_error_instantiate(self) -> None:
        """APIConnectionError should be instantiable."""
        exc = APIConnectionError("timeout", provider="google")
        assert exc.provider == "google"

    def test_rate_limit_error_inherits(self) -> None:
        """APIRateLimitError should inherit from APIError."""
        assert issubclass(APIRateLimitError, APIError)

    def test_rate_limit_error_fields(self) -> None:
        """APIRateLimitError should have retry_after field."""
        exc = APIRateLimitError(
            "rate limited", provider="anthropic", retry_after=30.0
        )
        assert exc.retry_after == 30.0
        assert exc.provider == "anthropic"

    def test_rate_limit_default_retry(self) -> None:
        """APIRateLimitError should default retry_after to None."""
        exc = APIRateLimitError("limited")
        assert exc.retry_after is None

    def test_auth_error_inherits(self) -> None:
        """APIAuthError should inherit from APIError."""
        assert issubclass(APIAuthError, APIError)

    def test_auth_error_instantiate(self) -> None:
        """APIAuthError should be instantiable."""
        exc = APIAuthError("bad key", provider="deepseek")
        assert exc.provider == "deepseek"

    def test_response_error_inherits(self) -> None:
        """APIResponseError should inherit from APIError."""
        assert issubclass(APIResponseError, APIError)

    def test_response_error_fields(self) -> None:
        """APIResponseError should have status_code and response_body fields."""
        exc = APIResponseError(
            "server error",
            status_code=500,
            response_body='{"error": "internal"}',
            provider="openai",
            model="gpt-4o",
        )
        assert exc.status_code == 500
        assert exc.response_body == '{"error": "internal"}'
        assert exc.provider == "openai"
        assert exc.model == "gpt-4o"

    def test_response_error_optional_fields(self) -> None:
        """APIResponseError should default optional fields to None."""
        exc = APIResponseError("error")
        assert exc.status_code is None
        assert exc.response_body is None
        assert exc.provider is None
        assert exc.model is None


# ── Tests: pipeline exceptions ────────────────────────────────────────


class TestPipelineExceptions:
    """Tests for pipeline-related exceptions."""

    def test_pipeline_error_inherits(self) -> None:
        """PipelineError should inherit from StoryGeneratorError."""
        assert issubclass(PipelineError, StoryGeneratorError)

    def test_step_error_inherits(self) -> None:
        """StepError should inherit from PipelineError."""
        assert issubclass(StepError, PipelineError)
        assert issubclass(StepError, StoryGeneratorError)

    def test_step_error_fields(self) -> None:
        """StepError should have step_name and recoverable fields."""
        exc = StepError("step failed", step_name="concept", recoverable=True)
        assert exc.step_name == "concept"
        assert exc.recoverable is True
        assert str(exc) == "step failed"

    def test_step_error_default_recoverable(self) -> None:
        """StepError should default recoverable to True."""
        exc = StepError("fail", step_name="evaluate")
        assert exc.recoverable is True

    def test_step_error_non_recoverable(self) -> None:
        """StepError can be marked non-recoverable."""
        exc = StepError("fatal", step_name="outline", recoverable=False)
        assert exc.recoverable is False

    def test_evaluation_error_inherits(self) -> None:
        """EvaluationError should inherit from PipelineError."""
        assert issubclass(EvaluationError, PipelineError)

    def test_state_error_inherits(self) -> None:
        """StateError should inherit from PipelineError."""
        assert issubclass(StateError, PipelineError)


# ── Tests: config exceptions ──────────────────────────────────────────


class TestConfigExceptions:
    """Tests for configuration exceptions."""

    def test_config_error_inherits(self) -> None:
        """ConfigError should inherit from StoryGeneratorError."""
        assert issubclass(ConfigError, StoryGeneratorError)

    def test_config_error_instantiate(self) -> None:
        """ConfigError should be instantiable with a message."""
        exc = ConfigError("bad config")
        assert str(exc) == "bad config"


# ── Tests: prompt exceptions ──────────────────────────────────────────


class TestPromptExceptions:
    """Tests for prompt template exceptions."""

    def test_prompt_error_inherits(self) -> None:
        """PromptTemplateError should inherit from StoryGeneratorError."""
        assert issubclass(PromptTemplateError, StoryGeneratorError)

    def test_prompt_error_fields(self) -> None:
        """PromptTemplateError should have template_name field."""
        exc = PromptTemplateError(
            "template not found", template_name="concept"
        )
        assert exc.template_name == "concept"
        assert str(exc) == "template not found"

    def test_prompt_error_default_none(self) -> None:
        """PromptTemplateError should default template_name to None."""
        exc = PromptTemplateError("error")
        assert exc.template_name is None


# ── Tests: export exceptions ──────────────────────────────────────────


class TestExportExceptions:
    """Tests for export exceptions."""

    def test_export_error_inherits(self) -> None:
        """ExportError should inherit from StoryGeneratorError."""
        assert issubclass(ExportError, StoryGeneratorError)

    def test_export_error_fields(self) -> None:
        """ExportError should have export_format field."""
        exc = ExportError("failed to export", export_format="ssml")
        assert exc.export_format == "ssml"

    def test_export_error_default_none(self) -> None:
        """ExportError should default export_format to None."""
        exc = ExportError("error")
        assert exc.export_format is None


# ── Tests: exception chaining ─────────────────────────────────────────


class TestExceptionChaining:
    """Tests for proper exception chaining with 'from'."""

    def test_chaining_preserved(self) -> None:
        """Exceptions wrapped with 'from' should preserve the chain."""
        original = ValueError("original")
        try:
            try:
                raise original
            except ValueError as exc:
                raise StepError(
                    "wrapped", step_name="test", recoverable=True
                ) from exc
        except StepError as exc:
            assert exc.__cause__ is original

    def test_isinstance_checks(self) -> None:
        """All exceptions should satisfy isinstance checks up the tree."""
        exc = APIRateLimitError("limited", provider="openai", retry_after=5.0)
        assert isinstance(exc, APIRateLimitError)
        assert isinstance(exc, APIError)
        assert isinstance(exc, StoryGeneratorError)
        assert isinstance(exc, Exception)
