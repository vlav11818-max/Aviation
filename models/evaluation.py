"""Evaluation data models.

Captures the results of the 4-level evaluation system:
L1 Technical, L2 Linguistic, L3 Content, L4 Voiceover.
Each level has its own score and list of issues.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class EvaluationLevel(str, Enum):
    """The four evaluation levels."""

    L1_TECHNICAL = "L1_technical"
    L2_LINGUISTIC = "L2_linguistic"
    L3_CONTENT = "L3_content"
    L4_VOICEOVER = "L4_voiceover"


class IssueSeverity(str, Enum):
    """Severity levels for evaluation issues."""

    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    INFO = "info"


class EvaluationIssue(BaseModel):
    """A single issue found during evaluation.

    Attributes:
        level: Which evaluation level detected this issue.
        category: Issue category (e.g., "grammar", "pacing", "marker").
        description: Human-readable description of the issue.
        severity: How severe the issue is.
        line_reference: Optional line or character reference in the text.
    """

    level: EvaluationLevel = Field(
        description="Evaluation level that detected this issue.",
    )
    category: str = Field(
        description="Issue category.",
    )
    description: str = Field(
        description="Human-readable issue description.",
    )
    severity: IssueSeverity = Field(
        default=IssueSeverity.MINOR,
        description="Issue severity.",
    )
    line_reference: str = Field(
        default="",
        description="Line or character reference in the text.",
    )


class LevelResult(BaseModel):
    """Score and issues for a single evaluation level.

    Attributes:
        score: Score for this level (0.0–10.0).
        issues: Issues detected at this level.
    """

    score: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Score for this evaluation level (0–10).",
    )
    issues: list[EvaluationIssue] = Field(
        default_factory=list,
        description="Issues found at this level.",
    )


class EvaluationResult(BaseModel):
    """Complete result of a 4-level story evaluation.

    Attributes:
        l1_technical: L1 Technical evaluation (length, markers, encoding).
        l2_linguistic: L2 Linguistic evaluation (grammar, naturalness).
        l3_content: L3 Content evaluation (topic, completeness, logic).
        l4_voiceover: L4 Voiceover evaluation (readability, punctuation).
        overall_score: Weighted overall score (0.0–10.0).
        passed: Whether the overall score meets the minimum threshold.
        summary: Brief textual summary of the evaluation.
        critical_issues: List of issues classified as critical.
        attempt_number: Which evaluation attempt this is (1-based).
        timestamp: ISO-8601 timestamp of the evaluation.
    """

    l1_technical: LevelResult = Field(
        default_factory=LevelResult,
        description="L1 Technical evaluation result.",
    )
    l2_linguistic: LevelResult = Field(
        default_factory=LevelResult,
        description="L2 Linguistic evaluation result.",
    )
    l3_content: LevelResult = Field(
        default_factory=LevelResult,
        description="L3 Content evaluation result.",
    )
    l4_voiceover: LevelResult = Field(
        default_factory=LevelResult,
        description="L4 Voiceover evaluation result.",
    )
    overall_score: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Weighted overall score.",
    )
    passed: bool = Field(
        default=False,
        description="Whether the score meets the minimum threshold.",
    )
    summary: str = Field(
        default="",
        description="Brief evaluation summary.",
    )
    critical_issues: list[EvaluationIssue] = Field(
        default_factory=list,
        description="Issues with critical severity.",
    )
    attempt_number: int = Field(
        default=1,
        ge=1,
        description="Evaluation attempt number (1-based).",
    )
    timestamp: str = Field(
        default="",
        description="ISO-8601 timestamp of evaluation.",
    )

    @property
    def all_issues(self) -> list[EvaluationIssue]:
        """Return a flat list of all issues across all levels."""
        return (
            self.l1_technical.issues
            + self.l2_linguistic.issues
            + self.l3_content.issues
            + self.l4_voiceover.issues
        )

    @property
    def issue_count(self) -> int:
        """Total number of issues across all levels."""
        return len(self.all_issues)
