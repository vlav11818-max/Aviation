"""Legacy shim — the individual per-provider adapters were replaced by
the LiteLLM-backed router in :mod:`core.llm.router`.

Only the :class:`APIResponse` dataclass is kept for backward compatibility
with any code that imported it from here. New code should use
:class:`core.llm.LLMResponse` directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Re-export the new response type under the old name so imports don't break.
from core.llm import LLMResponse  # noqa: F401


@dataclass(frozen=True)
class APIResponse:
    """Deprecated alias for :class:`core.llm.LLMResponse`.

    Retained so pre-refactor imports still work; instantiated only in a
    handful of tests. Prefer :class:`LLMResponse` in new code.
    """

    text: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    model: str = ""
    raw: dict = field(default_factory=dict)
