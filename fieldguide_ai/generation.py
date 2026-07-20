"""Normalized LLM generation results."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

TokenCount = Annotated[int, Field(ge=0)]
LatencyMilliseconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class TokenUsage(BaseModel):
    """Provider-agnostic token counts when available."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    input_tokens: TokenCount | None = None
    output_tokens: TokenCount | None = None
    total_tokens: TokenCount | None = None


class GenerationResult(BaseModel):
    """Normalized generation output plus optional raw provider metadata.

    ``text`` is the application-facing assistant content used in chat history.
    Remaining fields capture telemetry for debugging, evaluation, and later
    persistence. ``raw`` should be JSON-serializable when present.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    text: str
    provider: str
    model: str
    response_id: str | None = None
    finish_reason: str | None = None
    usage: TokenUsage | None = None
    latency_ms: LatencyMilliseconds | None = None
    raw: dict[str, JsonValue] | None = None

    @field_validator("provider", "model")
    @classmethod
    def identifiers_must_not_be_blank(cls, value: str) -> str:
        """Reject identifiers that contain no visible characters."""
        if not value.strip():
            raise ValueError("must not be blank")
        return value
