"""Anthropic provider for the Fieldguide AI."""

import time
from typing import Any

from anthropic import Anthropic
from anthropic.types import Message

from fieldguide_ai.generation import GenerationResult, TokenUsage
from fieldguide_ai.messages import ChatMessage
from fieldguide_ai.providers.base import LLMProvider


def to_anthropic_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """Convert conversation turns to Anthropic Messages API message dicts."""
    return [{"role": message.role, "content": message.content} for message in messages]


def from_anthropic_response(
    response: Message,
    *,
    model: str,
    latency_ms: float | None = None,
) -> GenerationResult:
    """Normalize an Anthropic Messages API payload into ``GenerationResult``."""
    content = getattr(response, "content", None) or []
    text = "".join(
        block.text for block in content if getattr(block, "type", None) == "text"
    )

    usage = None
    usage_payload = getattr(response, "usage", None)
    if usage_payload is not None:
        input_tokens = getattr(usage_payload, "input_tokens", None)
        output_tokens = getattr(usage_payload, "output_tokens", None)
        total_tokens = None
        if input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens
        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

    raw = response.model_dump(mode="json")

    return GenerationResult(
        text=text,
        provider="anthropic",
        model=model,
        response_id=getattr(response, "id", None),
        finish_reason=getattr(response, "stop_reason", None),
        usage=usage,
        latency_ms=latency_ms,
        raw=raw,
    )


class AnthropicProvider(LLMProvider):
    """Anthropic provider for the Fieldguide AI."""

    def __init__(
        self,
        api_key: str | None,
        model: str,
        message_history: list[ChatMessage] | None = None,
        system_prompt: str | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")

        super().__init__(
            message_history=message_history,
            system_prompt=system_prompt,
        )
        self.model = model
        self.client = Anthropic(api_key=api_key)

    def generate(self, messages: list[ChatMessage]) -> GenerationResult:
        """Generate a response from the model."""
        create_kwargs: dict[str, Any] = {
            "max_tokens": 1000,
            "model": self.model,
            "messages": to_anthropic_messages(messages),
        }
        if self.system_prompt is not None:
            create_kwargs["system"] = self.system_prompt

        started = time.perf_counter()
        response = self.client.messages.create(**create_kwargs)
        latency_ms = (time.perf_counter() - started) * 1000
        return self._record_generation(
            from_anthropic_response(
                response,
                model=self.model,
                latency_ms=latency_ms,
            )
        )

    def list_models(self) -> list[str]:
        """List available models."""
        return sorted(model.id for model in self.client.models.list().data)


def main() -> None:
    """Run the main function."""
    import os

    from dotenv import load_dotenv

    load_dotenv()
    provider = AnthropicProvider(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        model="claude-haiku-4-5-20251001",
        system_prompt="Answer briefly.",
    )
    result = provider.generate(
        [
            ChatMessage(
                role="user",
                content="Hello, how are you? Please tell me about yourself.",
            )
        ]
    )
    print(result.text)
    print(provider.list_models())


if __name__ == "__main__":
    main()
