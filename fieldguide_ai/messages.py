from dataclasses import dataclass
from typing import Any, Literal

Provider = Literal["openai"]
Role = Literal["user", "assistant", "system"]


@dataclass(frozen=True)
class ChatMessage:
    """A message in a chat conversation."""

    role: Role
    content: str

    def to_provider_message(self, provider: Provider) -> dict[str, Any]:
        if provider == "openai":
            return {
                "role": self.role,
                "content": self.content,
            }
        raise ValueError(f"Unsupported provider: {provider}")

    def to_input_item(self) -> dict[str, Any]:
        """Return the OpenAI input representation kept for backwards compatibility."""
        return self.to_provider_message("openai")
