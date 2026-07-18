"""Base class for LLM providers."""

from abc import ABC, abstractmethod

from fieldguide_ai.generation import GenerationResult
from fieldguide_ai.messages import ChatMessage


class LLMProvider(ABC):
    """Base class for LLM providers."""

    def __init__(
        self,
        message_history: list[ChatMessage] | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.system_prompt = system_prompt
        self.message_history = list(message_history or [])
        self._generation_log: list[GenerationResult] = []

    @abstractmethod
    def generate(self, messages: list[ChatMessage]) -> GenerationResult:
        """Return normalized model output for conversation turns."""

    @abstractmethod
    def list_models(self) -> list[str]:
        """List available models."""

    def add_message(self, message: ChatMessage) -> None:
        """Add a message to the conversation history."""
        self.message_history.append(message)

    def clear_history(self) -> None:
        """Clear conversation turns without changing the system prompt."""
        self.message_history.clear()

    def get_history(self) -> list[ChatMessage]:
        """Get a copy of the conversation history."""
        return list(self.message_history)

    def get_generation_log(self) -> list[GenerationResult]:
        """Get independent copies of recorded generation results."""
        return [result.model_copy(deep=True) for result in self._generation_log]

    @property
    def last_result(self) -> GenerationResult | None:
        """Independent copy of the most recent generation result, if any."""
        if not self._generation_log:
            return None
        return self._generation_log[-1].model_copy(deep=True)

    def _record_generation(self, result: GenerationResult) -> GenerationResult:
        """Append a defensive snapshot to the in-memory log and return the result."""
        self._generation_log.append(result.model_copy(deep=True))
        return result

    def chat(self, message: str | ChatMessage) -> str:
        """Send a user turn to the model and record the assistant response."""
        if isinstance(message, ChatMessage) and message.role != "user":
            raise ValueError("chat() requires a user-role ChatMessage")

        user_message = (
            message
            if isinstance(message, ChatMessage)
            else ChatMessage(role="user", content=message)
        )
        self.add_message(user_message)

        result = self.generate(self.message_history)
        self.add_message(ChatMessage(role="assistant", content=result.text))

        return result.text
