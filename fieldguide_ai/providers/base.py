"""Base class for LLM providers."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from fieldguide_ai.chat import ChatMessage, GenerationResult


class LLMProvider(ABC):
    """Base class for LLM providers."""

    def __init__(
        self,
        message_history: Sequence[ChatMessage] | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self._system_prompt = system_prompt
        self._message_history = list(message_history or ())
        self._generation_log: list[GenerationResult] = []

    @abstractmethod
    def generate(self, messages: Sequence[ChatMessage]) -> GenerationResult:
        """Return normalized model output for conversation turns."""

    @property
    def system_prompt(self) -> str | None:
        """The system prompt configured for this conversation."""
        return self._system_prompt

    def set_system_prompt(self, system_prompt: str | None) -> None:
        """Replace the system prompt without changing conversation turns."""
        self._system_prompt = system_prompt

    def clear_history(self) -> None:
        """Clear conversation turns without changing the system prompt."""
        self._message_history.clear()

    def get_history(self) -> list[ChatMessage]:
        """Get a copy of the conversation history."""
        return list(self._message_history)

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

    def complete_turn(
        self,
        message: str | ChatMessage,
        *,
        generation_message: ChatMessage | None = None,
    ) -> GenerationResult:
        """Generate and atomically record one user/assistant conversation turn.

        ``generation_message`` lets an application add transient context while the
        canonical user message remains in visible history.
        """
        user_message = self._as_user_message(message)
        outbound_message = generation_message or user_message
        if outbound_message.role != "user":
            raise ValueError("generation_message must have the user role")

        result = self.generate([*self._message_history, outbound_message])
        self._message_history.extend(
            [user_message, ChatMessage(role="assistant", content=result.text)]
        )
        return result

    def chat(self, message: str | ChatMessage) -> str:
        """Send and atomically record one user/assistant conversation turn."""
        return self.complete_turn(message).text

    @staticmethod
    def _as_user_message(message: str | ChatMessage) -> ChatMessage:
        if isinstance(message, ChatMessage):
            if message.role != "user":
                raise ValueError("chat() requires a user-role ChatMessage")
            return message
        return ChatMessage(role="user", content=message)


class ProviderBackend(ABC):
    """Provider integration responsible for SDK access and chat construction."""

    @abstractmethod
    def build_provider(
        self,
        model: str,
        *,
        message_history: Sequence[ChatMessage] | None = None,
        system_prompt: str | None = None,
    ) -> LLMProvider:
        """Build a chat provider configured for a model."""

    @abstractmethod
    def list_models(self) -> list[str]:
        """List models available through the provider SDK."""
