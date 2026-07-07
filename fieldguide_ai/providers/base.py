from abc import ABC, abstractmethod

from fieldguide_ai.messages import ChatMessage


class LLMProvider(ABC):
    def __init__(self, message_history: list[ChatMessage] | None = None) -> None:
        self.message_history = list(message_history or [])

    @abstractmethod
    def generate(self, messages: list[ChatMessage]) -> str:
        """Return model output for a chat-style prompt."""

    def add_message(self, message: ChatMessage) -> None:
        self.message_history.append(message)

    def clear_history(self) -> None:
        self.message_history.clear()

    def get_history(self) -> list[ChatMessage]:
        return list(self.message_history)

    def chat(self, message: str | ChatMessage) -> str:
        user_message = (
            message if isinstance(message, ChatMessage) else ChatMessage(role="user", content=message)
        )
        self.add_message(user_message)

        response_text = self.generate(self.message_history)
        self.add_message(ChatMessage(role="assistant", content=response_text))

        return response_text
