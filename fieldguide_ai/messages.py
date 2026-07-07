from dataclasses import dataclass


@dataclass(frozen=True)
class ChatMessage:
    """A message in a chat conversation."""

    role: str
    content: str

    def to_input_item(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}
