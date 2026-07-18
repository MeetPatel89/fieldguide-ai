"""Canonical chat messages for Fieldguide AI."""

from dataclasses import dataclass
from typing import Literal

Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    """A provider-agnostic conversation turn (user or assistant)."""

    role: Role
    content: str
