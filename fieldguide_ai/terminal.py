"""Shared terminal rendering for provider conversation state."""

import sys
from typing import TextIO

from model_runtime import ChatSession


def write_history(
    provider: ChatSession,
    output_stream: TextIO = sys.stdout,
) -> None:
    """Write a provider's system prompt and conversation turns to a stream."""
    index = 1
    if provider.system_prompt is not None:
        output_stream.write(f"{index}. system: {provider.system_prompt}\n")
        index += 1
    for message in provider.history:
        output_stream.write(f"{index}. {message.role.value}: {message.text}\n")
        index += 1
