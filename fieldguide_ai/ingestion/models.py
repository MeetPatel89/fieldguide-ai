"""Models for the ingestion pipeline."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, init=False)
class MarkdownDocument:
    """A raw markdown document plus parsed frontmatter metadata."""

    source_path: Path
    body: str
    _metadata: dict[str, Any] = field(default_factory=dict, repr=False)

    def __init__(
        self,
        source_path: Path,
        body: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(source_path, Path):
            raise TypeError("source_path must be a pathlib.Path")
        if not isinstance(body, str):
            raise TypeError("document body must be text")
        object.__setattr__(self, "source_path", source_path)
        object.__setattr__(self, "body", body)
        object.__setattr__(self, "_metadata", deepcopy(dict(metadata or {})))

    @property
    def metadata(self) -> dict[str, Any]:
        """An independent copy of document metadata."""
        return deepcopy(self._metadata)

    @property
    def doc_id(self) -> str:
        """The document ID.

        Returns
        -------
        doc_id : str
            The document ID.
        """
        return str(self._metadata.get("doc_id") or self.source_path.stem)

    @property
    def title(self) -> str:
        """The document title.

        Returns
        -------
        title : str
            The document title.
        """
        return str(self._metadata.get("title") or self.source_path.stem)


@dataclass(frozen=True, init=False)
class DocumentChunk:
    """A retrieval-ready chunk with document and section metadata attached."""

    chunk_id: str
    doc_id: str
    content: str
    source_path: Path
    section_path: tuple[str, ...]
    chunk_index: int
    _metadata: dict[str, Any] = field(default_factory=dict, repr=False)

    def __init__(
        self,
        chunk_id: str,
        doc_id: str,
        content: str,
        source_path: Path,
        section_path: tuple[str, ...],
        chunk_index: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not chunk_id.strip():
            raise ValueError("chunk_id must not be blank")
        if not doc_id.strip():
            raise ValueError("doc_id must not be blank")
        if not content.strip():
            raise ValueError("chunk content must not be blank")
        if not isinstance(source_path, Path):
            raise TypeError("source_path must be a pathlib.Path")
        if not section_path or any(not section.strip() for section in section_path):
            raise ValueError("section_path must contain non-blank sections")
        if chunk_index < 0:
            raise ValueError("chunk_index cannot be negative")

        object.__setattr__(self, "chunk_id", chunk_id)
        object.__setattr__(self, "doc_id", doc_id)
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "source_path", source_path)
        object.__setattr__(self, "section_path", tuple(section_path))
        object.__setattr__(self, "chunk_index", chunk_index)
        object.__setattr__(self, "_metadata", deepcopy(dict(metadata or {})))

    @property
    def metadata(self) -> dict[str, Any]:
        """An independent copy of chunk metadata."""
        return deepcopy(self._metadata)

    def to_record(self) -> dict[str, Any]:
        """Convert the document chunk to a record.

        Returns
        -------
        record : dict[str, Any]
            The record of the document chunk.
        """
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "content": self.content,
            "source_path": str(self.source_path),
            "section_path": list(self.section_path),
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
        }
