"""Models for the ingestion pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MarkdownDocument:
    """A raw markdown document plus parsed frontmatter metadata."""

    source_path: Path
    body: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        """The document ID.

        Returns
        -------
        doc_id : str
            The document ID.
        """
        return str(self.metadata.get("doc_id") or self.source_path.stem)

    @property
    def title(self) -> str:
        """The document title.

        Returns
        -------
        title : str
            The document title.
        """
        return str(self.metadata.get("title") or self.source_path.stem)


@dataclass(frozen=True)
class DocumentChunk:
    """A retrieval-ready chunk with document and section metadata attached."""

    chunk_id: str
    doc_id: str
    content: str
    source_path: Path
    section_path: tuple[str, ...]
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

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
