"""Pipeline for loading, chunking, and indexing Markdown documents."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from fieldguide_ai.ingestion.chunkers import MarkdownSectionChunker
from fieldguide_ai.ingestion.loader import load_markdown_documents
from fieldguide_ai.ingestion.models import MarkdownDocument
from fieldguide_ai.vectorstore.base import DocumentIndex


@dataclass(frozen=True)
class IndexingResult:
    """Counts produced by an indexing operation."""

    document_count: int
    chunk_count: int

    def __post_init__(self) -> None:
        """Reject impossible indexing counts."""
        if self.document_count < 0 or self.chunk_count < 0:
            raise ValueError("indexing counts cannot be negative")


class DocumentIndexingPipeline:
    """Orchestrate Markdown loading, chunking, and document-level replacement."""

    def __init__(
        self,
        vector_store: DocumentIndex,
        chunker: MarkdownSectionChunker | None = None,
        document_loader: Callable[
            [str | Path], Sequence[MarkdownDocument]
        ] = load_markdown_documents,
    ) -> None:
        self._vector_store = vector_store
        self._chunker = chunker if chunker is not None else MarkdownSectionChunker()
        self._document_loader = document_loader

    def index_path(self, path: str | Path) -> IndexingResult:
        """Load and index every Markdown document under a path."""
        return self.index_documents(self._document_loader(path))

    def index_documents(self, documents: Sequence[MarkdownDocument]) -> IndexingResult:
        """Chunk and index a sequence of Markdown documents."""
        chunk_count = 0
        for document in documents:
            chunks = self._chunker.chunk_document(document)
            if chunks:
                self._vector_store.replace_chunks(chunks)
            else:
                self._vector_store.delete_documents([document.doc_id])
            chunk_count += len(chunks)

        return IndexingResult(document_count=len(documents), chunk_count=chunk_count)
