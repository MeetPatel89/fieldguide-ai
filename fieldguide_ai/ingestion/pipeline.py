from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from fieldguide_ai.ingestion.chunkers import MarkdownSectionChunker
from fieldguide_ai.ingestion.loader import load_markdown_documents
from fieldguide_ai.ingestion.models import MarkdownDocument
from fieldguide_ai.vectorstore.base import VectorStore


@dataclass(frozen=True)
class IndexingResult:
    document_count: int
    chunk_count: int


class DocumentIndexingPipeline:
    """Orchestrate Markdown loading, chunking, and document-level replacement."""

    def __init__(
        self,
        vector_store: VectorStore,
        chunker: MarkdownSectionChunker | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.chunker = chunker or MarkdownSectionChunker()

    def index_path(self, path: str | Path) -> IndexingResult:
        return self.index_documents(load_markdown_documents(path))

    def index_documents(self, documents: Sequence[MarkdownDocument]) -> IndexingResult:
        chunk_count = 0
        for document in documents:
            chunks = self.chunker.chunk_document(document)
            if chunks:
                self.vector_store.replace_chunks(chunks)
            else:
                self.vector_store.delete_documents([document.doc_id])
            chunk_count += len(chunks)

        return IndexingResult(document_count=len(documents), chunk_count=chunk_count)
