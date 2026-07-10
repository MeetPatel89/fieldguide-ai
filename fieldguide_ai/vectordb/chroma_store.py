import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from fieldguide_ai.ingestion.models import DocumentChunk
from fieldguide_ai.vectordb.embeddings import OpenAIEmbeddingProvider


DEFAULT_COLLECTION_NAME = "documents"


class ChromaVectorStore:
    """Persist document chunks into a named Chroma collection."""

    def __init__(
        self,
        path: str,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_provider: OpenAIEmbeddingProvider | None = None,
        client: ClientAPI | None = None,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.client = client or chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def index_chunks(self, chunks: Sequence[DocumentChunk]) -> None:
        if not chunks:
            return
        if self.embedding_provider is None:
            raise ValueError("embedding_provider is required to index chunks")

        documents = [chunk.content for chunk in chunks]
        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=documents,
            embeddings=self.embedding_provider.embed_texts(documents),
            metadatas=[serialize_chunk_metadata(chunk) for chunk in chunks],
        )

    def get_collection(self) -> Collection:
        return self.collection

    def query(self, query_text: str, n_results: int = 10) -> dict[str, Any]:
        if self.embedding_provider is None:
            raise ValueError("embedding_provider is required to query the collection")

        query_embeddings = self.embedding_provider.embed_texts([query_text])
        return self.collection.query(query_embeddings=query_embeddings, n_results=n_results)


def serialize_chunk_metadata(chunk: DocumentChunk) -> dict[str, str | int | float | bool | None]:
    metadata: dict[str, str | int | float | bool | None] = {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "source_path": str(chunk.source_path),
        "chunk_index": chunk.chunk_index,
        "section_path": " > ".join(chunk.section_path),
        "section_path_json": json.dumps(list(chunk.section_path)),
        "section_title": _section_title(chunk),
    }

    for key, value in chunk.metadata.items():
        metadata[key] = _normalize_metadata_value(value)

    return metadata


def _section_title(chunk: DocumentChunk) -> str:
    value = chunk.metadata.get("section_title")
    if isinstance(value, str) and value:
        return value
    if chunk.section_path:
        return chunk.section_path[-1]
    return ""


def _normalize_metadata_value(value: Any) -> str | int | float | bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (str, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return json.dumps(_json_ready(value), sort_keys=True)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return json.dumps(_json_ready(value), sort_keys=True)
    return str(value)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def main():
    import os
    from dotenv import load_dotenv
    load_dotenv()

    embedding_provider = OpenAIEmbeddingProvider(api_key=os.getenv("OPENAI_API_KEY"))
    client = chromadb.PersistentClient(path="chroma_db")
    chroma_store = ChromaVectorStore(path="chroma_db", embedding_provider=embedding_provider, client=client)
    query = "Several branch users cannot access Employee Portal after certificate rotation. Some coworkers can log in. What is the likely issue?"
    results = chroma_store.query(query, n_results=10)
    print("--------------------------------")
    print("results: ", results)
    print("--------------------------------")

    # from fieldguide_ai.ingestion import MarkdownSectionChunker, load_markdown_documents
    # chunks = MarkdownSectionChunker(max_words=1000).chunk_documents(load_markdown_documents("data/corpora/nautilus/raw"))
    # chroma_store.index_chunks(chunks)

    # print("--------------------------------")
    # print("chunks indexed: ", len(chunks))
    # print("--------------------------------")


if __name__ == "__main__":
    main()
