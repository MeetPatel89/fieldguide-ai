"""Chunkers for the ingestion pipeline."""

import re
from dataclasses import dataclass
from typing import Any

from fieldguide_ai.ingestion.models import DocumentChunk, MarkdownDocument

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
DEFAULT_MAX_WORDS = 900
DEFAULT_OVERLAP_WORDS = 75


@dataclass(frozen=True)
class MarkdownSection:
    """A markdown section."""

    title: str
    section_path: tuple[str, ...]
    content: str


class MarkdownSectionChunker:
    """Chunk markdown by section boundaries, then paragraphs and word splits.

    Parameters
    ----------
    max_words : int
        The maximum number of words per chunk.
    overlap_words : int
        The number of words to overlap between chunks.
    """

    def __init__(
        self,
        max_words: int = DEFAULT_MAX_WORDS,
        overlap_words: int = DEFAULT_OVERLAP_WORDS,
    ) -> None:
        if max_words < 100:
            raise ValueError("max_words must be at least 100")
        if overlap_words < 0:
            raise ValueError("overlap_words cannot be negative")
        if overlap_words >= max_words:
            raise ValueError("overlap_words must be smaller than max_words")

        self.max_words = max_words
        self.overlap_words = overlap_words

    def chunk_document(self, document: MarkdownDocument) -> list[DocumentChunk]:
        """Chunk a markdown document into chunks.

        Parameters
        ----------
        document : MarkdownDocument
            The markdown document to chunk.

        Returns
        -------
        chunks : list[DocumentChunk]
            The chunks of the markdown document.
        """
        sections = split_markdown_sections(document.body, fallback_title=document.title)
        chunks: list[DocumentChunk] = []

        for section in sections:
            for content in split_large_section(
                section.content, self.max_words, self.overlap_words
            ):
                chunk_index = len(chunks)
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{document.doc_id}::chunk-{chunk_index:04d}",
                        doc_id=document.doc_id,
                        content=content,
                        source_path=document.source_path,
                        section_path=section.section_path,
                        chunk_index=chunk_index,
                        metadata=build_chunk_metadata(document, section),
                    )
                )

        return chunks

    def chunk_documents(self, documents: list[MarkdownDocument]) -> list[DocumentChunk]:
        """Chunk multiple Markdown documents in input order."""
        chunks: list[DocumentChunk] = []
        for document in documents:
            chunks.extend(self.chunk_document(document))
        return chunks


def split_markdown_sections(body: str, fallback_title: str) -> list[MarkdownSection]:
    """Split a markdown document into sections.

    Parameters
    ----------
    body : str
        The body of the markdown document.
    fallback_title : str
        The fallback title of the markdown document.

    Returns
    -------
    sections : list[MarkdownSection]
        The sections of the markdown document.
    """
    lines = body.splitlines()
    document_title = fallback_title
    h1_seen = False
    sections: list[MarkdownSection] = []
    current_title: str | None = None
    current_lines: list[str] = []

    for line in lines:
        heading = HEADING_PATTERN.match(line)
        if heading is None:
            current_lines.append(line)
            continue

        level = len(heading.group(1))
        title = heading.group(2).strip()

        if level == 1 and not h1_seen:
            document_title = title
            h1_seen = True
            current_lines = []
            continue

        if level <= 2:
            _append_section(sections, document_title, current_title, current_lines)
            current_title = title
            current_lines = [line]
            continue

        current_lines.append(line)

    _append_section(sections, document_title, current_title, current_lines)

    if not sections:
        stripped_body = body.strip()
        if stripped_body:
            return [
                MarkdownSection(
                    title=document_title,
                    section_path=(document_title,),
                    content=stripped_body,
                )
            ]
    return sections


def split_large_section(content: str, max_words: int, overlap_words: int) -> list[str]:
    """Split an oversized section into overlapping chunks."""
    if count_words(content) <= max_words:
        return [content.strip()]

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", content)
        if paragraph.strip()
    ]
    chunks: list[str] = []
    current_paragraphs: list[str] = []

    for paragraph in paragraphs:
        paragraph_words = count_words(paragraph)
        current_text = "\n\n".join(current_paragraphs)
        current_words = count_words(current_text)

        if paragraph_words > max_words:
            if current_paragraphs and not _is_heading_only(current_paragraphs):
                chunks.append(current_text)
                current_paragraphs = []
                chunks.extend(_split_words(paragraph, max_words, overlap_words))
                continue

            prefix = "\n\n".join(current_paragraphs).strip()
            chunks.extend(
                _split_words_with_prefix(paragraph, prefix, max_words, overlap_words)
            )
            current_paragraphs = []
            continue

        if current_paragraphs and current_words + paragraph_words > max_words:
            chunks.append(current_text)
            current_paragraphs = [_tail_words(current_text, overlap_words), paragraph]
        else:
            current_paragraphs.append(paragraph)

    if current_paragraphs:
        chunks.append(
            "\n\n".join(
                paragraph for paragraph in current_paragraphs if paragraph
            ).strip()
        )

    return [chunk for chunk in chunks if chunk]


def count_words(text: str) -> int:
    """Count whitespace-delimited words in text."""
    return len(re.findall(r"\S+", text))


def build_chunk_metadata(
    document: MarkdownDocument, section: MarkdownSection
) -> dict[str, Any]:
    """Build retrieval metadata for a document section."""
    keys = [
        "title",
        "doc_type",
        "service_id",
        "owner_group",
        "visibility",
        "status",
        "created_at",
        "updated_at",
        "related_records",
        "source_facts",
        "difficulty_tags",
        "servicenow",
    ]
    metadata = {
        key: document.metadata.get(key) for key in keys if key in document.metadata
    }
    metadata["section_title"] = section.title
    return metadata


def _append_section(
    sections: list[MarkdownSection],
    document_title: str,
    current_title: str | None,
    current_lines: list[str],
) -> None:
    content = "\n".join(current_lines).strip()
    if not content:
        return

    section_title = current_title or document_title
    section_path = (
        (document_title,) if current_title is None else (document_title, current_title)
    )
    sections.append(
        MarkdownSection(
            title=section_title,
            section_path=section_path,
            content=content,
        )
    )


def _split_words(text: str, max_words: int, overlap_words: int) -> list[str]:
    words = re.findall(r"\S+", text)
    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(0, end - overlap_words)

    return chunks


def _tail_words(text: str, word_count: int) -> str:
    if word_count == 0:
        return ""
    words = re.findall(r"\S+", text)
    return " ".join(words[-word_count:])


def _is_heading_only(paragraphs: list[str]) -> bool:
    return len(paragraphs) == 1 and HEADING_PATTERN.match(paragraphs[0]) is not None


def _split_words_with_prefix(
    text: str,
    prefix: str,
    max_words: int,
    overlap_words: int,
) -> list[str]:
    if not prefix:
        return _split_words(text, max_words, overlap_words)

    prefix_words = count_words(prefix)
    if prefix_words >= max_words:
        return _split_words(text, max_words, overlap_words)

    body_max_words = max_words - prefix_words
    body_overlap_words = min(overlap_words, max(0, body_max_words - 1))
    body_chunks = _split_words(text, body_max_words, body_overlap_words)
    return [f"{prefix}\n\n{chunk}" for chunk in body_chunks]
