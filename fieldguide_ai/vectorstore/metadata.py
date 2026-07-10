import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from fieldguide_ai.ingestion.models import DocumentChunk

VectorMetadataValue = str | int | float | bool | None


def serialize_chunk_metadata(chunk: DocumentChunk) -> dict[str, VectorMetadataValue]:
    metadata: dict[str, VectorMetadataValue] = {
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


def _normalize_metadata_value(value: Any) -> VectorMetadataValue:
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
