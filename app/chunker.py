from __future__ import annotations

import hashlib
import json
import re

from app.models import Chunk, LoadedSection


_WHITESPACE_RE = re.compile(r"[ \t]+")
_BREAK_CHARS = set("。！？!?；;\n")


def normalize_text(text: str) -> str:
    lines = [_WHITESPACE_RE.sub(" ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _choose_end(text: str, start: int, chunk_size: int) -> int:
    target = min(start + chunk_size, len(text))
    if target == len(text):
        return target
    lower_bound = start + int(chunk_size * 0.65)
    for index in range(target, lower_bound, -1):
        if text[index - 1] in _BREAK_CHARS:
            return index
    return target


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("切分参数无效")
    cleaned = normalize_text(text)
    if not cleaned:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = _choose_end(cleaned, start, chunk_size)
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(cleaned):
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_sections(
    sections: list[LoadedSection], chunk_size: int, overlap: int
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in sections:
        for index, text in enumerate(split_text(section.text, chunk_size, overlap), start=1):
            identity = "|".join(
                [
                    section.domain,
                    section.source,
                    str(section.page),
                    str(section.section),
                    str(index),
                    text,
                    json.dumps(section.metadata, ensure_ascii=False, sort_keys=True),
                ]
            )
            chunk_id = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16]
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=text,
                    source=section.source,
                    domain=section.domain,
                    doc_type=section.doc_type,
                    page=section.page,
                    section=section.section,
                    metadata=section.metadata,
                )
            )
    return chunks

