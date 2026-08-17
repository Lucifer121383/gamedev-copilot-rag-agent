from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from docx import Document
from pypdf import PdfReader

from app.models import LoadedSection, VALID_DOMAINS


SUPPORTED_SUFFIXES = {".txt", ".md", ".log", ".pdf", ".docx", ".json", ".csv"}


def _read_text(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法识别文件编码: {path.name}")


def _scalar_metadata(record: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in record.items()
        if isinstance(value, (str, int, float, bool)) and str(value).strip()
    }


def _record_text(record: dict[str, Any]) -> str:
    lines: list[str] = []
    for key, value in record.items():
        if value is None or value == "":
            continue
        if isinstance(value, list):
            rendered = "；".join(str(item) for item in value)
        elif isinstance(value, dict):
            rendered = json.dumps(value, ensure_ascii=False)
        else:
            rendered = str(value)
        lines.append(f"{key}: {rendered}")
    return "\n".join(lines)


def load_document(
    path: Path,
    *,
    domain: str,
    doc_type: str,
    display_source: str | None = None,
) -> list[LoadedSection]:
    if domain not in VALID_DOMAINS:
        raise ValueError(f"不支持的工作空间: {domain}")
    source = display_source or path.name
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"不支持的文件格式: {suffix}")
    base_metadata = {
        "domain": domain,
        "doc_type": doc_type,
        "filename": path.name,
    }

    if suffix in {".txt", ".md", ".log"}:
        text = _read_text(path).strip()
        return [
            LoadedSection(text, source, domain, doc_type, metadata=base_metadata)
        ] if text else []

    if suffix == ".json":
        payload = json.loads(_read_text(path))
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict):
            records = payload.get("records", [payload])
        else:
            raise ValueError(f"JSON内容必须是对象或对象列表: {path.name}")
        if not isinstance(records, list):
            raise ValueError(f"records必须是列表: {path.name}")
        sections: list[LoadedSection] = []
        for index, record in enumerate(records, start=1):
            if not isinstance(record, dict):
                continue
            text = _record_text(record).strip()
            if text:
                metadata = {**base_metadata, **_scalar_metadata(record), "record": index}
                heading = str(record.get("id") or record.get("bug_id") or f"记录{index}")
                sections.append(
                    LoadedSection(
                        text, source, domain, doc_type, section=heading, metadata=metadata
                    )
                )
        return sections

    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            records = list(csv.DictReader(file))
        sections = []
        for index, record in enumerate(records, start=1):
            text = _record_text(record).strip()
            if not text:
                continue
            sections.append(
                LoadedSection(
                    text,
                    source,
                    domain,
                    doc_type,
                    section=str(
                        record.get("id") or record.get("case_id") or f"记录{index}"
                    ),
                    metadata={**base_metadata, **_scalar_metadata(record), "record": index},
                )
            )
        return sections

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        sections = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                sections.append(
                    LoadedSection(
                        text,
                        source,
                        domain,
                        doc_type,
                        page=page_number,
                        metadata=base_metadata,
                    )
                )
        return sections

    document = Document(str(path))
    sections: list[LoadedSection] = []
    heading = "正文"
    paragraphs: list[str] = []

    def flush() -> None:
        if paragraphs:
            sections.append(
                LoadedSection(
                    "\n".join(paragraphs),
                    source,
                    domain,
                    doc_type,
                    section=heading,
                    metadata=base_metadata,
                )
            )
            paragraphs.clear()

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        if paragraph.style and paragraph.style.name.lower().startswith("heading"):
            flush()
            heading = text
        else:
            paragraphs.append(text)
    flush()
    return sections


def load_workspace(
    domain_dir: Path, upload_dir: Path, domain: str
) -> tuple[list[LoadedSection], list[str]]:
    sections: list[LoadedSection] = []
    files: list[str] = []
    seen: set[Path] = set()
    for root, prefix in ((domain_dir, domain), (upload_dir, f"uploads/{domain}")):
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            relative = path.relative_to(root)
            doc_type = relative.parts[0] if len(relative.parts) > 1 else "uploads"
            label = f"{prefix}/{relative.as_posix()}"
            loaded = load_document(
                path, domain=domain, doc_type=doc_type, display_source=label
            )
            if loaded:
                files.append(label)
                sections.extend(loaded)
    return sections, files

