"""Knowledge-base loading. One Markdown file per document: '# Title' line, then the body."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from app.config import KB_DIR

# Split after . ! ? followed by whitespace. The KB is written without abbreviations
# ("e.g.", "Dr.") so this simple rule is exact for our documents; tests enforce it.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_BOUNDARY.split(text.strip()) if s.strip()]


@dataclass(frozen=True)
class KBDocument:
    doc_id: str
    title: str
    text: str

    @property
    def sentences(self) -> list[str]:
        return split_sentences(self.text)


def parse_document(path: Path) -> KBDocument:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    if not lines or not lines[0].startswith("# "):
        raise ValueError(f"{path.name}: first line must be a '# Title' heading")
    body = " ".join(line.strip() for line in lines[1:] if line.strip())
    if not body:
        raise ValueError(f"{path.name}: document body is empty")
    return KBDocument(doc_id=path.stem, title=lines[0][2:].strip(), text=body)


def load_kb(kb_dir: Path = KB_DIR) -> list[KBDocument]:
    paths = sorted(kb_dir.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"No knowledge-base documents found in {kb_dir}")
    return [parse_document(p) for p in paths]
