"""Markdown Loader for .md files.

This module loads plain Markdown files into the standardized Document format,
with consistent metadata (source_path, doc_type, doc_hash, optional title).
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader

logger = logging.getLogger(__name__)


class MarkdownLoader(BaseLoader):
    """Loader for Markdown (.md) files.

    Reads file content as UTF-8 and returns a Document with:
    - id: doc_{first_16_chars_of_sha256}
    - text: raw file content (already Markdown)
    - metadata: source_path, doc_type="markdown", doc_hash, optional title from first # heading
    """

    def load(self, file_path: str | Path) -> Document:
        """Load and parse a Markdown file.

        Args:
            file_path: Path to the .md file.

        Returns:
            Document with text and metadata.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file is not a .md file.
        """
        path = self._validate_file(file_path)
        if path.suffix.lower() != ".md":
            raise ValueError(f"File is not a Markdown file: {path}")

        doc_hash = self._compute_file_hash(path)
        doc_id = f"doc_{doc_hash[:16]}"

        try:
            text_content = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.error(f"Failed to read Markdown file {path}: {e}")
            raise RuntimeError(f"Markdown file read failed: {e}") from e

        metadata: Dict[str, Any] = {
            "source_path": str(path),
            "doc_type": "markdown",
            "doc_hash": doc_hash,
        }
        title = self._extract_title(text_content)
        if title:
            metadata["title"] = title

        return Document(
            id=doc_id,
            text=text_content,
            metadata=metadata,
        )

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file content."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _extract_title(self, text: str) -> Optional[str]:
        """Extract title from first Markdown heading or first non-empty line."""
        lines = text.split("\n")
        for line in lines[:20]:
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()
        for line in lines[:10]:
            line = line.strip()
            if line:
                return line
        return None
