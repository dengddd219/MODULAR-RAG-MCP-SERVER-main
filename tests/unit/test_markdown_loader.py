"""Unit tests for Markdown Loader."""

from pathlib import Path

import pytest

from src.core.types import Document
from src.libs.loader.markdown_loader import MarkdownLoader


class TestMarkdownLoader:
    """Tests for MarkdownLoader."""

    def test_load_returns_document(self, tmp_path: Path) -> None:
        md_file = tmp_path / "readme.md"
        md_file.write_text("# Hello\n\nThis is **markdown**.", encoding="utf-8")
        loader = MarkdownLoader()
        doc = loader.load(md_file)
        assert isinstance(doc, Document)
        assert doc.text == "# Hello\n\nThis is **markdown**."
        assert doc.metadata["source_path"] == str(md_file.resolve())
        assert doc.metadata["doc_type"] == "markdown"
        assert "doc_hash" in doc.metadata
        assert doc.id.startswith("doc_")
        assert len(doc.id) == 20  # "doc_" + 16 hex chars

    def test_extracts_title_from_heading(self, tmp_path: Path) -> None:
        md_file = tmp_path / "doc.md"
        md_file.write_text("# My Title\n\nBody here.", encoding="utf-8")
        loader = MarkdownLoader()
        doc = loader.load(md_file)
        assert doc.metadata.get("title") == "My Title"

    def test_rejects_non_md_extension(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "file.txt"
        txt_file.write_text("plain text", encoding="utf-8")
        loader = MarkdownLoader()
        with pytest.raises(ValueError, match="not a Markdown file"):
            loader.load(txt_file)

    def test_file_not_found(self) -> None:
        loader = MarkdownLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/doc.md")

    def test_same_content_same_doc_id(self, tmp_path: Path) -> None:
        content = "# Same\n\nContent"
        a = tmp_path / "a.md"
        b = tmp_path / "b.md"
        a.write_text(content, encoding="utf-8")
        b.write_text(content, encoding="utf-8")
        loader = MarkdownLoader()
        doc_a = loader.load(a)
        doc_b = loader.load(b)
        assert doc_a.id == doc_b.id
        assert doc_a.metadata["doc_hash"] == doc_b.metadata["doc_hash"]
