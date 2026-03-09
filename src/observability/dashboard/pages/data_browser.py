"""Data Browser page – browse ingested documents, chunks, and images.

Layout:
1. Collection selector (sidebar)
2. Document list with chunk counts
3. Expandable document detail → chunk cards with text + metadata
4. Image preview gallery
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.observability.dashboard.i18n import t
from src.observability.dashboard.services.data_service import get_data_service


def render() -> None:
    """Render the Data Browser page."""
    st.header(t("🔍 Data Browser", "🔍 数据浏览"))

    try:
        svc = get_data_service()
    except Exception as exc:
        st.error(t("Failed to initialise DataService: ", "初始化 DataService 失败：") + str(exc))
        return

    # ── Collection selector ────────────────────────────────────────
    collection = st.text_input(
        t("Collection name (leave blank = `default`)", "Collection 名称（留空则为 `default`）"),
        value="default",
        key="db_collection_filter",
    )
    coll_arg = collection.strip() if collection.strip() else None

    # ── Document list ──────────────────────────────────────────────
    try:
        docs = svc.list_documents(coll_arg)
    except Exception as exc:
        st.error(t("Failed to load documents: ", "加载文档失败：") + str(exc))
        return

    if not docs:
        st.info(t("No documents found. Ingest some data first!", "未找到文档。请先导入一些数据。"))
        return

    st.subheader(t(f"📄 Documents ({len(docs)})", f"📄 文档 ({len(docs)})"))

    for idx, doc in enumerate(docs):
        source_name = Path(doc["source_path"]).name
        label = f"📑 {source_name}  —  {doc['chunk_count']} chunks · {doc['image_count']} images"
        with st.expander(label, expanded=(len(docs) == 1)):
            # ── Document metadata ──────────────────────────────────
            col_a, col_b, col_c = st.columns(3)
            col_a.metric(t("Chunks", "分块数"), doc["chunk_count"])
            col_b.metric(t("Images", "图片数"), doc["image_count"])
            col_c.metric(t("Collection", "Collection"), doc.get("collection", "—"))
            st.caption(
                f"**{t('Source', '来源')}:** {doc['source_path']}  ·  "
                f"**Hash:** `{doc['source_hash'][:16]}…`  ·  "
                f"**{t('Processed', '处理时间')}:** {doc.get('processed_at', '—')}"
            )

            st.divider()

            # ── Chunk cards ────────────────────────────────────────
            chunks = svc.get_chunks(doc["source_hash"], coll_arg)
            if chunks:
                st.markdown(t(f"### 📦 Chunks ({len(chunks)})", f"### 📦 分块 ({len(chunks)})"))
                for cidx, chunk in enumerate(chunks):
                    text = chunk.get("text", "")
                    meta = chunk.get("metadata", {})
                    chunk_id = chunk["id"]

                    # Title from metadata or first line
                    title = meta.get("title", "")
                    if not title:
                        title = text[:60].replace("\n", " ").strip()
                        if len(text) > 60:
                            title += "…"

                    with st.container(border=True):
                        st.markdown(
                            f"**{t('Chunk', '分块')} {cidx + 1}** · `{chunk_id[-16:]}` · "
                            f"{len(text)} {t('chars', '字符')}"
                        )
                        # Show the actual chunk text (scrollable)
                        _height = max(120, min(len(text) // 2, 600))
                        st.text_area(
                            t("Content", "内容"),
                            value=text,
                            height=_height,
                            disabled=True,
                            key=f"chunk_text_{idx}_{cidx}",
                            label_visibility="collapsed",
                        )
                        # Expandable metadata
                        with st.expander(t("📋 Metadata", "📋 元数据"), expanded=False):
                            st.json(meta)
            else:
                st.caption(t("No chunks found in vector store for this document.", "向量库中没有找到该文档的分块。"))

            # ── Image preview ──────────────────────────────────────
            images = svc.get_images(doc["source_hash"], coll_arg)
            if images:
                st.divider()
                st.markdown(t(f"### 🖼️ Images ({len(images)})", f"### 🖼️ 图片 ({len(images)})"))
                img_cols = st.columns(min(len(images), 4))
                for iidx, img in enumerate(images):
                    with img_cols[iidx % len(img_cols)]:
                        img_path = Path(img.get("file_path", ""))
                        if img_path.exists():
                            st.image(str(img_path), caption=img["image_id"], width=200)
                        else:
                            st.caption(t(f"{img['image_id']} (file missing)", f"{img['image_id']}（文件缺失）"))
