"""Ingestion Manager page – upload files, trigger ingestion, delete documents.

Layout:
1. File uploader + collection selector
2. Ingest button → progress bar (using on_progress callback)
3. Document list with delete buttons
"""

from __future__ import annotations

from pathlib import Path
import json
import shutil

import streamlit as st

from src.observability.dashboard.services.data_service import get_data_service


def _get_collections_override_path() -> Path:
    """Return path to dashboard-level logical collection override mapping."""
    from src.core.settings import resolve_path

    return resolve_path(Path("data/db/dashboard_collections.json"))


def _load_collection_overrides() -> dict[str, str]:
    """Load source_hash → logical_collection mapping used only by the dashboard.

    This acts as a light-weight tagging layer on top of the physical storage
    collections managed by the core pipeline.
    """
    path = _get_collections_override_path()
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            # ensure all keys/values are strings
            return {str(k): str(v) for k, v in data.items()}
        return {}
    except Exception:
        return {}


def _save_collection_overrides(mapping: dict[str, str]) -> None:
    """Persist logical collection overrides to disk."""
    path = _get_collections_override_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)


def _run_ingestion(
    uploaded_file: "st.runtime.uploaded_file_manager.UploadedFile",
    collection: str,
    progress_bar: "st.delta_generator.DeltaGenerator",
    status_text: "st.delta_generator.DeltaGenerator",
) -> None:
    """Save the uploaded file to a stable location and run the pipeline.

    The raw file is stored under ``data/documents_all/{collection}/`` so that
    downstream components (and the dashboard) can reliably reference it for
    intent-based views or re-processing.
    """
    from src.core.settings import load_settings, resolve_path
    from src.core.trace import TraceContext, TraceCollector
    from src.ingestion.pipeline import IngestionPipeline

    settings = load_settings()

    # Treat `collection` from the UI as a *logical* label only.
    logical_collection = (collection or "").strip() or "default"

    # Persist uploaded file into a stable "all documents" library path.
    # Layout: data/documents_all/{logical_collection}/{filename}
    all_lib_dir = resolve_path(Path(f"data/documents_all/{logical_collection}"))
    all_lib_dir.mkdir(parents=True, exist_ok=True)

    target_path = all_lib_dir / uploaded_file.name
    try:
        with open(target_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    except Exception as exc:
        status_text.error(f"Failed to save uploaded file: {exc}")
        return

    _STAGE_LABELS = {
        "integrity": "🔍 Checking file integrity…",
        "load": "📄 Loading document…",
        "split": "✂️ Chunking document…",
        "transform": "🔄 Transforming chunks (LLM refine + enrich)…",
        "embed": "🔢 Encoding vectors…",
        "upsert": "💾 Storing to database…",
    }

    def on_progress(stage: str, current: int, total: int) -> None:
        frac = (current - 1) / total  # stage just started, show partial progress
        label = _STAGE_LABELS.get(stage, stage)
        progress_bar.progress(frac, text=f"[{current}/{total}] {label}")
        status_text.caption(label)

    trace = TraceContext(trace_type="ingestion")
    trace.metadata["source_path"] = uploaded_file.name
    trace.metadata["collection"] = logical_collection
    trace.metadata["source"] = "dashboard"

    try:
        # Physical collection for the core pipeline remains stable (e.g. "default").
        # Logical grouping for the dashboard is managed separately via overrides.
        pipeline = IngestionPipeline(settings, collection="default")
        result = pipeline.run(
            file_path=str(target_path),
            trace=trace,
            on_progress=on_progress,
        )
        progress_bar.progress(1.0, text="✅ Complete")
        status_text.success(f"Successfully ingested **{uploaded_file.name}** into collection **{collection}**.")

        # Persist / update logical collection override using the final doc hash.
        try:
            doc_hash = result.doc_id or (result.stages or {}).get("integrity", {}).get("file_hash")
            if isinstance(doc_hash, str):
                overrides = _load_collection_overrides()
                overrides[doc_hash] = logical_collection
                _save_collection_overrides(overrides)
        except Exception:
            # Non-fatal: logical collection override is best-effort.
            pass

        # ── Document intent review & manual override UI ─────────────────
        try:
            intent_info = (result.stages or {}).get("transform", {}).get("document_intent")
        except Exception:
            intent_info = None

        predicted_value: str | None = None
        confidence: float | None = None
        distribution: dict | None = None

        # Backwards compatibility: older PipelineResult may store a plain string
        if isinstance(intent_info, dict):
            predicted_value = intent_info.get("value")
            confidence = intent_info.get("confidence")
            distribution = intent_info.get("distribution") or {}
        elif isinstance(intent_info, str):
            predicted_value = intent_info

        if predicted_value:
            st.subheader("🧠 Document Intent (Model Suggestion & Manual Label)")

            # Human-friendly display label
            def _pretty_label(value: str) -> str:
                return value.replace("_", " ").title()

            display_default = _pretty_label(predicted_value)

            if confidence is not None:
                st.markdown(
                    f"**Model suggestion**: `{predicted_value}` "
                    f"（显示名：**{display_default}**，置信度：**{confidence:.1%}**）"
                )
            else:
                st.markdown(
                    f"**Model suggestion**: `{predicted_value}` "
                    f"（显示名：**{display_default}**）"
                )

            # Optional: show per-class probability table if available
            if isinstance(distribution, dict) and distribution:
                # Map raw labels to pretty names when possible
                intent_rows = []
                for raw_label, prob in sorted(distribution.items(), key=lambda kv: kv[1], reverse=True):
                    intent_rows.append(
                        {
                            "raw_label": raw_label,
                            "intent_display": _pretty_label(raw_label),
                            "probability": round(float(prob), 4),
                        }
                    )
                with st.expander("查看模型对各意图类别的概率分布", expanded=False):
                    st.dataframe(intent_rows, hide_index=True)

            st.markdown("请选择本篇文档的**最终意图标签**（模型结果仅作为默认参考，上传者拥有最终决定权）：")

            INTENT_OPTIONS = [
                "unknown",     # 无法归类 / 与五大分类均无关
                "chitchat",
                "escalation",
                "fabric_care",
                "returns",
                "styling",
            ]

            # Ensure the default is one of the options; otherwise fall back到 unknown
            default_value = predicted_value if predicted_value in INTENT_OPTIONS else "unknown"

            final_intent = st.selectbox(
                "最终分类标签",
                options=INTENT_OPTIONS,
                format_func=lambda v: _pretty_label(v),
                index=INTENT_OPTIONS.index(default_value),
                key=f"final_doc_intent_{uploaded_file.name}",
                help="从 5 个业务意图中选择一个作为该文档的最终标签。",
            )

            # 二阶段确认：Start Ingestion 负责“进入 all 库”（已完成向量/BM25 等存储），
            # 下方按钮用于用户确认最终标签，并将原始文件拷贝到对应 intent folder。
            if st.button("✅ 确认最终分类", key=f"confirm_intent_{uploaded_file.name}"):
                try:
                    # 计算源文件在 "all documents" 库中的路径
                    from src.core.settings import resolve_path

                    source_collection = collection
                    if source_collection in ("all", ""):
                        source_collection = "default"

                    all_lib_dir = resolve_path(Path(f"data/documents_all/{source_collection}"))
                    source_path = all_lib_dir / uploaded_file.name

                    if not source_path.exists():
                        st.warning(
                            f"找不到源文件 `{source_path}`，但已记录最终标签："
                            f"**{_pretty_label(final_intent)}**。"
                        )
                    else:
                        # 按最终 intent 归档到 intent-specific 文件夹
                        intent_dir = resolve_path(
                            Path(f"data/documents_by_intent/{final_intent}/{source_collection}")
                        )
                        intent_dir.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source_path, intent_dir / uploaded_file.name)

                        st.success(
                            f"已确认最终标签：**{_pretty_label(final_intent)}** "
                            f"(内部值：`{final_intent}`)。\n\n"
                            f"- 已在 all 库中可用（collection=`{source_collection}`）\n"
                            f"- 并已归档到 `data/documents_by_intent/{final_intent}/{source_collection}/`"
                        )
                except Exception as exc:
                    st.error(f"确认最终分类时出错：{exc}")
    except Exception as exc:
        status_text.error(f"Ingestion failed: {exc}")
    finally:
        TraceCollector().collect(trace)


def render() -> None:
    """Render the Ingestion Manager page."""
    st.header("📥 Ingestion Manager")

    # 在 Session State 中记住当前选中的 collection 过滤器；
    # 若尚未选择，则默认使用 "all"（文档列表显示全部，上传时实际写入 default）。
    current_collection = st.session_state.get("collection_filter", "all")

    # ── Upload section ─────────────────────────────────────────────
    st.subheader("📤 Upload & Ingest")

    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded = st.file_uploader(
            "Select a file to ingest",
            type=["pdf", "txt", "md", "docx"],
            key="ingest_uploader",
        )
    with col2:
        # 右侧暂时空出，用于后续扩展（例如展示 collection 说明等）。
        st.empty()

    # 仅在用户显式点击“开始摄取”后，才真正将文件写入知识库。
    # 注意：摄取使用“当前 Collection 过滤器”作为目标库；当为 all 时，默认写入 default。
    if uploaded is not None:
        st.caption("已选择文件：**%s**" % uploaded.name)

        if st.button("🚀 Start Ingestion", key="btn_ingest"):
            # 根据当前过滤器决定本次上传写入哪个 collection
            target_collection = (
                "default" if current_collection in ("all", "") else current_collection
            )

            progress_bar = st.progress(0, text="Preparing…")
            status_text = st.empty()

            # 真正执行摄取（包含分类、向量入库、BM25 建索引等）
            _run_ingestion(uploaded, target_collection, progress_bar, status_text)

    st.divider()

    # ── Document management section ────────────────────────────────
    st.subheader("🗑️ Manage Documents")

    # Collection 选择框放在 Manage Documents 下方，既用作列表筛选，也作为上传时的“目标库”选择。
    COLLECTION_FILTER_OPTIONS = [
        "all",          # 显示所有已成功摄取的文档
        "default",      # 通用 / 与服装客服无关
        "chitchat",     # 闲聊
        "escalation",   # 投诉 / 升级
        "fabric_care",  # 面料洗护
        "returns",      # 退货 / 售后
        "styling",      # 穿搭建议
    ]

    def _collection_label(value: str) -> str:
        mapping = {
            "all": "all（显示所有 collection）",
            "default": "default（通用库）",
            "chitchat": "chitchat（闲聊库）",
            "escalation": "escalation（投诉/升级库）",
            "fabric_care": "fabric_care（洗护知识库）",
            "returns": "returns（退货/售后库）",
            "styling": "styling（穿搭建议库）",
        }
        return mapping.get(value, value)

    collection_filter = st.selectbox(
        "Collection 过滤器",
        options=COLLECTION_FILTER_OPTIONS,
        index=COLLECTION_FILTER_OPTIONS.index(current_collection)
        if current_collection in COLLECTION_FILTER_OPTIONS
        else 0,
        format_func=_collection_label,
        key="collection_filter",
        help=(
            "选择要查看/管理的知识库；选择 all 时列表显示所有已摄取文档，"
            "但新增上传会默认写入 default。"
        ),
    )

    # 更新本次渲染的 current_collection（供上方 Upload 区域展示使用）
    current_collection = collection_filter

    # 加载 Dashboard 级别的“逻辑 collection” 覆盖映射：source_hash -> logical_collection
    overrides = _load_collection_overrides()

    try:
        svc = get_data_service()
        # 始终从底层加载"物理"所有文档，然后在 Dashboard 层按逻辑标签过滤。
        docs = svc.list_documents(collection=None)
    except Exception as exc:
        st.error(f"Failed to load documents: {exc}")
        return

    # 为每个文档打上 logical_collection 标签
    for doc in docs:
        logical_coll = overrides.get(doc.get("source_hash", ""), doc.get("collection", "default"))
        doc["logical_collection"] = logical_coll

    # 根据当前过滤器做逻辑筛选
    if collection_filter != "all":
        docs = [d for d in docs if d.get("logical_collection") == collection_filter]

    if not docs:
        st.info("No documents ingested yet.")
        return

    for idx, doc in enumerate(docs):
        col_info, col_actions = st.columns([4, 2])
        with col_info:
            # 第一行仅展示文件名
            st.markdown(f"**{doc['source_path']}**")
            # 第二行单独展示 collection / chunks / images 信息，视觉上更清晰
            st.caption(
                f"collection: `{doc.get('logical_collection', '—')}` | "
                f"chunks: {doc['chunk_count']} | "
                f"images: {doc['image_count']}"
            )
        with col_actions:
            # 左侧：修改 collection（通过逻辑标签实现“移动”效果）
            options = [
                "default",
                "chitchat",
                "escalation",
                "fabric_care",
                "returns",
                "styling",
            ]
            current_logical = doc.get("logical_collection", "default")
            # 当前 logical_collection 可能不在预设列表中（例如自定义 collection 名），
            # 此时回退到第一个选项，避免 .index() 抛出 ValueError。
            try:
                default_index = options.index(current_logical)
            except ValueError:
                default_index = 0

            new_collection = st.selectbox(
                "Move to",
                options=options,
                index=default_index,
                key=f"move_collection_{idx}",
            )
            move_col, delete_col = st.columns(2)
            with move_col:
                if st.button("↪️ Move", key=f"move_{idx}"):
                    try:
                        # 仅更新 Dashboard 层的逻辑 collection 标签，不触碰底层物理存储。
                        overrides = _load_collection_overrides()
                        source_hash = doc.get("source_hash", "")
                        if source_hash:
                            overrides[source_hash] = new_collection
                            _save_collection_overrides(overrides)

                        old_collection = doc.get("logical_collection", "default")
                        st.success(
                            f"Updated logical collection from `{old_collection}` "
                            f"to `{new_collection}` for this document."
                        )
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Move failed: {exc}")
            with delete_col:
                if st.button("🗑️ Delete", key=f"del_{idx}"):
                    try:
                        result = svc.delete_document(
                            source_path=doc["source_path"],
                            collection=doc.get("collection", "default"),
                            source_hash=doc.get("source_hash"),
                        )
                        if result.success:
                            st.success(
                                f"Deleted: {result.chunks_deleted} chunks, "
                                f"{result.images_deleted} images removed."
                            )
                            st.rerun()
                        else:
                            st.warning(f"Partial delete. Errors: {result.errors}")
                    except Exception as exc:
                        st.error(f"Delete failed: {exc}")
