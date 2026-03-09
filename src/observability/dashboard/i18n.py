from __future__ import annotations

import streamlit as st

LANG_EN = "English"
LANG_ZH = "中文"
LANGUAGE_OPTIONS = [LANG_EN, LANG_ZH]


def ensure_language() -> None:
    """Ensure a dashboard-wide language flag exists."""
    if "dashboard_language" not in st.session_state:
        st.session_state.dashboard_language = LANG_EN


def get_language() -> str:
    """Return the current dashboard language."""
    ensure_language()
    return st.session_state.get("dashboard_language", LANG_EN)


def is_chinese() -> bool:
    """Whether the current UI language is Chinese."""
    return get_language() == LANG_ZH


def t(en: str, zh: str) -> str:
    """Translate a string using the current dashboard language."""
    return zh if is_chinese() else en


def localized_mode(mode: str) -> str:
    """Translate chat mode labels."""
    mapping = {
        "Fast": t("Fast", "快速"),
        "Think": t("Think", "思考"),
        "Pro": "Pro",
    }
    return mapping.get(mode, mode)


def mode_options() -> list[str]:
    """Return canonical chat mode options."""
    return ["Fast", "Think", "Pro"]

