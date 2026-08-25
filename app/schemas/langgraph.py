"""
LangGraph Questionnaire Breakdown Schema Definitions
定義 LangGraph 狀態機圖運作所需的 State、大綱節點與目錄錨點型別。
"""
from __future__ import annotations

from typing import Any, NotRequired, TypedDict


class OutlineItem(TypedDict):
    """單一章節/小節大綱節點。"""
    section_id: str          # 純化主章節代號 (如 "A", "Chapter I", "B")
    subsection_id: str       # 小節代號 (如 "A.1", "1.2", "Chapter I.A")
    title: str               # 標題領域名稱 (如 "General Information", "Water Quality")
    page: int                # 所在頁碼 (1-indexed)
    position: str            # 頁內預估位置: "top" | "middle" | "bottom" | "full_page"
    is_major_section: bool   # 是否為新的主章節/Part/Chapter 起始
    depiction: str           # 前言說明文字 (若有)


class TocAnchor(TypedDict):
    """目錄錨點。"""
    id: str
    title: str
    start_page: int
    end_page: int
    is_calibrated: NotRequired[bool]


class QuestionnaireState(TypedDict):
    """LangGraph 擬人化問卷解析狀態機核心 State。"""
    # ── 靜態文件資訊 ──
    filename: NotRequired[str]
    total_pages: int
    image_pages: list[dict[str, Any]]

    # ── 全篇導航地圖與目錄大綱 ──
    toc_anchors: list[TocAnchor]
    page_section_map: dict[int, list[OutlineItem]]
    global_outline_tree: list[OutlineItem]
    section_depictions: dict[str, str]
    section_order: list[str]

    # ── 逐頁循序工作記憶 ──
    current_index: int
    active_chapter: str
    heading_stack: list[str]
    pending_context: str
    reading_memory_summary: str
    page_history: dict[int, dict[str, Any]]
    chapter_milestones: list[str]
    final_questions: list[dict[str, Any]]

    # ── 動態翻頁與回讀控制 ──
    loop_count: int
    is_back_reading: bool
    back_read_count_for_page: int
    requested_back_read_pages: list[int]
    back_read_trail: list[str]
