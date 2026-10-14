"""ParentDocument 存储：父块落盘，子块进向量库。"""
from __future__ import annotations

import json
import os
from typing import Any

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_STORE_PATH = os.path.join(_BASE_DIR, "parent_store.json")
CHROMA_DIR = os.path.join(_BASE_DIR, "chroma_db")

# 练习用尺寸：父块完整上下文，子块精准检索
PARENT_CHUNK_SIZE = 1000
PARENT_CHUNK_OVERLAP = 100
CHILD_CHUNK_SIZE = 150
CHILD_CHUNK_OVERLAP = 30


def source_title_from_path(filepath: str) -> str:
    return os.path.splitext(os.path.basename(filepath))[0]


def source_title_from_doc(content: str, filepath: str = "") -> str:
    """优先取文首【标题】，否则退回文件名。"""
    text = (content or "").lstrip()
    if text.startswith("【"):
        end = text.find("】")
        if end > 1:
            return text[1:end].strip() or source_title_from_path(filepath)
    return source_title_from_path(filepath) if filepath else "unknown"


def make_source_prefix(source_title: str, category: str) -> str:
    return f"[来源: {source_title}] [分类: {category}]"


def save_parent_store(parents: dict[str, dict[str, Any]]) -> None:
    with open(PARENT_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(parents, f, ensure_ascii=False, indent=2)


def load_parent_store() -> dict[str, dict[str, Any]]:
    if not os.path.isfile(PARENT_STORE_PATH):
        return {}
    with open(PARENT_STORE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_parent(parent_id: str) -> dict[str, Any] | None:
    return load_parent_store().get(parent_id)
