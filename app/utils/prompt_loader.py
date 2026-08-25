"""
Prompt Loader Utility
載入外部化的 System Prompt 與 Prompt 模板設定檔案。
"""
from functools import lru_cache
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# 指向 app/resources/prompts 目錄
_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "resources" / "prompts"


@lru_cache(maxsize=64)
def load_prompt(category: str, name: str) -> str:
    """
    從 app/resources/prompts/{category}/{name}.txt 載入指定的 Prompt 內容。
    具備 LRU Cache，避免重複磁碟 I/O。

    :param category: 提示詞分類子目錄（例如 "langgraph"）
    :param name: 提示詞名稱（例如 "page_analyzer", "toc_indexer"）
    :return: 提示詞字串
    """
    base_path = _PROMPTS_DIR / category / name
    for candidate in [base_path.with_suffix(".txt"), base_path.with_suffix(".md"), base_path]:
        if candidate.is_file():
            try:
                content = candidate.read_text(encoding="utf-8").strip()
                return content
            except Exception as e:
                logger.error("讀取 Prompt 檔案失敗: %s, 錯誤: %s", candidate, e)
                raise

    err_msg = f"找不到指定的 Prompt 設定檔: category='{category}', name='{name}' (搜尋路徑: {_PROMPTS_DIR / category})"
    logger.error(err_msg)
    raise FileNotFoundError(err_msg)


def get_langgraph_prompt(name: str) -> str:
    """LangGraph 專用快捷 Prompt 讀取函式。"""
    return load_prompt("langgraph", name)
