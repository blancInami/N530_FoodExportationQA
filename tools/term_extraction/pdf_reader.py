"""
模組一（Part B）：PDF 文件讀取

功能：
    透過遠端 doc-to-json API 將 PDF 轉換為結構化 JSON，
    再從中擷取純文字內容。中英文 PDF 使用相同讀取流程。

API 端點：
    POST http://<host>:<port>/api/doc-to-json/run
    - 上傳 PDF 檔案（multipart/form-data）
    - 回傳 DocLayoutFieldList：逐頁的區塊辨識結果

輸出格式（DocLayoutFieldList 每頁的元素）：
    {
        "content": "文字內容",
        "classification": "plain text" | "title" | "table" | "table_footnote" | ...,
        "confidence": 0.95,
        "bbox": {...},
        ...
    }

僅保留 classification 為以下類型的元素：
    plain text, title, table, table_footnote
"""

import io
import logging
from pathlib import Path

import requests
from PyPDF2 import PdfReader, PdfWriter

logger = logging.getLogger(__name__)

# ── 預設 API 端點 ─────────────────────────────────────────────────────────────

DEFAULT_API_URL = "http://10.166.57.22:43002/api/doc-to-json/run"

# 要保留的文件區塊分類類型
_VALID_CLASSIFICATIONS = {"plain text", "title", "table", "table_footnote"}


# ── 主要函式 ──────────────────────────────────────────────────────────────────

def read_pdf(file_path: Path, api_url: str = DEFAULT_API_URL, pages=None) -> str:
    """
    透過 doc-to-json API 讀取 PDF 並擷取純文字內容。
    中英文 PDF 使用相同流程。

    Args:
        file_path: PDF 檔案的 Path 物件
        api_url  : doc-to-json API 的完整 URL
        pages    : 頁面選擇參數（預設 None = 全部頁面）
                   - None / "all"：全部頁面
                   - int          ：指定單頁（如 1）
                   - tuple(int,int)：頁面範圍（如 (1, 5)）
                   - list[int]    ：指定多頁（如 [1, 3, 5]）

    Returns:
        從 PDF 擷取的純文字字串（各區塊以換行連接）。
        若讀取失敗回傳空字串。
    """
    logger.info("讀取 PDF：%s", file_path.name)

    try:
        page_info = _process_page_parameter(pages)
        pdf_data, upload_filename = _process_pdf_pages(file_path, page_info)
        if pdf_data is None:
            logger.error("PDF 頁面處理失敗：%s", file_path.name)
            return ""

        pdfid = _generate_pdfid(file_path.stem, page_info)
        raw_result = _upload_to_api(pdf_data, upload_filename, pdfid, api_url)
        if raw_result is None:
            return ""

        text = _extract_text_from_result(raw_result)
        logger.info("PDF 讀取完成：%s  字元數=%d", file_path.name, len(text))
        return text

    except Exception as exc:
        logger.error("PDF 讀取失敗，回傳空字串：%s  錯誤：%s", file_path.name, exc)
        return ""


def read_pdf_structured(
    file_path: Path,
    api_url: str = DEFAULT_API_URL,
    pages=None,
) -> list[dict]:
    """
    透過 doc-to-json API 讀取 PDF，回傳結構化區塊列表（保留版面語意）。

    每個區塊格式：
        {
            "type" : "plain text" | "title" | "table" | "table_footnote",
            "text" : str,
            "page" : int,   # 1-based 頁碼
            "order": int,   # 頁內元素順序
        }

    Args 與 read_pdf() 相同。

    Returns:
        結構化區塊列表；失敗時回傳空列表。
    """
    logger.info("結構化讀取 PDF：%s", file_path.name)

    try:
        page_info = _process_page_parameter(pages)
        pdf_data, upload_filename = _process_pdf_pages(file_path, page_info)
        if pdf_data is None:
            logger.error("PDF 頁面處理失敗：%s", file_path.name)
            return []

        pdfid = _generate_pdfid(file_path.stem, page_info)
        raw_result = _upload_to_api(pdf_data, upload_filename, pdfid, api_url)
        if raw_result is None:
            return []

        blocks = _extract_blocks_from_result(raw_result)
        logger.info(
            "PDF 結構化讀取完成：%s  區塊數=%d  (title=%d, plain=%d, table=%d)",
            file_path.name, len(blocks),
            sum(1 for b in blocks if b["type"] == "title"),
            sum(1 for b in blocks if b["type"] == "plain text"),
            sum(1 for b in blocks if b["type"] in ("table", "table_footnote")),
        )
        return blocks

    except Exception as exc:
        logger.error("PDF 結構化讀取失敗，回傳空列表：%s  錯誤：%s", file_path.name, exc)
        return []


# ── API 上傳函式 ──────────────────────────────────────────────────────────────

def _upload_to_api(
    pdf_data: bytes,
    filename: str,
    pdfid: str,
    api_url: str,
) -> dict | None:
    """
    將 PDF 資料上傳至 doc-to-json API。

    Args:
        pdf_data : PDF 的原始位元組資料
        filename : 上傳時使用的檔案名稱
        pdfid    : 文件識別碼（供 API 快取或辨識用）
        api_url  : API 端點 URL

    Returns:
        API 回傳的 JSON dict，或 None（上傳失敗時）。
    """
    logger.info("上傳 PDF 至 API：%s  PDFID=%s  URL=%s", filename, pdfid, api_url)

    files = {"file": (filename, pdf_data, "application/pdf")}
    data = {"PDFID": pdfid, "enable_doc_layout_analysis": True}

    try:
        response = requests.post(api_url, files=files, data=data, timeout=600)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        logger.error("API 請求逾時（600 秒）：%s", api_url)
        return None
    except requests.exceptions.ConnectionError as exc:
        logger.error("無法連線 API：%s  錯誤：%s", api_url, exc)
        return None
    except requests.exceptions.HTTPError as exc:
        logger.error("API 回傳 HTTP 錯誤：狀態碼=%s  URL=%s", exc.response.status_code, api_url)
        return None

    try:
        return response.json()
    except ValueError:
        logger.error("API 回傳非 JSON 格式：%s", response.text[:200])
        return None


# ── 文字擷取函式 ──────────────────────────────────────────────────────────────

def _extract_blocks_from_result(result: dict) -> list[dict]:
    """
    從 doc-to-json API 回應中擷取結構化區塊列表。

    每個區塊格式：
        {
            "type" : "plain text" | "title" | "table" | "table_footnote",
            "text" : str,
            "page" : int,   # 1-based 頁碼
            "order": int,   # 頁內順序
        }

    Args:
        result: API 回傳的原始 JSON dict

    Returns:
        結構化區塊列表；API 無資料時回傳空列表。
    """
    doc_layout = result.get("DocLayoutFieldList", [])
    if not doc_layout:
        logger.warning("API 回應中無 DocLayoutFieldList 欄位")
        return []

    blocks: list[dict] = []

    for page_idx, page_fields in enumerate(doc_layout):
        if isinstance(page_fields, dict) and "elements" in page_fields:
            elements = page_fields["elements"]
        elif isinstance(page_fields, list):
            elements = page_fields
        else:
            logger.debug("第 %d 頁格式不符預期，跳過", page_idx + 1)
            continue

        for order, element in enumerate(elements):
            cls = element.get("classification", "")
            if cls in _VALID_CLASSIFICATIONS:
                content = element.get("content", "").strip()
                if content:
                    blocks.append({
                        "type" : cls,
                        "text" : content,
                        "page" : page_idx + 1,
                        "order": order,
                    })

    return blocks


def _extract_text_from_result(result: dict) -> str:
    """
    從 doc-to-json API 回應中擷取純文字內容。

    處理流程：
    1. 取得 DocLayoutFieldList（逐頁的版面辨識結果）
    2. 對每一頁，過濾出 classification 為指定類型的元素
    3. 擷取各元素的 content 欄位
    4. 以雙換行符號連接所有文字區塊

    Args:
        result: API 回傳的原始 JSON dict

    Returns:
        合併後的純文字字串
    """
    blocks = _extract_blocks_from_result(result)
    return "\n\n".join(b["text"] for b in blocks)


# ── 頁面處理函式 ──────────────────────────────────────────────────────────────

def _process_page_parameter(pages) -> dict:
    """
    將頁面選擇參數轉換為統一的處理資訊結構。

    Args:
        pages: 頁面選擇參數（None / "all" / int / tuple / list）

    Returns:
        dict: 含 'type', 'description', 'pages', 'suffix' 欄位
    """
    if pages is None or pages == "all":
        return {"type": "all", "description": "全部頁面", "pages": None, "suffix": ""}
    elif isinstance(pages, int):
        if pages <= 0:
            raise ValueError("頁碼必須大於 0")
        return {
            "type": "single",
            "description": f"第 {pages} 頁",
            "pages": [pages],
            "suffix": f"_page{pages}",
        }
    elif isinstance(pages, tuple) and len(pages) == 2:
        start, end = pages
        if start <= 0 or end <= 0 or start > end:
            raise ValueError("頁碼範圍必須為正數且起始頁不能大於結束頁")
        return {
            "type": "range",
            "description": f"第 {start} 到 {end} 頁",
            "pages": list(range(start, end + 1)),
            "suffix": f"_pages{start}-{end}",
        }
    elif isinstance(pages, (list, tuple)):
        if not all(isinstance(p, int) and p > 0 for p in pages):
            raise ValueError("所有頁碼必須為正整數")
        page_list = sorted(set(pages))
        return {
            "type": "list",
            "description": f"第 {', '.join(map(str, page_list))} 頁",
            "pages": page_list,
            "suffix": f"_pages{'_'.join(map(str, page_list))}",
        }
    else:
        raise ValueError(f"不支援的頁面參數格式: {pages}")


def _generate_pdfid(filename: str, page_info: dict) -> str:
    """根據檔案名稱和頁面資訊生成 PDFID。"""
    if page_info["type"] == "all":
        return filename
    return filename + page_info["suffix"]


def _process_pdf_pages(
    pdf_file_path: Path,
    page_info: dict,
) -> tuple[bytes | None, str]:
    """
    根據頁面資訊處理 PDF 文件，回傳（可能裁切後的）PDF 位元組資料。

    Args:
        pdf_file_path: PDF 檔案路徑
        page_info    : _process_page_parameter() 的回傳結果

    Returns:
        (PDF 位元組資料, 上傳用檔名)，失敗時回傳 (None, "")
    """
    try:
        filename = pdf_file_path.name

        # 全部頁面：直接讀取原始檔案
        if page_info["type"] == "all":
            with open(pdf_file_path, "rb") as f:
                return f.read(), filename

        # 指定頁面：使用 PyPDF2 裁切後輸出至記憶體
        with open(pdf_file_path, "rb") as f:
            reader = PdfReader(f)
            writer = PdfWriter()
            total_pages = len(reader.pages)

            for page_num in page_info["pages"]:
                if page_num > total_pages:
                    raise ValueError(f"頁面 {page_num} 超出 PDF 總頁數 {total_pages}")
                writer.add_page(reader.pages[page_num - 1])  # 轉為 0-based 索引

            output_buffer = io.BytesIO()
            writer.write(output_buffer)

            new_filename = f"{pdf_file_path.stem}{page_info['suffix']}.pdf"
            return output_buffer.getvalue(), new_filename

    except Exception as exc:
        logger.error("PDF 頁面處理錯誤：%s  錯誤：%s", pdf_file_path.name, exc)
        return None, ""
