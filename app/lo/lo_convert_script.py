"""UNO 橋接轉檔腳本：由 LibreOffice Portable 內建的 python.exe 直接執行。

不屬於 FastAPI venv，禁止引入任何第三方套件。
只能使用 Python 標準程式庫和 LibreOffice 隨附的 uno / com.sun.star 模組。

用法：
    python.exe lo_convert_script.py <port> <input_abs_path> <output_abs_path>

Exit code：
    0  — 轉換成功
    1  — 連線失敗 / 轉換失敗（詳細訊息輸出至 stderr）
"""
from __future__ import annotations

import sys
from pathlib import Path


def _to_url(path: str) -> str:
    """將絕對路徑轉換為 LibreOffice 接受的 file:// URL 格式。

    Windows 路徑範例：
        C:\\foo\\bar.docx  →  file:///C:/foo/bar.docx
    """
    p = Path(path).resolve()
    # Path.as_uri() 產生正確的 file:// URL（跨平台）
    return p.as_uri()


def main() -> None:
    if len(sys.argv) != 4:
        print(
            "Usage: python.exe lo_convert_script.py <port> <input_path> <output_path>",
            file=sys.stderr,
        )
        sys.exit(1)

    port = int(sys.argv[1])
    input_path = sys.argv[2]
    output_path = sys.argv[3]

    try:
        import uno  # noqa: PLC0415  # type: ignore[import]
        from com.sun.star.beans import PropertyValue  # type: ignore[import]
        from com.sun.star.lang import DisposedException  # type: ignore[import]
    except ImportError as exc:
        print(f"[lo_convert_script] 無法匯入 UNO 模組：{exc}", file=sys.stderr)
        sys.exit(1)

    local_ctx = uno.getComponentContext()
    smgr = local_ctx.ServiceManager
    resolver = smgr.createInstanceWithContext(
        "com.sun.star.bridge.UnoUrlResolver", local_ctx
    )

    try:
        ctx = resolver.resolve(
            f"uno:socket,host=127.0.0.1,port={port};urp;StarOffice.ComponentContext"
        )
    except Exception as exc:
        print(
            f"[lo_convert_script] 無法連線至 LibreOffice daemon port={port}：{exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    smgr2 = ctx.ServiceManager
    desktop = smgr2.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)

    # 開啟文件（Hidden=True 不顯示視窗）
    def _make_prop(name: str, value: object) -> PropertyValue:
        pv = PropertyValue()
        pv.Name = name
        pv.Value = value
        return pv

    open_props = (_make_prop("Hidden", True), _make_prop("MacroExecutionMode", 4))
    input_url = _to_url(input_path)

    try:
        doc = desktop.loadComponentFromURL(input_url, "_blank", 0, open_props)
    except Exception as exc:
        print(
            f"[lo_convert_script] 無法開啟文件 {input_path}：{exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    if doc is None:
        print(
            f"[lo_convert_script] loadComponentFromURL 回傳 None，input={input_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    # 依文件類型自動選擇 PDF 匯出 filter
    def _detect_pdf_filter(d: object) -> str:
        """根據文件服務類型回傳對應的 PDF 匯出 FilterName。"""
        try:
            if d.supportsService(  # type: ignore[attr-defined]
                "com.sun.star.presentation.PresentationDocument"
            ):
                return "impress_pdf_Export"
            if d.supportsService(  # type: ignore[attr-defined]
                "com.sun.star.sheet.SpreadsheetDocument"
            ):
                return "calc_pdf_Export"
        except Exception:
            pass
        # Writer / Draw / ODT / RTF 等一律 fallback 至 writer_pdf_Export
        return "writer_pdf_Export"

    # 匯出為 PDF
    output_url = _to_url(output_path)
    filter_name = _detect_pdf_filter(doc)
    export_props = (_make_prop("FilterName", filter_name),)

    try:
        doc.storeToURL(output_url, export_props)
    except Exception as exc:
        print(
            f"[lo_convert_script] PDF 匯出失敗 {output_path}：{exc}",
            file=sys.stderr,
        )
        try:
            doc.close(True)
        except DisposedException:
            pass
        sys.exit(1)

    try:
        doc.close(True)
    except DisposedException:
        pass

    print(f"[lo_convert_script] 轉換成功：{output_path}")


if __name__ == "__main__":
    main()
