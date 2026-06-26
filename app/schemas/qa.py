"""
Pydantic schemas for QA endpoints.
"""
from pydantic import BaseModel, Field, ConfigDict


class ReferenceSource(BaseModel):
    """單筆參考來源，對應一筆命中切塊的問卷主檔 metadata。"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "問卷主檔主鍵": "Q20240001",
                "問卷名稱": "110年農產品輸日檢疫問卷",
                "輸出國家": "日本",
                "輸出品項": "鳳梨",
                "題目序號": "3.2",
            }
        }
    )
    問卷主檔主鍵: str = Field(..., description="問卷主檔主鍵")
    問卷名稱: str = Field(default="", description="問卷名稱")
    輸出國家: str = Field(default="", description="輸出國家（目的地國家）")
    輸出品項: str = Field(default="", description="輸出品項（農產品名稱）")
    題目序號: str = Field(default="", description="問卷中關聯題目序號")


class AskRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "What are the pesticide residue requirements for exporting pineapples to Japan?",
                "similarity_threshold": 0.75,
                "top_n": 5,
            }
        }
    )
    question: str = Field(
        ..., min_length=1, max_length=2000,
        description="使用者提問內容，支援中英文，長度 1～2000 字元"
    )
    similarity_threshold: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="向量搜尋距離閾值（0.0～1.0）。值越小要求越相似，未帶入則使用系統環境變數 SIMILARITY_THRESHOLD（預設 0.75）"
    )
    top_n: int | None = Field(
        default=None, ge=1, le=100,
        description="最終聚合取回筆數（1～100），未帶入則使用系統環境變數 TOP_N（預設 5）"
    )    
    intent_top_n: int | None = Field(
        default=None, ge=1, le=10,
        description="意圖分類回傳的負責單位最高數量（1～10），未帶入則使用系統預設值 INTENT_TOP_N"
    )

class KnowledgeReferenceSource(BaseModel):
    """單筆知識來源，對應一筆命中節點的知識文獻主檔 metadata。"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "文獻主檔主鍵": "D20240001",
                "文獻名稱": "農藥殘留標準法規彙編",
                "文獻類型": "REGULATION",
                "節點標題路徑": "第三章 > 第十二條 > 鳳梨農藥殘留限量",
            }
        }
    )
    文獻主檔主鍵: str = Field(..., description="知識文獻主檔主鍵")
    文獻名稱: str = Field(default="", description="文獻名稱")
    文獻類型: str = Field(
        default="",
        description="文獻類型：`REGULATION`（法規）、`GUIDELINE`（作業指引）、`QA`（問答集）"
    )
    節點標題路徑: str = Field(default="", description="節點的完整階層標題路徑，以 `>` 分隔各層級")


class AskResponse(BaseModel):
    """
    QA response payload — 雙語問答結果，包含負責單位、中英回覆與雙軌參考來源。
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "負責單位": [{"機關": "行政院農業委員會", "單位": "動植物防疫檢疫局", "理由": "農藥殘留"}],
                "原始題目": "What are the pesticide residue requirements for exporting pineapples to Japan?",
                "中文譯文": "出口鳳梨到日本的農藥殘留要求為何？",
                "英文回覆": "According to the regulations, the maximum residue limit (MRL) for chlorpyrifos in pineapples exported to Japan is 0.01 ppm.",
                "中文回覆": "根據相關法規，出口至日本的鳳梨中，克氯醯的最大農藥殘留限量（MRL）為 0.01 ppm。",
                "附件超連結": ["/files/questionnaire_2024001.pdf"],
                "參考來源": [
                    {"問卷主檔主鍵": "Q20240001", "問卷名稱": "110年農產品輸日檢疫問卷", "輸出國家": "日本", "輸出品項": "鳳梨", "題目序號": "3.2"}
                ],
                "知識參考來源": [
                    {"文獻主檔主鍵": "D20240001", "文獻名稱": "農藥殘留標準法規彙編", "文獻類型": "REGULATION", "節點標題路徑": "第三章 > 第十二條 > 鳳梨農藥殘留限量"}
                ],
            }
        }
    )
    負責單位: list[dict] = Field(
        default_factory=list,
        description="由意圖分類判斷之負責機關與單位清單，每個物件包含 `機關`、`單位` 及 `理由`（Layer 1 為命中關鍵字；Layer 2 為 LLM 推導依據）"
    )
    原始題目: str = Field(..., description="使用者的原始輸入提問（原文，未經翻譯）")
    中文譯文: str = Field(default="", description="若原始題目為外文，提供 LLM 翻譯後的繁體中文；若原始輸入已為中文則此欄位為空")
    中文回覆: str = Field(default="", description="LLM 產出的正體中文回覆")
    英文回覆: str = Field(default="", description="LLM 產出的英文回覆")
    附件超連結: list[str] = Field(default_factory=list, description="命中問卷的附件檔案路徑清單")
    參考來源: list[ReferenceSource] = Field(default_factory=list, description="Track A 檢索命中的問卷主檔資訊清單")
    知識參考來源: list[KnowledgeReferenceSource] = Field(
        default_factory=list, description="Track B 檢索命中的知識文獻來源（法規 / 作業指引 / 問答集）"
    )


class IngestItemResult(BaseModel):
    問卷主檔主鍵: str = Field(..., description="問卷主檔主鍵")
    status: str = Field(..., description="處理結果狀態：`success`、`not_found`、`error`")
    items_processed: int = Field(default=0, description="本筆問卷已處理的題目數量")
    total_chunks_written: int = Field(default=0, description="本筆問卷寫入的切塊總數")
    message: str = Field(default="", description="補充說明（失敗時顯示錯誤訊息）")


class IngestRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "問卷主檔主鍵清單": ["Q20240001", "Q20240002"]
            }
        }
    )
    問卷主檔主鍵清單: list[str] = Field(
        ..., min_length=1,
        description="問卷主檔主鍵清單，可一次傳入多筆，批次觸發切塊向量化"
    )


class IngestResponse(BaseModel):
    total_questionnaires: int = Field(default=0, description="本次請求處理的問卷總數")
    total_items_processed: int = Field(default=0, description="成功處理的題目總數")
    total_chunks_written: int = Field(default=0, description="寫入資料庫的切塊總數")
    results: list[IngestItemResult] = Field(default_factory=list)
