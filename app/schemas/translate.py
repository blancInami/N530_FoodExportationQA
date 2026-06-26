"""
Pydantic schemas for translation endpoint.
"""
from pydantic import BaseModel, Field, ConfigDict


class TranslateRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"example": {"text": "農藥殘留限量標準"}}
    )
    text: str = Field(..., min_length=1, max_length=5000, description="要翻譯的中文文本，長度 1～5000 字元")


class TranslateResponse(BaseModel):
    original: str = Field(..., description="原始中文輸入文本")
    translation: str = Field(..., description="英文翻譯結果（官方正規術語已替換為核定英文名詞）")
