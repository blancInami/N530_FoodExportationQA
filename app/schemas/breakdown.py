"""
Pydantic schemas for the /qa/breakdown endpoint.
"""
from pydantic import BaseModel, Field, ConfigDict


class BreakdownItem(BaseModel):
    """單筆問卷題目萃取結果。"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question_id": "1.2.3",
                "question_text": "What is the maximum residue limit for chlorpyrifos in pineapples?",
            }
        }
    )
    question_id: str = Field(
        ...,
        description="點分十進位格式題號，例如 `1`、`1.1`、`2.2.1`、`1.2.4.a`"
    )
    question_text: str = Field(
        ...,
        description="純英文題目內容（不含題號）"
    )
