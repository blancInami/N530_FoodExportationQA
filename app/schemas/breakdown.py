from pydantic import AliasChoices, BaseModel, Field, ConfigDict


class BreakdownQuestion(BaseModel):
    """單筆問卷題目萃取結果。"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question_id": "1.2.3",
                "question_text": "<p>Please provide details of approved slaughterhouses in the table below:</p><table border=\"1\"><thead><tr><th>Name</th><th>Capacity</th></tr></thead><tbody><tr><td>ABC Abattoir</td><td>500</td></tr></tbody></table>",
            }
        }
    )
    question_id: str = Field(
        ...,
        description="扁平化題號，例如 `1`、`1.1`、`2.2.1`、`II.1.2.a`"
    )
    question_text: str = Field(
        ...,
        description="題目內容（包含問句本體，隨附表格或選項已轉換為標準語意 HTML 格式）"
    )


# 保留舊名稱作為相容性別名
BreakdownItem = BreakdownQuestion


class BreakdownSectionDetail(BaseModel):
    """區塊/Part 詳情，包含前言說明與題目列表。"""
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "depiction": "This part applies to live, chilled and frozen bivalve molluscs produced for export to the EU.",
                "question": [
                    {
                        "question_id": "A.1",
                        "question_text": "controls are in place to classify and monitor growing areas for bivalve molluscs"
                    }
                ]
            }
        }
    )
    depiction: str = Field(
        default="",
        description="該區塊/Part 之說明文字、前言或法規範圍（無則為空字串）"
    )
    question: list[BreakdownQuestion] = Field(
        default_factory=list,
        validation_alias=AliasChoices("question", "questions"),
        description="該區塊/Part 下所屬的題目列表"
    )

    @property
    def questions(self) -> list[BreakdownQuestion]:
        return self.question
