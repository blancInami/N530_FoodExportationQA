from markitdown import MarkItDown
from openai import OpenAI

LLM_BASE_URL = "http://10.166.57.22:40041/v1"
LLM_MODEL    = "gemma-4-26B-A4B-it-mtp"
_OCR_MODEL = "gemma-4-26B-A4B-it-mtp"
file_path = r'C:\Users\6747\Desktop\Projekt\2026\N530\N530_FoodExportationQA\tools\中英法規\1_食品安全衛生管理法\食品安全衛生管理法_EN.pdf'
client = OpenAI(
    base_url=LLM_BASE_URL,
    api_key="not-needed",  # 本地模型不驗證金鑰，填入佔位字串
)

md = MarkItDown(
    enable_plugins=True,
    llm_client=client,
    llm_model=_OCR_MODEL,
)
result = md.convert(file_path)

print(result.text_content)