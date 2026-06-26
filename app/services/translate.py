"""
Translation service: uses LLM with mandatory dictionary for official terminology.
"""
import asyncio
import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.llm import chat_completion
from app.utils.dictionary import build_dictionary_xml, get_dictionary_matcher

logger = logging.getLogger(__name__)


async def translate_with_dictionary(
    text: str,
    session: AsyncSession,
    interrupt_event: asyncio.Event | None = None,
) -> str:
    """
    Translate Chinese text to English using the mandatory dictionary
    for official terminology replacement.
    """
    logger.info("翻譯文字：輸入長度=%d", len(text))
    matcher = await get_dictionary_matcher(session)
    filtered_dict = matcher.filter_by_text(text)
    dictionary_xml = build_dictionary_xml(filtered_dict)
    logger.info("字典篩選完成：命中 %d 筆術語", len(filtered_dict))

    prompt = f"""You are an official translation assistant for food export regulations.

{dictionary_xml}

<Translation_Task>
Translate the following Chinese text into English.
You MUST use the exact English terms from the <Mandatory_Dictionary> above
when translating any matching Chinese terms. Do not paraphrase or substitute
the official English terms.

Source text:
{text}
</Translation_Task>

<Output_Instructions>
Provide ONLY the English translation. Do not include any explanation or the original Chinese text.
</Output_Instructions>"""

    result = await chat_completion(prompt, temperature=0, max_tokens=2048, interrupt_event=interrupt_event)
    result = result.strip()
    logger.info("翻譯完成：輸出長度=%d", len(result))
    return result