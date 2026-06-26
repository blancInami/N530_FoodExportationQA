"""
OpenCC wrapper: simplified Chinese → Traditional Chinese (s2t).
Uses opencc-python-reimplemented (pure Python, no C compilation needed).
"""
from opencc import OpenCC

_converter = OpenCC("s2t")


def s2t(text: str) -> str:
    """Convert simplified Chinese to traditional Chinese."""
    if not text:
        return text
    return _converter.convert(text)
