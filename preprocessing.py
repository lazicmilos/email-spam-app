"""Text preprocessing shared by training and inference.

This module is the single source of truth for cleaning text. The fitted
TfidfVectorizer stores a reference to `preprocess_text`, so the saved
pipeline always applies exactly this function at prediction time.
"""
import re

_DISALLOWED_CHARS = re.compile(r'[^a-z0-9\s£$€!?]')
_WHITESPACE = re.compile(r'\s+')


def preprocess_text(text):
    """Lowercase, keep letters, digits and spam-signal symbols (£$€!?), collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _DISALLOWED_CHARS.sub(' ', text)
    return _WHITESPACE.sub(' ', text).strip()
