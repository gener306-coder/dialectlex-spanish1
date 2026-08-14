"""
preprocessing.py — Text preprocessing pipeline for Spanish toxicity detection.

Applies minimal normalization that preserves dialectal markers,
following the approach described in the paper.
"""

import re
from typing import Optional


# Replacement tokens (compatible with BETO tokenizer)
URL_TOKEN      = "[URL]"
USER_TOKEN     = "@USER"
HASHTAG_TOKEN  = "[HASHTAG]"
NUMBER_TOKEN   = "[NUMBER]"
EMAIL_TOKEN    = "[EMAIL]"
EMOJI_TOKEN    = "[EMOJI]"
CURRENCY_TOKEN = "[CURRENCY]"

# Basic emoji range pattern
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002500-\U00002BEF"
    "\U00010000-\U0010FFFF"
    "]+",
    flags=re.UNICODE,
)

_URL_PATTERN      = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_USER_PATTERN     = re.compile(r"@\w+")
_HASHTAG_PATTERN  = re.compile(r"#\w+")
_EMAIL_PATTERN    = re.compile(r"\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b", re.IGNORECASE)
_NUMBER_PATTERN   = re.compile(r"\b\d+([.,]\d+)*\b")
_CURRENCY_PATTERN = re.compile(r"[$€£¥₹]\d+|\d+[$€£¥₹]")
_CTRL_PATTERN     = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_SPACE_PATTERN    = re.compile(r"\s+")


def preprocess(
    text: str,
    replace_urls: bool = True,
    replace_users: bool = True,
    replace_hashtags: bool = True,
    replace_numbers: bool = True,
    replace_emails: bool = True,
    replace_emojis: bool = True,
    replace_currency: bool = True,
    lowercase: bool = False,
) -> str:
    """
    Preprocess a Spanish text for toxicity detection.

    Intentionally preserves dialectal markers (slang, regional vocabulary,
    informal expressions) — only removes noise elements.

    NOTE: We do NOT apply lemmatization, stemming, or stopword removal
    because these operations would destroy the dialectal markers we want
    to preserve.

    Parameters
    ----------
    text : str
        Input text.
    replace_urls : bool
        Replace URLs with [URL] token (default: True).
    replace_users : bool
        Replace @mentions with @USER token (default: True).
    replace_hashtags : bool
        Replace #hashtags with [HASHTAG] token (default: True).
    replace_numbers : bool
        Replace numbers with [NUMBER] token (default: True).
    replace_emails : bool
        Replace email addresses with [EMAIL] token (default: True).
    replace_emojis : bool
        Replace emoji characters with [EMOJI] token (default: True).
    replace_currency : bool
        Replace currency symbols with [CURRENCY] token (default: True).
    lowercase : bool
        Convert text to lowercase (default: False — BETO handles casing).

    Returns
    -------
    str
        Preprocessed text.
    """
    if not isinstance(text, str):
        text = str(text)

    # Remove control characters
    text = _CTRL_PATTERN.sub(" ", text)

    # Replace elements in order (most specific first)
    if replace_emails:
        text = _EMAIL_PATTERN.sub(EMAIL_TOKEN, text)
    if replace_urls:
        text = _URL_PATTERN.sub(URL_TOKEN, text)
    if replace_users:
        text = _USER_PATTERN.sub(USER_TOKEN, text)
    if replace_hashtags:
        text = _HASHTAG_PATTERN.sub(HASHTAG_TOKEN, text)
    if replace_currency:
        text = _CURRENCY_PATTERN.sub(CURRENCY_TOKEN, text)
    if replace_numbers:
        text = _NUMBER_PATTERN.sub(NUMBER_TOKEN, text)
    if replace_emojis:
        text = _EMOJI_PATTERN.sub(EMOJI_TOKEN, text)

    # Normalize whitespace
    text = _SPACE_PATTERN.sub(" ", text).strip()

    if lowercase:
        text = text.lower()

    return text


def preprocess_batch(texts: list, **kwargs) -> list:
    """
    Preprocess a list of texts.

    Parameters
    ----------
    texts : list of str
        Input texts.
    **kwargs
        Keyword arguments passed to preprocess().

    Returns
    -------
    list of str
        Preprocessed texts.
    """
    return [preprocess(t, **kwargs) for t in texts]
