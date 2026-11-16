"""Small text helpers shared by search, reranking and evaluation."""
from __future__ import annotations

import re
import unicodedata

STOPWORDS = frozenset(
    """a an the and or but if then of to in on at by for with from as is are was were be been being do does did
    has have had this that these those it its which who whom what when where why how not no yes can could should
    would may might must shall will about into over under than so such there their they them he she his her we our
    you your i me my also any all each per via much many often long s""".split()
)

_WORD = re.compile(r"[a-z0-9]+")


def stem(word: str) -> str:
    """A deliberately tiny stemmer: plurals only."""
    if len(word) > 3 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def content_tokens(text: str) -> list[str]:
    """Lower-cased words without stopwords, plural-stemmed."""
    text = unicodedata.normalize("NFKC", text).lower()
    return [stem(w) for w in _WORD.findall(text) if w not in STOPWORDS]
