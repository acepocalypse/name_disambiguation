"""Name normalization helpers."""

from __future__ import annotations

import re
from typing import Tuple

import ftfy
from unidecode import unidecode


_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
_TITLES_PATTERN = re.compile(r"^(dr|prof|mr|mrs|ms|hon)\.?\s+", re.IGNORECASE)
_SUFFIX_PATTERN = re.compile(r",?\s+(jr|sr|i{1,3}|iv|v)\.?$", re.IGNORECASE)
_NON_WORD_PATTERN = re.compile(r"[^\w\s-]")
_MULTI_SPACE_PATTERN = re.compile(r"\s+")


def normalize(name: str) -> str:
    """Return a normalized representation of `name` for matching."""

    raw = str(name or "").strip()
    if not raw:
        return ""

    fixed = ftfy.fix_text(raw)
    if "," in fixed:
        head, tail = fixed.split(",", 1)
        after_comma = tail.strip().lower().replace(".", "")
        if after_comma and after_comma not in _SUFFIXES:
            last_name = head.strip()
            first_middle = tail.strip()
            fixed = f"{first_middle} {last_name}"

    ascii_friendly = unidecode(fixed).lower()
    ascii_friendly = _TITLES_PATTERN.sub("", ascii_friendly)
    ascii_friendly = _SUFFIX_PATTERN.sub("", ascii_friendly)
    ascii_friendly = ascii_friendly.replace(".", " ")
    ascii_friendly = _NON_WORD_PATTERN.sub("", ascii_friendly)
    collapsed = _MULTI_SPACE_PATTERN.sub(" ", ascii_friendly)
    return collapsed.strip()


def get_block_key(name: str) -> Tuple[str, str]:
    """Return a blocking key to limit candidate comparisons for `name`."""

    parts = [p for p in name.split() if p]
    if len(parts) > 1:
        return parts[0][0], parts[-1]
    if parts:
        return parts[0][0], parts[0]
    return "", ""
