"""Name Matcher library initialization."""

from .pipeline import NameMatcher, NameMatcherConfig, NameMatcherResult, NameMatcherStats
from .llm import LLMConfig
from .canonical import generate_clean_canonical
from .normalization import normalize, get_block_key
from .runner import match_file

__all__ = [
    "NameMatcher",
    "NameMatcherConfig",
    "NameMatcherResult",
    "NameMatcherStats",
    "LLMConfig",
    "generate_clean_canonical",
    "normalize",
    "get_block_key",
    "match_file",
]
