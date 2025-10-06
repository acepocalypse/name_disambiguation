"""LLM integration helpers."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import requests


_DEFAULT_PROMPT = (
    "You are a precise data deduplication expert. Your task is to determine if the name pairs below refer to the same person. "
    "Your judgment is the final step for ambiguous cases.\n\n"
    "**CRITICAL RULES (NON-NEGOTIABLE):**\n"
    "1. **DIFFERENT LAST NAMES:** If the last names are clearly different (e.g., 'Miller' vs. 'Durand'), it is **NEVER** a match. This includes hyphenated names like 'Smith-Jones' vs 'Smith'.\n"
    "2. **CONFLICTING INITIALS:** If the names share a first and last name but have explicitly contradictory middle names or initials (e.g., 'John B. Smith' vs. 'John G. Smith'), it is **NEVER** a match.\n"
    "3. **NICKNAME HANDLING:** Common nicknames (e.g., 'Chris' for 'Christopher') ARE matches\n"
    "4. **HYPHENATED NAMES:** Hyphenated first names match their non-hyphenated variants when initials match (e.g., 'Carl-Wilhelm' = 'Carl')\n\n"
    "**GUIDING PRINCIPLE:** Match if one name is a plausible expansion, abbreviation, or version of the other. An initial is a plausible version of a full name (e.g., 'J. Smith' for 'John Smith').\n\n"
    "**MATCH THESE PATTERNS:**\n"
    "- `{'Chris Beard', 'Christopher Beard'}` (Nickname match)\n"
    "- `{'Carl R. de Boor', 'Carl-Wilhelm Reinhold de Boor'}` (Hyphenated expansion)\n"
    "- `{'Samuel F.B. Morse', 'Samuel Finley Breese Morse'}` (Multi-initial expansion)\n"
    "- `{'C. N. R. Rao', 'Chintamani Nagesa Ramachandra Rao'}` (Complex Initials for full names)\n"
    "- `{'J. B. Smith', 'John Barrett Smith'}` (Initials for full names)\n"
    "- `{'Stephen Bechtel', 'Stephen Davison Bechtel'}` (Missing middle name)\n\n"
    "**DO NOT MATCH (Based on Critical Rules):**\n"
    "- `{'John B. Smith', 'John G. Smith'}` (Conflicting initials)\n"
    "- `{'John Barrett Smith', 'John Garrett Smith'}` (Conflicting full names)\n\n"
    "Return a single JSON object in a ```json code block. The object must have one key, 'results', an array of objects. "
    "Each object needs an 'index' (integer) and a 'match' key ('yes' or 'no'). Do not add comments."
)


@dataclass
class LLMConfig:
    """Configuration for the optional LLM disambiguation step."""

    enabled: bool = True
    token: str | None = None
    url: str = "https://genai.rcac.purdue.edu/api/chat/completions"
    model: str = "gpt-oss:latest"
    batch_size: int = 25
    max_retries: int = 2
    concurrent_requests: int = 3
    timeout_seconds: int = 120
    prompt: str = _DEFAULT_PROMPT

    def __post_init__(self) -> None:
        if self.token is None:
            self.token = os.getenv("LLM_TOKEN")
        if not self.url:
            self.url = os.getenv("LLM_URL", "https://genai.rcac.purdue.edu/api/chat/completions")
        if not self.model:
            self.model = os.getenv("LLM_MODEL", "gpt-oss:latest")

    def headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers


class LLMClient:
    """Send ambiguous pairs to an LLM for final adjudication."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()

    def ask_batch(self, pairs: Sequence[Tuple[str, str]]) -> List[bool]:
        if not pairs:
            return []
        if not self.config.enabled:
            return [False] * len(pairs)
        return _ask_llm_batch_match(pairs, self.config)

    def review_pairs(self, pairs: Sequence[Tuple[str, str]]) -> List[bool]:
        """Compatibility shim for previous API."""

        return self.ask_batch(pairs)


def _ask_llm_batch_match(pairs_to_check: Sequence[Tuple[str, str]], config: LLMConfig) -> List[bool]:
    if not pairs_to_check:
        return []

    formatted_pairs = [
        {"index": i, "name1": first, "name2": second}
        for i, (first, second) in enumerate(pairs_to_check)
    ]

    body = {
        "model": config.model,
        "messages": [
            {
                "role": "user",
                "content": f"{config.prompt}\n\nPAIRS TO ANALYZE:\n{json.dumps(formatted_pairs, indent=2)}",
            }
        ],
        "stream": False,
        "reasoning_effort": "low",
    }

    decisions = [False] * len(pairs_to_check)
    for attempt in range(1, config.max_retries + 1):
        try:
            response = requests.post(
                config.url,
                headers=config.headers(),
                json=body,
                timeout=config.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()["choices"][0]["message"]["content"]
            _update_decisions_from_payload(payload, decisions)
            return decisions
        except Exception as exc:
            print(f"   LLM Error on attempt {attempt}: {exc}. Retrying...")
            time.sleep(2)
    print("   LLM Error: Batch failed after all retries. Defaulting to 'no' for this batch.")
    return decisions


def _update_decisions_from_payload(payload: str, decisions: List[bool]) -> None:
    match = re.search(r"```json\s*(\{.*?\})\s*```", payload, re.DOTALL)
    if not match:
        match = re.search(r"(\{.*?\})", payload, re.DOTALL)
    if not match:
        return

    try:
        results_json = json.loads(re.sub(r"//.*", "", match.group(1)))
    except json.JSONDecodeError:
        return

    for item in results_json.get("results", []):
        index = item.get("index")
        match_flag = item.get("match")
        if isinstance(index, int) and 0 <= index < len(decisions):
            decisions[index] = isinstance(match_flag, str) and match_flag.lower() == "yes"
