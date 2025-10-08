"""LLM integration helpers."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import ftfy
import requests


_DEFAULT_PROMPT = """Determine if each name pair refers to the same person. Return JSON only.

RULES:
1. Different last names → NO match (e.g., Miller ≠ Durand)
2. Conflicting initials → NO match (e.g., John B. Smith ≠ John G. Smith)
3. Nicknames → YES match (e.g., Chris = Christopher)
4. Initials = full names → YES match (e.g., J. Smith = John Smith)
5. Missing middle → YES match (e.g., John Smith = John B. Smith)

EXAMPLES:
YES: Chris Beard = Christopher Beard
YES: Carl R. de Boor = Carl-Wilhelm Reinhold de Boor
YES: C. N. R. Rao = Chintamani Nagesa Ramachandra Rao
NO: John B. Smith ≠ John G. Smith
NO: Miller ≠ Durand

OUTPUT FORMAT:
```json
{"results": [{"index": 0, "match": "yes"}, {"index": 1, "match": "no"}]}
```"""


@dataclass
class LLMConfig:
    """Configuration for the optional LLM disambiguation step."""

    enabled: bool = True
    token: str | None = None
    url: str = "https://genai.rcac.purdue.edu/api/chat/completions"
    model: str = "gpt-oss:latest"
    batch_size: int = 25  # Reduced from 25 to prevent timeouts
    max_retries: int = 3  # Increased from 2 for better reliability
    concurrent_requests: int = 3
    timeout_seconds: int = 180  # Increased from 120 to handle slow responses
    connection_timeout: int = 30  # Separate connection timeout
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

    # Fix any encoding issues in the input names
    formatted_pairs = [
        {"index": i, "name1": ftfy.fix_text(first), "name2": ftfy.fix_text(second)}
        for i, (first, second) in enumerate(pairs_to_check)
    ]

    body = {
        "model": config.model,
        "messages": [
            {
                "role": "user",
                "content": f"{config.prompt}\n\nPairs:\n{json.dumps(formatted_pairs)}",
            }
        ],
        "stream": False,
        "reasoning_effort": 'low',
        "temperature": 0,  # Deterministic output
        "max_tokens": 500 + len(formatted_pairs) * 20,  # Dynamic limit based on batch size
    }

    decisions = [False] * len(pairs_to_check)
    for attempt in range(1, config.max_retries + 1):
        try:
            # Use tuple for timeout: (connection_timeout, read_timeout)
            # This prevents connection delays from consuming read timeout
            timeout_tuple = (config.connection_timeout, config.timeout_seconds)
            response = requests.post(
                config.url,
                headers=config.headers(),
                json=body,
                timeout=timeout_tuple,
            )
            response.raise_for_status()
            payload = response.json()["choices"][0]["message"]["content"]
            _update_decisions_from_payload(payload, decisions)
            return decisions
        except requests.exceptions.Timeout as exc:
            # Specific handling for timeout errors
            backoff = 2 ** attempt  # Exponential backoff: 2s, 4s, 8s
            if attempt <= config.max_retries:
                print(f"   LLM Timeout on attempt {attempt}/{config.max_retries}: {exc}. Retrying in {backoff}s...")
                time.sleep(backoff)
            else:
                print(f"   LLM Timeout: All {config.max_retries} attempts failed. Batch defaulted to 'no'.")
        except requests.exceptions.HTTPError as exc:
            # HTTP errors (4xx, 5xx)
            print(f"   LLM HTTP Error on attempt {attempt}/{config.max_retries}: {exc.response.status_code} - {exc}")
            if exc.response.status_code >= 500 and attempt <= config.max_retries:
                # Retry on server errors
                backoff = 2 ** attempt
                time.sleep(backoff)
            else:
                # Don't retry on client errors (4xx)
                break
        except Exception as exc:
            # Other errors (connection, parsing, etc.)
            backoff = 2 ** attempt
            if attempt <= config.max_retries:
                print(f"   LLM Error on attempt {attempt}/{config.max_retries}: {type(exc).__name__}: {exc}. Retrying in {backoff}s...")
                time.sleep(backoff)
            else:
                print(f"   LLM Error: All {config.max_retries} attempts failed. Batch defaulted to 'no'.")
    return decisions


def _update_decisions_from_payload(payload: str, decisions: List[bool]) -> None:
    # Fix any text encoding issues in the payload
    fixed_payload = ftfy.fix_text(payload)
    
    match = re.search(r"```json\s*(\{.*?\})\s*```", fixed_payload, re.DOTALL)
    if not match:
        match = re.search(r"(\{.*?\})", fixed_payload, re.DOTALL)
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
