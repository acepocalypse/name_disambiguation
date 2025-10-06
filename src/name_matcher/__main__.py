"""Command line entry point for the Name Matcher library."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .llm import LLMConfig
from .pipeline import NameMatcherConfig
from .runner import match_file


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster person names into canonical entities.")
    parser.add_argument("input", type=Path, help="Path to the input CSV or Excel file")
    parser.add_argument("output", type=Path, help="Path where the annotated results will be written")
    parser.add_argument("--name-column", default="name", help="Column containing the raw names (default: name)")
    parser.add_argument(
        "--auto-no-prob",
        type=float,
        default=0.35,
        help="Cosine similarity threshold for automatic outlier ejection",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Disable progress bars even if tqdm is installed",
    )
    parser.add_argument("--llm-url", default=os.getenv("LLM_URL", ""), help="Endpoint for the LLM reviewer")
    parser.add_argument("--llm-model", default=os.getenv("LLM_MODEL", "qwen2.5:72b"), help="LLM model identifier")
    parser.add_argument(
        "--llm-token",
        default=os.getenv("LLM_TOKEN"),
        help="Authentication token for the LLM service",
    )
    parser.add_argument(
        "--llm",
        dest="llm_enabled",
        action="store_true",
        help="Enable the LLM adjudication step",
    )
    parser.add_argument(
        "--no-llm",
        dest="llm_enabled",
        action="store_false",
        help="Disable the LLM adjudication step",
    )
    parser.set_defaults(llm_enabled=False)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    llm_config = LLMConfig(
        enabled=args.llm_enabled,
        token=args.llm_token,
        url=args.llm_url,
        model=args.llm_model,
    )

    matcher_config = NameMatcherConfig(
        name_column=args.name_column,
        auto_no_prob=args.auto_no_prob,
        use_tqdm=not args.disable_tqdm,
        llm=llm_config,
    )

    match_file(args.input, args.output, matcher_config)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
