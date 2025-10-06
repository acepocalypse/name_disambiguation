"""Canonical name selection utilities."""

from __future__ import annotations

from typing import Iterable

from .normalization import normalize


def generate_clean_canonical(original_names: Iterable[str]) -> str:
    """Return the best canonical representation for `original_names` cluster."""

    originals = list(original_names)
    if not originals:
        return ""

    normalized = [normalize(name) for name in originals]

    def score(index: int) -> tuple[int, int, int, int, int, int]:
        name = normalized[index]
        parts = name.split()
        has_full_first = int(len(parts) > 0 and len(parts[0]) > 1)
        has_full_last = int(len(parts) > 1 and len(parts[-1]) > 1)
        has_full_middles = int(len(parts) > 2 and all(len(p) > 1 for p in parts[1:-1]))
        has_any_middles = int(len(parts) > 2)
        num_parts = len(parts)
        name_length = len(name)
        return (
            has_full_first,
            has_full_last,
            has_full_middles,
            has_any_middles,
            num_parts,
            name_length,
        )

    best_index = max(range(len(normalized)), key=score)
    best_normalized = normalized[best_index]
    return " ".join(piece.capitalize() for piece in best_normalized.split())
