"""Name comparison heuristics."""

from __future__ import annotations

from typing import Iterable, Sequence

from .nicknames import NICKNAME_MAP


def _nickname_equivalent(first: str, second: str) -> bool:
    first_lower = first.lower()
    second_lower = second.lower()
    in_first = first_lower in NICKNAME_MAP and second_lower in NICKNAME_MAP[first_lower]
    in_second = second_lower in NICKNAME_MAP and first_lower in NICKNAME_MAP[second_lower]
    return in_first or in_second


def is_plausible_expansion(name1: str, name2: str) -> bool:
    """Return True when `name1` could be an expansion of `name2` or vice versa."""

    parts1 = name1.split()
    parts2 = name2.split()

    if not parts1 or not parts2 or parts1[-1] != parts2[-1]:
        return False

    if len(parts1) > len(parts2):
        full_parts, abbr_parts = parts1, parts2
    else:
        full_parts, abbr_parts = parts2, parts1

    full_core = full_parts[:-1]
    abbr_core = abbr_parts[:-1]

    if not abbr_core:
        return True
    if not full_core:
        return False

    full_idx = 0
    abbr_idx = 0

    while abbr_idx < len(abbr_core) and full_idx < len(full_core):
        abbr_piece = abbr_core[abbr_idx]
        full_piece = full_core[full_idx]

        if abbr_piece == full_piece:
            abbr_idx += 1
            full_idx += 1
            continue

        if _nickname_equivalent(abbr_piece, full_piece):
            abbr_idx += 1
            full_idx += 1
            continue

        if len(abbr_piece) == 1 and full_piece.startswith(abbr_piece):
            abbr_idx += 1
            full_idx += 1
            continue

        if "-" in full_piece and full_piece.startswith(abbr_piece):
            abbr_idx += 1
            full_idx += 1
            continue

        if len(abbr_piece) > 1 and abbr_piece.isalpha():
            num_initials = len(abbr_piece)
            if full_idx + num_initials <= len(full_core):
                initials = "".join(core[0] for core in full_core[full_idx : full_idx + num_initials] if core)
                if abbr_piece == initials:
                    abbr_idx += 1
                    full_idx += num_initials
                    continue

        full_idx += 1

    return abbr_idx == len(abbr_core)


def is_hard_conflict(name1: str, name2: str) -> bool:
    """Return True when `name1` and `name2` cannot belong to the same person."""

    parts1 = name1.split()
    parts2 = name2.split()

    if not parts1 or not parts2 or parts1[-1] != parts2[-1]:
        return True

    first1, first2 = parts1[0], parts2[0]
    middles1 = parts1[1:-1]
    middles2 = parts2[1:-1]

    if len(first1) > 1 and len(first2) > 1 and first1 != first2:
        if first1.startswith(first2) or first2.startswith(first1):
            pass
        elif _nickname_equivalent(first1, first2):
            pass
        else:
            return True

    for m1, m2 in zip(middles1, middles2):
        if m1 == m2:
            continue
        if len(m1) > 1 and len(m2) > 1 and m1 != m2:
            return True
        if len(m1) == 1 and len(m2) == 1 and m1 != m2:
            return True
        if len(m1) == 1 and len(m2) > 1 and not m2.startswith(m1):
            return True
        if len(m2) == 1 and len(m1) > 1 and not m1.startswith(m2):
            return True

    return False


def clusters_conflict(
    left_indices: Iterable[int],
    right_indices: Iterable[int],
    normalized_names: Sequence[str],
) -> bool:
    """Return True when any name pair across two clusters is incompatible."""

    for li in left_indices:
        left_name = normalized_names[li]
        for ri in right_indices:
            right_name = normalized_names[ri]
            if is_hard_conflict(left_name, right_name):
                return True
    return False
