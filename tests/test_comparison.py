from name_matcher.comparison import is_hard_conflict, is_plausible_expansion


def test_plausible_expansion_accepts_multi_initials():
    assert is_plausible_expansion(
        "samuel f b morse",
        "samuel finley breese morse",
    )


def test_plausible_expansion_rejects_different_last_names():
    assert not is_plausible_expansion("john smith", "john jones")


def test_hard_conflict_detects_conflicting_initials():
    assert is_hard_conflict("john b smith", "john g smith")


def test_hard_conflict_allows_nicknames():
    assert not is_hard_conflict("chuck smith", "charles smith")
