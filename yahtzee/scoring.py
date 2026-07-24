"""Category definitions and scoring, including the Yahtzee joker / +100 rules.

Scoring here follows standard (Hasbro) Yahtzee scoring:

    3-of-a-kind / 4-of-a-kind / Chance ...... sum of all five dice
    Full house .............................. 25
    Small straight (4 in a row) ............. 30
    Large straight (5 in a row) ............. 40
    Yahtzee (5 of a kind) ................... 50
    Upper section bonus ..................... +35 when the upper total reaches 63

Extra-Yahtzee rule (the "joker"), as requested:
  * Each additional Yahtzee rolled after the Yahtzee box was filled with 50
    scores a +100 bonus.
  * The roll must then be placed: it goes in the matching upper box if that box
    is still open; otherwise in any open lower box (where Full House / Small /
    Large straight pay their full 25 / 30 / 40 even though the five-of-a-kind
    does not "really" form them); otherwise it is a forced 0 in an open upper box.
"""

from typing import List, Optional, Tuple

from .dice import Counts, NUM_DICE, NUM_FACES

# ---- Category indices -------------------------------------------------------

ONES, TWOS, THREES, FOURS, FIVES, SIXES = 0, 1, 2, 3, 4, 5
THREE_KIND, FOUR_KIND, FULL_HOUSE = 6, 7, 8
SMALL_STRAIGHT, LARGE_STRAIGHT, YAHTZEE, CHANCE = 9, 10, 11, 12

NUM_CATEGORIES = 13
UPPER_CATEGORIES = (ONES, TWOS, THREES, FOURS, FIVES, SIXES)
LOWER_CATEGORIES = (THREE_KIND, FOUR_KIND, FULL_HOUSE,
                    SMALL_STRAIGHT, LARGE_STRAIGHT, YAHTZEE, CHANCE)

CATEGORY_NAMES = [
    "Ones", "Twos", "Threes", "Fours", "Fives", "Sixes",
    "Three of a Kind", "Four of a Kind", "Full House",
    "Small Straight", "Large Straight", "Yahtzee", "Chance",
]

UPPER_BONUS_THRESHOLD = 63
UPPER_BONUS = 35
YAHTZEE_BONUS = 100

ALL_CATEGORIES = tuple(range(NUM_CATEGORIES))


def is_upper(cat: int) -> bool:
    return cat <= SIXES


def _pip_sum(counts: Counts) -> int:
    return sum((face + 1) * counts[face] for face in range(NUM_FACES))


def _has_run(counts: Counts, length: int) -> bool:
    run = 0
    for face in range(NUM_FACES):
        if counts[face] > 0:
            run += 1
            if run >= length:
                return True
        else:
            run = 0
    return False


def raw_score(cat: int, counts: Counts) -> int:
    """Score of placing ``counts`` in ``cat`` under normal rules (no joker, no
    Yahtzee bonus).  This is what the dice are "worth" ignoring extra-Yahtzee.
    """
    if cat <= SIXES:  # upper section
        return (cat + 1) * counts[cat]
    if cat == THREE_KIND:
        return _pip_sum(counts) if max(counts) >= 3 else 0
    if cat == FOUR_KIND:
        return _pip_sum(counts) if max(counts) >= 4 else 0
    if cat == FULL_HOUSE:
        nz = sorted(c for c in counts if c > 0)
        return 25 if nz == [2, 3] else 0
    if cat == SMALL_STRAIGHT:
        return 30 if _has_run(counts, 4) else 0
    if cat == LARGE_STRAIGHT:
        return 40 if _has_run(counts, 5) else 0
    if cat == YAHTZEE:
        return 50 if max(counts) == NUM_DICE else 0
    if cat == CHANCE:
        return _pip_sum(counts)
    raise ValueError("bad category %r" % (cat,))


def is_yahtzee(counts: Counts) -> bool:
    return max(counts) == NUM_DICE


def yahtzee_face(counts: Counts) -> int:
    """The face (1..6) of a five-of-a-kind roll.  Undefined if not a Yahtzee."""
    return counts.index(NUM_DICE) + 1


def legal_categories(open_mask: int, counts: Counts, bonus_active: bool
                     ) -> List[int]:
    """Categories the player is *allowed* to place ``counts`` into, honouring the
    Yahtzee joker forcing rule.

    ``open_mask`` has bit ``c`` set when category ``c`` is still open.
    ``bonus_active`` is True when the Yahtzee box was previously filled with 50
    (so further Yahtzees earn +100); it does not affect legality but is passed
    through for clarity.
    """
    open_cats = [c for c in ALL_CATEGORIES if open_mask & (1 << c)]

    # Joker forcing only applies when we roll a Yahtzee AND the Yahtzee box is
    # already used up.  Otherwise every open category is legal.
    if not (is_yahtzee(counts) and not (open_mask & (1 << YAHTZEE))):
        return open_cats

    face = yahtzee_face(counts)
    matching_upper = face - 1  # category index of the matching upper box
    if open_mask & (1 << matching_upper):
        # Must use the matching upper box.
        return [matching_upper]
    open_lower = [c for c in LOWER_CATEGORIES if open_mask & (1 << c)]
    if open_lower:
        # Free choice among open lower boxes (joker values apply, see score).
        return open_lower
    # Only upper boxes remain: forced 0 in any of them.
    return [c for c in UPPER_CATEGORIES if open_mask & (1 << c)]


def score_placement(cat: int, counts: Counts, bonus_active: bool) -> int:
    """Points scored by placing ``counts`` in ``cat``, including the +100
    extra-Yahtzee bonus and joker values for straights / full house.

    Does NOT include the upper-section bonus (that is applied by the state
    transition when the upper total crosses the threshold).
    """
    yahtzee_bonus = 0
    joker = is_yahtzee(counts) and bonus_active
    if joker:
        # Extra Yahtzee after a scored 50: +100 regardless of where it lands.
        yahtzee_bonus = YAHTZEE_BONUS

    if is_yahtzee(counts):
        # Joker scoring for the lower-section pattern categories.
        if cat == FULL_HOUSE:
            return 25 + yahtzee_bonus
        if cat == SMALL_STRAIGHT:
            return 30 + yahtzee_bonus
        if cat == LARGE_STRAIGHT:
            return 40 + yahtzee_bonus

    return raw_score(cat, counts) + yahtzee_bonus


def upper_delta(cat: int, counts: Counts, upper_before: int) -> Tuple[int, int]:
    """Return ``(new_capped_upper_total, upper_bonus_earned)`` for placing
    ``counts`` into ``cat``.

    Only the *raw* upper contribution ``(cat+1) * counts[cat]`` counts toward the
    63 threshold -- the +100 extra-Yahtzee bonus never does.  A forced-0 joker
    placement in a non-matching upper box contributes 0.
    """
    if not is_upper(cat):
        return upper_before, 0
    raw_upper = (cat + 1) * counts[cat]
    new_upper = min(UPPER_BONUS_THRESHOLD, upper_before + raw_upper)
    bonus = 0
    if upper_before < UPPER_BONUS_THRESHOLD <= new_upper:
        bonus = UPPER_BONUS
    return new_upper, bonus
