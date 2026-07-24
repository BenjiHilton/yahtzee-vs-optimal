"""Dice representation and reroll probability transitions.

A roll of 5 dice is represented as a *count vector*: a tuple of length 6 where
index ``i`` holds how many dice show face ``i+1``.  For example five dice showing
``(1, 1, 3, 4, 4)`` is ``(2, 0, 1, 2, 0, 0)`` and always sums to 5.

Working in count vectors (rather than ordered tuples) collapses the 6**5 = 7776
ordered outcomes down to the 252 distinct multisets, which is what makes the
game DP tractable.
"""

from functools import lru_cache
from itertools import product
from math import factorial
from typing import Dict, List, Tuple

Counts = Tuple[int, int, int, int, int, int]

NUM_DICE = 5
NUM_FACES = 6


def _all_count_vectors(n: int) -> List[Counts]:
    """All length-6 count vectors summing to ``n`` (multisets of ``n`` dice)."""
    out: List[Counts] = []

    def rec(face: int, remaining: int, acc: List[int]) -> None:
        if face == NUM_FACES - 1:
            out.append(tuple(acc + [remaining]))  # type: ignore[arg-type]
            return
        for c in range(remaining + 1):
            rec(face + 1, remaining - c, acc + [c])

    rec(0, n, [])
    return out


# The 252 possible full rolls of 5 dice.
ALL_ROLLS: List[Counts] = _all_count_vectors(NUM_DICE)


def _multinomial(counts: Counts) -> int:
    """Number of ordered sequences that map to this count vector."""
    total = sum(counts)
    result = factorial(total)
    for c in counts:
        result //= factorial(c)
    return result


def roll_probability(counts: Counts) -> float:
    """Probability of obtaining ``counts`` when rolling ``sum(counts)`` fair dice."""
    n = sum(counts)
    if n == 0:
        return 1.0
    return _multinomial(counts) / (NUM_FACES ** n)


# Probability distribution over the result of rerolling exactly ``m`` dice,
# for m in 0..5.  Each entry maps a count vector of ``m`` dice to its probability.
_REROLL_DIST: Dict[int, List[Tuple[Counts, float]]] = {}
for _m in range(NUM_DICE + 1):
    _REROLL_DIST[_m] = [(cv, roll_probability(cv)) for cv in _all_count_vectors(_m)]


@lru_cache(maxsize=None)
def keep_options(roll: Counts) -> Tuple[Counts, ...]:
    """All distinct sub-multisets of ``roll`` that a player may keep.

    Each option is itself a count vector with ``kept[i] <= roll[i]``.  Keeping
    everything (no reroll) and keeping nothing (reroll all five) are both
    included.
    """
    ranges = [range(roll[i] + 1) for i in range(NUM_FACES)]
    return tuple(product(*ranges))  # type: ignore[return-value]


@lru_cache(maxsize=None)
def reroll_outcomes(kept: Counts) -> Tuple[Tuple[Counts, float], ...]:
    """Distribution over full 5-dice rolls after keeping ``kept`` and rerolling
    the remaining ``5 - sum(kept)`` dice.

    Returns a tuple of ``(full_roll_counts, probability)`` pairs.
    """
    m = NUM_DICE - sum(kept)
    out: List[Tuple[Counts, float]] = []
    for new_counts, prob in _REROLL_DIST[m]:
        full = tuple(kept[i] + new_counts[i] for i in range(NUM_FACES))
        out.append((full, prob))  # type: ignore[arg-type]
    return tuple(out)


# ---- Index tables for the hot path -----------------------------------------
# Every full 5-dice roll gets a stable integer index so value tables can be
# plain lists instead of dict-keyed-by-tuple.
FULL_INDEX: Dict[Counts, int] = {cv: i for i, cv in enumerate(ALL_ROLLS)}
ROLL_PROB_LIST: List[float] = [roll_probability(cv) for cv in ALL_ROLLS]

# For each keep option, the reroll outcome distribution as (full_index, prob).
REROLL_IDX: Dict[Counts, Tuple[Tuple[int, float], ...]] = {}
for _k in range(NUM_DICE + 1):
    for _kept in _all_count_vectors(_k):
        REROLL_IDX[_kept] = tuple(
            (FULL_INDEX[full], prob) for full, prob in reroll_outcomes(_kept))

# For each full roll, the tuple of keep options (sub-multisets).
KEEP_OPTS: List[Tuple[Counts, ...]] = [keep_options(cv) for cv in ALL_ROLLS]


def counts_to_dice(counts: Counts) -> List[int]:
    """Expand a count vector into a sorted list of face values, e.g.
    ``(2, 0, 1, 0, 0, 0) -> [1, 1, 3]``."""
    dice: List[int] = []
    for face in range(NUM_FACES):
        dice.extend([face + 1] * counts[face])
    return dice


def dice_to_counts(dice) -> Counts:
    """Build a count vector from an iterable of face values (1..6)."""
    counts = [0] * NUM_FACES
    for d in dice:
        if not 1 <= d <= NUM_FACES:
            raise ValueError("die value %r out of range 1..6" % (d,))
        counts[d - 1] += 1
    if sum(counts) != NUM_DICE:
        raise ValueError("expected %d dice, got %d" % (NUM_DICE, sum(counts)))
    return tuple(counts)  # type: ignore[return-value]
