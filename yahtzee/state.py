"""Game-state representation and the shared state-transition enumerator."""

from typing import Iterator, NamedTuple, Tuple

from .dice import Counts
from . import scoring as S


class State(NamedTuple):
    """A single player's scorecard state.

    open_mask:     bit ``c`` set  => category ``c`` is still open (unscored).
    upper_total:   sum of the upper section so far, capped at 63.
    bonus_active:  True once the Yahtzee box has been filled with 50, meaning
                   further Yahtzees earn the +100 bonus.
    """
    open_mask: int
    upper_total: int
    bonus_active: bool


FULL_OPEN_MASK = (1 << S.NUM_CATEGORIES) - 1


def new_game_state() -> State:
    return State(FULL_OPEN_MASK, 0, False)


def is_terminal(state: State) -> bool:
    return state.open_mask == 0


def categories_open(state: State) -> Tuple[int, ...]:
    return tuple(c for c in S.ALL_CATEGORIES if state.open_mask & (1 << c))


class Placement(NamedTuple):
    category: int
    gain: int          # total points added to the final score by this placement
    child: State       # resulting state


# Per-category loose upper bound on the points one placement can ever score
# (ignoring the +100 Yahtzee bonus, which is added separately below).
_CAT_MAX = {
    S.ONES: 5, S.TWOS: 10, S.THREES: 15, S.FOURS: 20, S.FIVES: 25, S.SIXES: 30,
    S.THREE_KIND: 30, S.FOUR_KIND: 30, S.FULL_HOUSE: 25,
    S.SMALL_STRAIGHT: 30, S.LARGE_STRAIGHT: 40, S.YAHTZEE: 50, S.CHANCE: 30,
}

_MAX_ADD_MEMO: dict = {}


def max_additional_ub(state: State) -> int:
    """A safe (over-)estimate of the most points obtainable from ``state`` to the
    end of the game.  Used only for exact saturation pruning, so any valid upper
    bound is fine -- it is generous on purpose."""
    cached = _MAX_ADD_MEMO.get(state)
    if cached is not None:
        return cached
    open_cats = categories_open(state)
    total = sum(_CAT_MAX[c] for c in open_cats)
    if state.upper_total < S.UPPER_BONUS_THRESHOLD:
        total += S.UPPER_BONUS
    # Each remaining turn could in principle land an extra Yahtzee (+100).
    total += S.YAHTZEE_BONUS * len(open_cats)
    _MAX_ADD_MEMO[state] = total
    return total


def enumerate_placements(state: State, dice: Counts) -> Iterator[Placement]:
    """Yield every legal ``Placement`` for scoring ``dice`` from ``state``.

    ``gain`` includes the category points, the +100 extra-Yahtzee bonus, and the
    +35 upper bonus at the moment the upper total crosses 63 -- i.e. everything
    that will show up in the final score.
    """
    for cat in S.legal_categories(state.open_mask, dice, state.bonus_active):
        pts = S.score_placement(cat, dice, state.bonus_active)
        new_upper, upper_bonus = S.upper_delta(cat, dice, state.upper_total)
        gain = pts + upper_bonus
        new_bonus = state.bonus_active or (cat == S.YAHTZEE and S.is_yahtzee(dice))
        child = State(state.open_mask & ~(1 << cat), new_upper, new_bonus)
        yield Placement(cat, gain, child)
