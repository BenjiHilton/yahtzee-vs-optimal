"""Top-level API: given a game state, return the win-optimal play.

Example
-------
    from yahtzee.solver import Solver, GameState, OpponentInfo

    solver = Solver()
    decision = solver.best_play(GameState(
        open_categories={...},          # names or indices still open
        upper_total=52,
        bonus_active=False,
        current_score=180,
        dice=[6, 6, 6, 2, 1],
        rerolls_left=1,
    ), opponent=OpponentInfo(finished_score=205))
    print(decision.describe())
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from . import scoring as S
from .dice import Counts, counts_to_dice, dice_to_counts
from .dice import keep_options
from .engine import (ExpectedScoreSolver, WinProbabilitySolver,
                     _expected_after_keep_idx, _v0_list, rank_keeps,
                     rank_placements)
from .state import enumerate_placements
from .opponent import OpponentModel, make_beat_reward, point_mass
from .state import State, is_terminal

# Map category names (case/spacing insensitive) to indices, plus the indices.
_NAME_TO_IDX = {}
for _i, _n in enumerate(S.CATEGORY_NAMES):
    _NAME_TO_IDX[_n.lower()] = _i
    _NAME_TO_IDX[_n.lower().replace(" ", "")] = _i
# handy aliases
_NAME_TO_IDX.update({
    "3kind": S.THREE_KIND, "threekind": S.THREE_KIND, "3ofakind": S.THREE_KIND,
    "4kind": S.FOUR_KIND, "fourkind": S.FOUR_KIND, "4ofakind": S.FOUR_KIND,
    "fh": S.FULL_HOUSE, "smallstraight": S.SMALL_STRAIGHT, "sm": S.SMALL_STRAIGHT,
    "ss": S.SMALL_STRAIGHT, "largestraight": S.LARGE_STRAIGHT, "lg": S.LARGE_STRAIGHT,
    "ls": S.LARGE_STRAIGHT, "yatzy": S.YAHTZEE, "yahtzee": S.YAHTZEE,
})


def category_index(cat: Union[int, str]) -> int:
    if isinstance(cat, int):
        if not 0 <= cat < S.NUM_CATEGORIES:
            raise ValueError("category index %r out of range" % (cat,))
        return cat
    key = str(cat).strip().lower()
    if key in _NAME_TO_IDX:
        return _NAME_TO_IDX[key]
    key2 = key.replace(" ", "")
    if key2 in _NAME_TO_IDX:
        return _NAME_TO_IDX[key2]
    raise ValueError("unknown category %r" % (cat,))


def _mask_from_open(open_categories: Iterable[Union[int, str]]) -> int:
    mask = 0
    for c in open_categories:
        mask |= 1 << category_index(c)
    return mask


@dataclass
class GameState:
    """The querying player's situation at the moment of decision."""
    open_categories: Iterable[Union[int, str]]
    upper_total: int
    bonus_active: bool
    current_score: int
    dice: Sequence[int]
    rerolls_left: int  # 1 = first roll just made (may reroll once); 0 = must score

    def to_state(self) -> State:
        return State(_mask_from_open(self.open_categories),
                     min(S.UPPER_BONUS_THRESHOLD, self.upper_total),
                     bool(self.bonus_active))

    def counts(self) -> Counts:
        return dice_to_counts(self.dice)


@dataclass
class OpponentInfo:
    """Either a finished opponent (known score) or one still playing."""
    finished_score: Optional[int] = None
    # still-playing opponent:
    open_categories: Optional[Iterable[Union[int, str]]] = None
    upper_total: int = 0
    bonus_active: bool = False
    current_score: int = 0

    def is_finished(self) -> bool:
        return self.finished_score is not None


def _fmt_value(value: float, objective: str) -> str:
    if objective == "win":
        return "win %6.2f%%" % (100.0 * value)
    return "exp %7.2f pts" % value


@dataclass
class ScoreOption:
    category: int
    name: str
    points: int
    value: float          # win probability (objective="win") or expected pts

    def line(self, objective: str = "win") -> str:
        return "%-16s +%3d pts   %s" % (self.name, self.points,
                                        _fmt_value(self.value, objective))


@dataclass
class KeepOption:
    keep: List[int]
    reroll: List[int]
    value: float          # win probability (objective="win") or expected pts

    def line(self, objective: str = "win") -> str:
        keep = " ".join(map(str, self.keep)) or "(none)"
        rr = " ".join(map(str, self.reroll)) or "(none)"
        return "keep [%s]  reroll [%s]   %s" % (keep, rr,
                                                _fmt_value(self.value, objective))


@dataclass
class Decision:
    mode: str                      # "score" or "reroll"
    objective: str                 # "win" or "ev"
    dice: List[int]
    rerolls_left: int
    opponent_note: str
    score_options: List[ScoreOption] = field(default_factory=list)
    keep_options: List[KeepOption] = field(default_factory=list)
    current_win_prob: Optional[float] = None

    @property
    def best_score(self) -> Optional[ScoreOption]:
        return self.score_options[0] if self.score_options else None

    @property
    def best_keep(self) -> Optional[KeepOption]:
        return self.keep_options[0] if self.keep_options else None

    def describe(self, top: int = 5) -> str:
        out = []
        out.append("Dice on table: %s   (%s)" %
                   (" ".join(map(str, self.dice)),
                    "must score, no reroll left" if self.rerolls_left == 0
                    else "one reroll available"))
        out.append(self.opponent_note)
        unit = "win%" if self.objective == "win" else "exp. pts"
        if self.mode == "reroll":
            out.append("")
            out.append("Best move: %s" % self.best_keep.line(self.objective))
            out.append("Alternatives (ranked by %s):" % unit)
            for k in self.keep_options[1:top]:
                out.append("   " + k.line(self.objective))
        else:
            out.append("")
            out.append("Best move: score in %s" % self.best_score.name)
            out.append("All legal placements (ranked by %s):" % unit)
            for s in self.score_options[:top]:
                marker = " <-- best" if s is self.best_score else ""
                out.append("   " + s.line(self.objective) + marker)
        return "\n".join(out)


class Solver:
    """Reusable solver; caches the expected-score backbone across queries.

    ``max_exact_open`` bounds how many of *your* categories may still be open
    before an exact win-probability query is refused (it becomes slow because
    the running-score dimension multiplies the state space).  Exact
    win-probability is meant for the endgame -- the regime where it actually
    differs from expected-score play.  For earlier states use ``objective='ev'``
    (instant, and near-optimal for win rate when many turns remain), or pass
    ``allow_slow=True`` to force it.
    """

    #: default location of the precomputed expected-score table (see
    #: precompute_ev.py); auto-loaded if present.  The compact .npz is preferred
    #: (small + fast to load); the .pkl still works as a fallback.
    DEFAULT_EV_CACHE = "ev_table.npz"

    def __init__(self, tie_value: float = 0.5, max_exact_open: int = 7,
                 max_opp_dist_open: int = 6,
                 ev_cache: Optional[str] = DEFAULT_EV_CACHE) -> None:
        self.tie_value = tie_value
        self.max_exact_open = max_exact_open
        self.max_opp_dist_open = max_opp_dist_open
        self.ev = ExpectedScoreSolver()
        if ev_cache:
            import os
            path = ev_cache
            if not os.path.isabs(path) and not os.path.exists(path):
                # also look next to this package's parent (project root)
                here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                cand = os.path.join(here, ev_cache)
                if os.path.exists(cand):
                    path = cand
            if path.endswith(".npz"):
                self.ev.load_npz(path)
            else:
                self.ev.load_cache(path)
        self.opponent_model = OpponentModel(self.ev)

    # -- win-probability reward construction --------------------------------

    def _reward_and_note(self, gs: GameState, opponent: OpponentInfo):
        warning = None
        if opponent.is_finished():
            opp_pmf = point_mass(opponent.finished_score)
            note = ("Opponent has finished on %d. You are %+d going in; "
                    "playing to maximise P(final total > %d)." %
                    (opponent.finished_score,
                     gs.current_score - opponent.finished_score,
                     opponent.finished_score))
        else:
            if opponent.open_categories is None:
                raise ValueError("still-playing opponent needs open_categories")
            opp_state = State(_mask_from_open(opponent.open_categories),
                              min(S.UPPER_BONUS_THRESHOLD, opponent.upper_total),
                              bool(opponent.bonus_active))
            n_opp_open = bin(opp_state.open_mask).count("1")
            # Alternating play: on your move the opponent has n or n-1 open, where
            # n is your open count (equal if you move first in the round, one
            # fewer if you move second).  Anything else is almost certainly a
            # data-entry slip, so flag it.
            my_open = bin(gs.to_state().open_mask).count("1")
            if n_opp_open not in (my_open, my_open - 1):
                warning = (
                    "NOTE: you have %d categories open but the opponent has %d. "
                    "With alternating play the opponent should have %d or %d open "
                    "on your move -- double-check the opponent's open categories."
                    % (my_open, n_opp_open, my_open, max(my_open - 1, 0)))
            if n_opp_open <= self.max_opp_dist_open:
                opp_pmf = self.opponent_model.final_distribution(
                    opp_state, opponent.current_score)
                exp = sum(k * v for k, v in opp_pmf.items())
                note = ("Opponent still playing (modelled as expected-score-optimal); "
                        "their projected final total ~ %.1f (full distribution). "
                        "Maximising P(you beat them)." % exp)
            else:
                # Too many categories open to compute the full distribution
                # quickly; approximate the opponent by their expected final total.
                exp_add = self.opponent_model.ev.eadd(opp_state)
                exp = opponent.current_score + exp_add
                opp_pmf = point_mass(int(round(exp)))
                note = ("Opponent still playing with %d categories open; "
                        "approximated by their expected final total ~ %.1f "
                        "(mean only -- their variance is ignored this far out). "
                        "Maximising P(you beat that)." % (n_opp_open, exp))
        if warning:
            note = warning + "\n" + note
        return make_beat_reward(opp_pmf, self.tie_value), note

    # -- main entry point ---------------------------------------------------

    def best_play(self, gs: GameState,
                  opponent: OpponentInfo,
                  objective: str = "win",
                  allow_slow: bool = False) -> Decision:
        state = gs.to_state()
        if is_terminal(state):
            raise ValueError("no categories open -- game is over")
        dice = gs.counts()
        banked = gs.current_score

        # Secondary ranking key: among moves that tie on the primary objective
        # (e.g. every move when the win is already guaranteed at 100%), prefer the
        # one with the highest expected final score.  None means no tiebreak.
        tiebreak = None
        if objective == "ev":
            child_value = self.ev.child_value()
            note = "Objective: maximise your own expected final score (opponent ignored)."
        elif objective == "win":
            n_open = bin(state.open_mask).count("1")
            if n_open > self.max_exact_open and not allow_slow:
                raise ValueError(
                    "exact win-probability with %d categories still open is slow "
                    "(the running-score dimension blows up). This engine is exact "
                    "for the endgame (<= %d open). For an earlier state use "
                    "objective='ev' (near-optimal for win rate this far out), or "
                    "pass allow_slow=True to force it." % (n_open, self.max_exact_open))
            reward, note = self._reward_and_note(gs, opponent)
            win = WinProbabilitySolver(reward)
            child_value = win.child_value(banked)
            tiebreak = self.ev.child_value()  # expected score breaks win-prob ties
        else:
            raise ValueError("objective must be 'win' or 'ev'")

        if gs.rerolls_left <= 0:
            if tiebreak is None:
                ranked = rank_placements(state, dice, child_value)
                opts = [ScoreOption(p.category, S.CATEGORY_NAMES[p.category], p.gain, val)
                        for val, p in ranked]
            else:
                scored = [(child_value(p.child, p.gain), tiebreak(p.child, p.gain), p)
                          for p in enumerate_placements(state, dice)]
                # round the win value so float noise doesn't mask a true tie
                scored.sort(key=lambda t: (round(t[0], 9), t[1]), reverse=True)
                opts = [ScoreOption(p.category, S.CATEGORY_NAMES[p.category], p.gain, wv)
                        for wv, _tv, p in scored]
            return Decision("score", objective, list(gs.dice), 0, note,
                            score_options=opts)
        else:
            if tiebreak is None:
                ranked = [(val, None, kept) for val, kept in
                          rank_keeps(state, dice, child_value)]
            else:
                v0w = _v0_list(state, child_value)
                v0e = _v0_list(state, tiebreak)
                ranked = [(_expected_after_keep_idx(kept, v0w),
                           _expected_after_keep_idx(kept, v0e), kept)
                          for kept in keep_options(dice)]
                # round the win value so float noise doesn't mask a true tie
                ranked.sort(key=lambda t: (round(t[0], 9), t[1]), reverse=True)
            opts = []
            for val, _tv, kept in ranked:
                keep_dice = counts_to_dice(kept)
                reroll_dice = counts_to_dice(tuple(dice[i] - kept[i] for i in range(6)))
                opts.append(KeepOption(keep_dice, reroll_dice, val))
            return Decision("reroll", objective, list(gs.dice), 1, note,
                            keep_options=opts)
