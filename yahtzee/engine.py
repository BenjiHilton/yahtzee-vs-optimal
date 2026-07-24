"""The game DP: single-turn optimisation and full-game backward induction.

Two solvers share the same turn machinery:

* ``ExpectedScoreSolver`` computes, for every state, the expected additional
  final score under optimal play (leaf reward = the score itself).  This is the
  classic self-play-optimal Yahtzee strategy and the backbone used for opponent
  modelling.

* ``WinProbabilitySolver`` maximises ``E[G(final_total)]`` for an arbitrary
  non-decreasing reward ``G`` -- in practice ``G(t) = P(you beat the opponent
  when your final total is t)``.  This yields win-probability-optimal play.

A "turn" is: see the first roll (rr=1 reroll remaining), optionally keep a
sub-multiset and reroll the rest once (rr=0), then score.  The functions below
optimise the keep decision and the final category choice exactly.
"""

from typing import Callable, Dict, List, Optional, Tuple

from .dice import (ALL_ROLLS, Counts, KEEP_OPTS, REROLL_IDX, ROLL_PROB_LIST,
                   keep_options, reroll_outcomes)
from .scoring import NUM_CATEGORIES, UPPER_BONUS, UPPER_BONUS_THRESHOLD
from .state import (Placement, State, enumerate_placements, is_terminal,
                    max_additional_ub)

# child_value(child_state, gain) -> float
ChildValue = Callable[[State, int], float]

NEG_INF = float("-inf")


def best_placement(state: State, dice: Counts, child_value: ChildValue
                   ) -> Tuple[float, Placement]:
    """Best category to score ``dice`` in from ``state`` (the rr=0 decision)."""
    best_val = NEG_INF
    best_p: Optional[Placement] = None
    for p in enumerate_placements(state, dice):
        v = child_value(p.child, p.gain)
        if v > best_val:
            best_val = v
            best_p = p
    assert best_p is not None
    return best_val, best_p


def best_placement_value(state: State, dice: Counts, child_value: ChildValue) -> float:
    """Just the value of the best placement (hot path -- avoids building tuples)."""
    best_val = NEG_INF
    for p in enumerate_placements(state, dice):
        v = child_value(p.child, p.gain)
        if v > best_val:
            best_val = v
    return best_val


def rank_placements(state: State, dice: Counts, child_value: ChildValue
                    ) -> List[Tuple[float, Placement]]:
    """All legal placements for ``dice``, ranked best-first."""
    scored = [(child_value(p.child, p.gain), p) for p in enumerate_placements(state, dice)]
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored


def _v0_list(state: State, child_value: ChildValue) -> List[float]:
    """best-placement value for every final roll, indexed by FULL_INDEX."""
    return [best_placement_value(state, full, child_value) for full in ALL_ROLLS]


def _expected_after_keep_idx(kept: Counts, v0: List[float]) -> float:
    total = 0.0
    for idx, prob in REROLL_IDX[kept]:
        total += prob * v0[idx]
    return total


try:
    import numpy as _np
    from .fastcore import (turn_value_from_v0 as _turn_value_from_v0,
                           RAW as _RAW, CNT as _CNT, YAHTZEE_ROLLS as _YZ_ROLLS)
    _HAVE_FAST = True
except Exception:  # pragma: no cover - numpy missing
    _HAVE_FAST = False


def _turn_value_from_v0_py(v0: List[float]) -> float:
    e_kept: Dict[Counts, float] = {}
    ekget = e_kept.get
    value = 0.0
    for ri in range(len(ALL_ROLLS)):
        best_e = NEG_INF
        for kept in KEEP_OPTS[ri]:
            e = ekget(kept)
            if e is None:
                e = _expected_after_keep_idx(kept, v0)
                e_kept[kept] = e
            if e > best_e:
                best_e = e
        value += ROLL_PROB_LIST[ri] * best_e
    return value


def turn_start_value(state: State, child_value: ChildValue) -> float:
    """Expected value at the very start of a turn (before the first roll), with
    one reroll available.  Averages over the opening roll, optimising the keep
    decision for each."""
    v0 = _v0_list(state, child_value)
    if _HAVE_FAST:
        return _turn_value_from_v0(v0)
    return _turn_value_from_v0_py(v0)


def rank_keeps(state: State, dice: Counts, child_value: ChildValue
               ) -> List[Tuple[float, Counts]]:
    """With one reroll available and current ``dice`` on the table, rank every
    keep option (sub-multiset to hold) best-first by expected resulting value."""
    v0 = _v0_list(state, child_value)
    ranked = [(_expected_after_keep_idx(kept, v0), kept) for kept in keep_options(dice)]
    ranked.sort(key=lambda t: t[0], reverse=True)
    return ranked


# Backwards-compatible name used by opponent.py
def _v0_table(state: State, child_value: ChildValue) -> List[float]:
    return _v0_list(state, child_value)


def _expected_after_keep(kept: Counts, v0: List[float]) -> float:
    return _expected_after_keep_idx(kept, v0)


# ---------------------------------------------------------------------------
# Expected-score solver (linear reward -> memoise on state alone)
# ---------------------------------------------------------------------------

class ExpectedScoreSolver:
    """Expected additional final score under optimal (self-play) play."""

    def __init__(self) -> None:
        self._eadd: Dict[State, float] = {}
        # Compact preloaded table: sorted uint32 packed-state keys + float32 values,
        # binary-searched.  ~5x less memory than a dict and loads near-instantly
        # (important on small/slow hosts -- avoids the unpickling memory spike).
        self._keys = None
        self._vals = None

    @staticmethod
    def _pack(state: State) -> int:
        # open_mask (13 bits) | upper_total 0..63 (6 bits) | bonus (1 bit)
        return (state.open_mask << 7) | (state.upper_total << 1) | (1 if state.bonus_active else 0)

    def load_cache(self, path: str) -> bool:
        """Load a previously saved eadd table.  Returns True on success."""
        import os
        import pickle
        if not os.path.exists(path):
            return False
        with open(path, "rb") as fh:
            self._eadd.update(pickle.load(fh))
        return True

    def save_cache(self, path: str) -> None:
        import pickle
        with open(path, "wb") as fh:
            pickle.dump(self._eadd, fh, protocol=pickle.HIGHEST_PROTOCOL)

    def load_npz(self, path: str) -> bool:
        """Load the compact array table (keys+vals). Returns True on success."""
        import os
        if not (_HAVE_FAST and os.path.exists(path)):
            return False
        data = _np.load(path)
        self._keys = data["keys"]
        self._vals = data["vals"]
        return True

    def save_npz(self, path: str) -> None:
        """Write the current dict table as a compact sorted key/value archive."""
        items = list(self._eadd.items())
        keys = _np.fromiter((self._pack(s) for s, _ in items),
                            dtype=_np.uint32, count=len(items))
        vals = _np.fromiter((v for _, v in items),
                            dtype=_np.float64, count=len(items))
        order = _np.argsort(keys, kind="stable")
        _np.savez(path, keys=keys[order], vals=vals[order].astype(_np.float32))

    def eadd(self, state: State) -> float:
        """Expected additional points from ``state`` to the end of the game."""
        if is_terminal(state):
            return 0.0
        cached = self._eadd.get(state)
        if cached is not None:
            return cached
        if self._keys is not None:
            packed = self._pack(state)
            idx = int(_np.searchsorted(self._keys, packed))
            if idx < self._keys.shape[0] and self._keys[idx] == packed:
                v = float(self._vals[idx])
                self._remember(state, v)  # cache so repeat lookups are O(1)
                return v
        if _HAVE_FAST:
            val = _turn_value_from_v0(self._v0_fast(state))
        else:
            val = turn_start_value(state, self._child_value)
        self._remember(state, val)
        return val

    def _remember(self, state: State, val: float) -> None:
        # Bounded runtime cache: the same child states are looked up hundreds of
        # times within one turn, so caching them makes each turn much faster; the
        # cap keeps memory low even after many games on a shared server.
        if len(self._eadd) >= 200000:
            self._eadd.clear()
        self._eadd[state] = val

    def _v0_fast(self, state: State):
        """Vectorised best-placement value for every final roll (expected-score
        DP only).  Exploits that ``eadd(child)`` is constant across rolls except
        for upper categories and the six five-of-a-kind rolls."""
        mask = state.open_mask
        upper = state.upper_total
        bonus = state.bonus_active
        contribs = []
        for c in range(NUM_CATEGORIES):
            if not (mask & (1 << c)):
                continue
            child_mask = mask & ~(1 << c)
            if c <= 5:  # upper box: child upper total varies with the face count
                addon = _np.empty(6)
                for cnt in range(6):
                    raw = (c + 1) * cnt
                    new_upper = upper + raw
                    if new_upper >= UPPER_BONUS_THRESHOLD:
                        new_upper = UPPER_BONUS_THRESHOLD
                    ub = UPPER_BONUS if (upper < UPPER_BONUS_THRESHOLD <=
                                         upper + raw) else 0
                    addon[cnt] = ub + self.eadd(State(child_mask, new_upper, bonus))
                contribs.append(_RAW[c] + addon[_CNT[c]])
            else:  # non-upper: child is the same for every non-Yahtzee roll
                const = self.eadd(State(child_mask, upper, bonus))
                contribs.append(_RAW[c] + const)
        v0 = _np.max(contribs, axis=0)
        # The six five-of-a-kind rolls need exact joker/+100 handling.
        cv = self._child_value
        for idx, _face in _YZ_ROLLS:
            v0[idx] = best_placement_value(state, ALL_ROLLS[idx], cv)
        return v0

    def _child_value(self, child: State, gain: int) -> float:
        return gain + self.eadd(child)

    def child_value(self) -> ChildValue:
        return self._child_value


# ---------------------------------------------------------------------------
# Win-probability solver (general reward -> memoise on (state, banked))
# ---------------------------------------------------------------------------

class WinProbabilitySolver:
    """Maximise ``E[reward(final_total)]``.

    ``reward`` is any function of the final total (typically the probability of
    beating the opponent).  ``banked`` threads the player's running total so the
    reward can be evaluated at the leaves.
    """

    def __init__(self, reward: Callable[[int], float]) -> None:
        self.reward = reward
        self._memo: Dict[Tuple[State, int], float] = {}

    def value(self, state: State, banked: int) -> float:
        if is_terminal(state):
            return self.reward(banked)
        # Exact saturation prune: additional points lie in [0, max_add_ub] and
        # reward() is non-decreasing, so if the reward is already the same at
        # both ends of that interval it is constant over every reachable
        # outcome -- no need to recurse.
        lo = self.reward(banked)
        hi = self.reward(banked + max_additional_ub(state))
        if lo == hi:
            return lo
        key = (state, banked)
        cached = self._memo.get(key)
        if cached is not None:
            return cached
        val = turn_start_value(state, self._child_value_for(banked))
        self._memo[key] = val
        return val

    def _child_value_for(self, banked: int) -> ChildValue:
        def cv(child: State, gain: int) -> float:
            return self.value(child, banked + gain)
        return cv

    def child_value(self, banked: int) -> ChildValue:
        return self._child_value_for(banked)
