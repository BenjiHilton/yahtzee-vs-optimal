"""Opponent modelling: the distribution of the opponent's final score.

To turn "maximise win probability" into a concrete reward, we need to know what
we are shooting at.  Two cases:

* The opponent has already finished -> their final total is a known constant
  (a point-mass distribution).  Play against it is then *exactly* optimal.

* The opponent is still playing -> we model them as playing the
  expected-score-optimal strategy and compute the exact distribution of their
  final total from their current state.  This is the one modelling assumption in
  the engine (documented in the README): we cannot cheaply solve the fully
  adaptive 2-player game, so we assume a strong, fixed opponent policy.

The distribution DP is vectorised (NumPy) so a still-playing opponent with
several categories open is solved in a fraction of a second rather than minutes.
"""

from typing import Dict

from .engine import ExpectedScoreSolver, best_placement
from .state import State, is_terminal, max_additional_ub

Pmf = Dict[int, float]

# Score-support length for the array-based distributions.  Additional points from
# any real state sit far below this; anything beyond is clamped into the top bin
# (it only ever means "opponent scored more than you can" -> a loss either way).
_SUPPORT = 512

try:
    import numpy as np
    from . import fastcore as _fc
    _HAVE_NP = True
except Exception:  # pragma: no cover
    _HAVE_NP = False


def _shift(pmf: Pmf, offset: int) -> Pmf:
    if offset == 0:
        return dict(pmf)
    return {k + offset: v for k, v in pmf.items()}


class OpponentModel:
    """Distribution of additional / final score under expected-score-optimal play."""

    def __init__(self, ev: ExpectedScoreSolver) -> None:
        self.ev = ev
        self._reldist_arr: Dict[State, "np.ndarray"] = {}
        self._reldist_pmf: Dict[State, Pmf] = {}

    # -- vectorised array DP ------------------------------------------------

    def _reldist_array(self, state: State):
        cached = self._reldist_arr.get(state)
        if cached is not None:
            return cached
        if is_terminal(state):
            arr = np.zeros(_SUPPORT)
            arr[0] = 1.0
            self._reldist_arr[state] = arr
            return arr

        cv = self.ev.child_value()
        v0 = self.ev._v0_fast(state)  # 252-vector of best rr=0 values (cached eadd)

        # D[full] = child additional-score distribution shifted by this turn's gain.
        D = np.zeros((_fc.N_FULL, _SUPPORT))
        from .dice import ALL_ROLLS
        for fi, full in enumerate(ALL_ROLLS):
            _, p = best_placement(state, full, cv)
            child_dist = self._reldist_array(p.child)
            g = p.gain
            if g < _SUPPORT:
                D[fi, g:] = child_dist[:_SUPPORT - g]

        # Distribution after each keep option, then pick the EV-optimal keep per
        # opening roll (same choice the expected-score policy makes).
        kept_dist = _fc.K.dot(D)                 # (462, SUPPORT)
        e_kept = _fc.K.dot(v0)                   # (462,) values for the argmax
        best_rows = np.empty(_fc.N_FULL, dtype=np.intp)
        for ri in range(_fc.N_FULL):
            rows = _fc.ROLL_KEEP_ROWS[ri]
            best = rows[0]
            bestv = e_kept[best]
            for r in rows[1:]:
                if e_kept[r] > bestv:
                    bestv = e_kept[r]
                    best = r
            best_rows[ri] = best
        result = _fc.ROLL_PROB.dot(kept_dist[best_rows])  # (SUPPORT,)
        self._reldist_arr[state] = result
        return result

    # -- public pmf interface ----------------------------------------------

    def relative_distribution(self, state: State) -> Pmf:
        """Pmf over the *additional* points scored from ``state`` to the end of
        the game, following the expected-score-optimal policy."""
        cached = self._reldist_pmf.get(state)
        if cached is not None:
            return cached
        if not _HAVE_NP:  # pragma: no cover - pure-Python fallback
            pmf = self._reldist_pmf_pure(state)
        else:
            arr = self._reldist_array(state)
            top = max_additional_ub(state)
            pmf = {i: float(arr[i]) for i in range(min(top + 1, _SUPPORT))
                   if arr[i] > 0.0}
        self._reldist_pmf[state] = pmf
        return pmf

    def final_distribution(self, state: State, current_score: int) -> Pmf:
        """Pmf over the opponent's *final total*, given their current banked
        score and state."""
        return _shift(self.relative_distribution(state), current_score)

    # -- pure-Python fallback (correct, slow) ------------------------------

    def _reldist_pmf_pure(self, state: State) -> Pmf:
        from .dice import ALL_ROLLS, keep_options, reroll_outcomes, roll_probability
        from .engine import _expected_after_keep, _v0_table
        if is_terminal(state):
            return {0: 1.0}
        cv = self.ev.child_value()
        v0 = _v0_table(state, cv)
        best_kept_cache: Dict = {}
        result: Pmf = {}
        for roll in ALL_ROLLS:
            prob_roll = roll_probability(roll)
            best_e = float("-inf")
            best_kept = None
            for kept in keep_options(roll):
                e = best_kept_cache.get(kept)
                if e is None:
                    e = _expected_after_keep(kept, v0)
                    best_kept_cache[kept] = e
                if e > best_e:
                    best_e = e
                    best_kept = kept
            for full, prob_full in reroll_outcomes(best_kept):
                _, placement = best_placement(state, full, cv)
                child = self._reldist_pmf_pure(placement.child)
                w = prob_roll * prob_full
                for k, v in child.items():
                    key = k + placement.gain
                    result[key] = result.get(key, 0.0) + w * v
        return result


def point_mass(final_score: int) -> Pmf:
    """Distribution for an opponent who has already finished on ``final_score``."""
    return {final_score: 1.0}


def make_beat_reward(opponent_final: Pmf, tie_value: float = 0.5):
    """Build ``reward(my_final_total) -> P(win)`` from the opponent's final-score
    distribution.

    ``tie_value`` is the credit for an exact tie (0.5 = a draw counts as half a
    win, 1.0 = ties go to you, 0.0 = ties go to the opponent).
    """
    support = sorted(opponent_final.items())

    def reward(my_final: int) -> float:
        below = 0.0
        equal = 0.0
        for score, prob in support:
            if score < my_final:
                below += prob
            elif score == my_final:
                equal += prob
            else:
                break
        return below + tie_value * equal

    return reward
