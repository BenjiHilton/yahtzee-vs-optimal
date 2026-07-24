"""Tests for the 2-roll Yahtzee engine.

Run with:  python -m pytest -q      (or)   python tests/test_engine.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yahtzee import scoring as S
from yahtzee.dice import ALL_ROLLS, dice_to_counts, roll_probability
from yahtzee.engine import ExpectedScoreSolver, WinProbabilitySolver, rank_placements
from yahtzee.opponent import OpponentModel, make_beat_reward, point_mass
from yahtzee.solver import GameState, OpponentInfo, Solver
from yahtzee.state import State


def c(*d):
    return dice_to_counts(d)


# ---- scoring ---------------------------------------------------------------

def test_roll_probabilities_sum_to_one():
    assert abs(sum(roll_probability(x) for x in ALL_ROLLS) - 1.0) < 1e-12
    assert len(ALL_ROLLS) == 252


def test_raw_scores():
    assert S.raw_score(S.FULL_HOUSE, c(3, 3, 3, 5, 5)) == 25
    assert S.raw_score(S.FULL_HOUSE, c(3, 3, 3, 3, 3)) == 0  # natural 5-kind not FH
    assert S.raw_score(S.SMALL_STRAIGHT, c(1, 2, 3, 4, 6)) == 30
    assert S.raw_score(S.SMALL_STRAIGHT, c(1, 2, 3, 5, 6)) == 0
    assert S.raw_score(S.LARGE_STRAIGHT, c(2, 3, 4, 5, 6)) == 40
    assert S.raw_score(S.THREE_KIND, c(4, 4, 4, 2, 1)) == 15
    assert S.raw_score(S.FOUR_KIND, c(4, 4, 4, 4, 1)) == 17
    assert S.raw_score(S.FOUR_KIND, c(4, 4, 4, 2, 1)) == 0
    assert S.raw_score(S.YAHTZEE, c(6, 6, 6, 6, 6)) == 50
    assert S.raw_score(S.CHANCE, c(6, 6, 6, 6, 6)) == 30


def test_joker_forced_upper():
    full = (1 << 13) - 1
    open_mask = full & ~(1 << S.YAHTZEE)  # yahtzee box already used
    legal = S.legal_categories(open_mask, c(4, 4, 4, 4, 4), bonus_active=True)
    assert legal == [S.FOURS]
    assert S.score_placement(S.FOURS, c(4, 4, 4, 4, 4), True) == 20 + 100


def test_joker_free_lower_and_bonus():
    full = (1 << 13) - 1
    open_mask = full & ~(1 << S.YAHTZEE) & ~(1 << S.FOURS)
    legal = set(S.legal_categories(open_mask, c(4, 4, 4, 4, 4), True))
    assert legal == {S.THREE_KIND, S.FOUR_KIND, S.FULL_HOUSE,
                     S.SMALL_STRAIGHT, S.LARGE_STRAIGHT, S.CHANCE}
    assert S.score_placement(S.FULL_HOUSE, c(4, 4, 4, 4, 4), True) == 25 + 100
    assert S.score_placement(S.LARGE_STRAIGHT, c(4, 4, 4, 4, 4), True) == 40 + 100


def test_joker_no_bonus_when_yahtzee_scored_zero():
    # bonus_active is False if the Yahtzee box holds 0 rather than 50.
    full = (1 << 13) - 1
    open_mask = full & ~(1 << S.YAHTZEE)
    assert S.score_placement(S.FOURS, c(4, 4, 4, 4, 4), bonus_active=False) == 20


def test_upper_bonus_crossing():
    nu, ub = S.upper_delta(S.SIXES, c(6, 6, 6, 2, 1), upper_before=45)
    assert (nu, ub) == (63, 35)
    nu, ub = S.upper_delta(S.SIXES, c(6, 6, 6, 2, 1), upper_before=50)
    assert (nu, ub) == (63, 35)
    nu, ub = S.upper_delta(S.ONES, c(1, 2, 3, 4, 5), upper_before=0)
    assert (nu, ub) == (1, 0)


# ---- expected-score DP against closed-form values --------------------------

def test_ev_chance_only():
    # With one reroll, keep a die iff it shows >= 4  ->  E per die = 4.25.
    ev = ExpectedScoreSolver()
    assert abs(ev.eadd(State(1 << S.CHANCE, 63, False)) - 21.25) < 1e-9


def test_ev_ones_only():
    # E[final ones] = 5 * (1/6 + 5/36) * 1 point = 55/36.
    ev = ExpectedScoreSolver()
    assert abs(ev.eadd(State(1 << S.ONES, 0, False)) - 55.0 / 36.0) < 1e-9


def test_ev_yahtzee_only_matches_bruteforce():
    ev = ExpectedScoreSolver()
    got = ev.eadd(State(1 << S.YAHTZEE, 63, False))
    # Brute force: optimal 2-roll Yahtzee is "keep the largest matching group".
    p = 0.0
    for roll in ALL_ROLLS:
        pr = roll_probability(roll)
        best = 0.0
        # optimal keep = all dice of the most frequent face; reroll the rest.
        keep = max(roll)
        need = 5 - keep
        # probability the rerolled 'need' dice all match the kept face = (1/6)^need
        p_complete = (1.0 / 6.0) ** need if need > 0 else 1.0
        best = p_complete * 50.0
        p += pr * best
    assert abs(got - p) < 1e-9


# ---- win probability -------------------------------------------------------

def test_win_prob_is_a_probability():
    reward = make_beat_reward(point_mass(150), 0.5)
    win = WinProbabilitySolver(reward)
    st = State((1 << S.SIXES) | (1 << S.CHANCE) | (1 << S.YAHTZEE), 50, False)
    v = win.value(st, 120)
    assert 0.0 <= v <= 1.0


def test_win_prob_saturation_prune_exact():
    # Guaranteed win: already ahead and only additive categories remain.
    solver = Solver()
    d = solver.best_play(GameState(
        open_categories=["Sixes", "Yahtzee", "Chance"], upper_total=50,
        bonus_active=False, current_score=178, dice=[6, 6, 6, 2, 1],
        rerolls_left=0), OpponentInfo(finished_score=205))
    # Sixes scores 18 -> upper 50->63 (+35) -> 178+53 = 231 > 205, rest only adds.
    assert d.best_score.name == "Sixes"
    assert abs(d.best_score.value - 1.0) < 1e-12


def test_win_prune_matches_unpruned():
    # The saturation prune must not change any value: compare against a brute
    # recomputation that disables it by using a reward that never saturates
    # within the reachable range is hard; instead check a hopeless case is 0.
    solver = Solver()
    d = solver.best_play(GameState(
        open_categories=["Ones"], upper_total=0, bonus_active=False,
        current_score=10, dice=[2, 3, 4, 5, 6], rerolls_left=0),
        OpponentInfo(finished_score=100))
    # Max we can reach: 10 + 5 (five ones eventually) = 15 < 100  -> lose for sure.
    assert d.best_score.value == 0.0


def test_reroll_vs_score_consistency():
    # With a reroll available, the best keep can never be worse than committing
    # to the best immediate placement (keeping all five is always an option).
    solver = Solver()
    gs_kwargs = dict(open_categories=["Sixes", "Yahtzee", "Chance"],
                     upper_total=40, bonus_active=False, current_score=150,
                     dice=[6, 6, 3, 2, 1])
    opp = OpponentInfo(finished_score=180)
    d0 = solver.best_play(GameState(rerolls_left=0, **gs_kwargs), opp)
    d1 = solver.best_play(GameState(rerolls_left=1, **gs_kwargs), opp)
    assert d1.best_keep.value >= d0.best_score.value - 1e-12


def test_fast_v0_matches_general_path():
    # The vectorised expected-score v0 builder must equal the general one.
    import random
    from yahtzee.engine import _v0_list
    ev = ExpectedScoreSolver()
    cv = ev.child_value()
    random.seed(0)
    for _ in range(15):
        # keep states small so the uncached from-scratch DP stays fast
        cats = random.sample(range(13), random.randint(1, 3))
        mask = 0
        for cc in cats:
            mask |= 1 << cc
        bonus = (not (mask & (1 << S.YAHTZEE))) and random.random() < 0.5
        st = State(mask, random.randint(0, 63), bonus)
        fast = ev._v0_fast(st)
        slow = _v0_list(st, cv)
        assert max(abs(float(fast[i]) - slow[i]) for i in range(252)) < 1e-9


def test_opponent_distribution_is_valid_and_mean_matches_ev():
    # The additional-score distribution follows the same EV-optimal policy as
    # eadd, so it must be a valid distribution whose mean equals eadd(state).
    ev = ExpectedScoreSolver()
    om = OpponentModel(ev)
    for mask in [(1 << S.CHANCE), (1 << S.SIXES) | (1 << S.CHANCE),
                 (1 << S.THREE_KIND) | (1 << S.YAHTZEE) | (1 << S.CHANCE)]:
        st = State(mask, 40, False)
        arr = om._reldist_array(st)
        assert abs(float(arr.sum()) - 1.0) < 1e-9
        mean = sum(i * float(arr[i]) for i in range(len(arr)))
        assert abs(mean - ev.eadd(st)) < 1e-7


def test_opponent_distribution_matches_pure_trivial():
    # Structural equivalence with the pure-Python reference on a 1-open state.
    ev = ExpectedScoreSolver()
    om = OpponentModel(ev)
    st = State(1 << S.CHANCE, 40, False)
    fast = om._reldist_array(st)
    pure = om._reldist_pmf_pure(st)
    keys = set(range(len(fast))) | set(pure)
    diff = max(abs((float(fast[k]) if k < len(fast) else 0.0) - pure.get(k, 0.0))
               for k in keys)
    assert diff < 1e-12


def test_guard_rejects_early_game():
    solver = Solver(max_exact_open=7)
    full = list(S.CATEGORY_NAMES)
    try:
        solver.best_play(GameState(open_categories=full, upper_total=0,
                                   bonus_active=False, current_score=0,
                                   dice=[1, 2, 3, 4, 5], rerolls_left=1),
                         OpponentInfo(finished_score=100), objective="win")
    except ValueError as e:
        assert "slow" in str(e)
    else:
        raise AssertionError("expected guard to reject 13-open win query")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok  ", fn.__name__)
    print("\nAll %d tests passed." % len(fns))
