"""Adversarial edge-case tests for the win-probability decision logic."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yahtzee import scoring as S
from yahtzee.dice import (counts_to_dice, dice_to_counts, keep_options,
                          reroll_outcomes)
from yahtzee.solver import GameState, OpponentInfo, Solver

solver = Solver()  # exact: distribution opponent, win-prob up to 7 open
PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  ok   " if cond else "  FAIL ") + name + (("  -- " + detail) if detail else ""))


def brute_best_prob_1cat(cat, dice, S_now, T, tie=0.5):
    """Independent optimum P(beat T) for a 1-category endgame with one reroll."""
    counts = dice_to_counts(dice)
    reward = lambda tot: 1.0 if tot > T else (tie if tot == T else 0.0)
    best = -1.0
    for kept in keep_options(counts):
        p = 0.0
        for full, pr in reroll_outcomes(kept):
            p += pr * reward(S_now + S.score_placement(cat, full, False))
        best = max(best, p)
    return best


print("\n--- Guaranteed loss: should maximise expected score ---")
loss_gs0 = GameState(["Sixes", "Yahtzee", "Chance"], 20, False, 10, [6, 6, 6, 2, 1], 0)
dw = solver.best_play(loss_gs0, OpponentInfo(finished_score=1000), objective="win")
de = solver.best_play(loss_gs0, OpponentInfo(finished_score=1000), objective="ev")
check("guaranteed-loss score: all moves 0% win",
      all(abs(o.value) < 1e-12 for o in dw.score_options))
check("guaranteed-loss score: picks the expected-score-best box",
      dw.best_score.category == de.best_score.category,
      "win chose %s, ev chose %s" % (dw.best_score.name, de.best_score.name))

loss_gs1 = GameState(["Sixes", "Yahtzee", "Chance"], 20, False, 10, [6, 6, 6, 2, 1], 1)
dw1 = solver.best_play(loss_gs1, OpponentInfo(finished_score=1000), objective="win")
de1 = solver.best_play(loss_gs1, OpponentInfo(finished_score=1000), objective="ev")
check("guaranteed-loss reroll: keeps the expected-score-best dice",
      dw1.best_keep.keep == de1.best_keep.keep,
      "win keep %s, ev keep %s" % (dw1.best_keep.keep, de1.best_keep.keep))


print("\n--- Guaranteed win: should maximise expected score ---")
win_gs = GameState(["Sixes", "Yahtzee", "Chance"], 50, False, 300, [6, 6, 6, 2, 1], 1)
gw = solver.best_play(win_gs, OpponentInfo(finished_score=100), objective="win")
ge = solver.best_play(win_gs, OpponentInfo(finished_score=100), objective="ev")
check("guaranteed-win reroll: all 100% and keeps expected-score-best dice",
      abs(gw.best_keep.value - 1.0) < 1e-9 and gw.best_keep.keep == ge.best_keep.keep,
      "win keep %s, ev keep %s" % (gw.best_keep.keep, ge.best_keep.keep))


print("\n--- Endgame optimality vs independent brute force (Chance, 1 reroll) ---")
for (Snow, T, dice) in [(100, 118, [6, 5, 4, 2, 1]),
                        (100, 125, [6, 6, 3, 2, 1]),
                        (90, 110, [4, 4, 3, 2, 1])]:
    gs = GameState(["Chance"], 0, False, Snow, dice, 1)
    d = solver.best_play(gs, OpponentInfo(finished_score=T), objective="win")
    brute = brute_best_prob_1cat(S.CHANCE, dice, Snow, T)
    check("Chance endgame S=%d T=%d dice=%s: solver reaches the optimal win prob"
          % (Snow, T, dice), abs(d.best_keep.value - brute) < 1e-9,
          "solver %.4f vs brute %.4f" % (d.best_keep.value, brute))
    # is win-play different from raw expected-score play here?
    de = solver.best_play(gs, OpponentInfo(finished_score=T), objective="ev")
    if d.best_keep.keep != de.best_keep.keep:
        print("       (win keep %s differs from EV keep %s -- variance-aware)"
              % (d.best_keep.keep, de.best_keep.keep))


print("\n--- Tie value changes borderline play ---")
# You can guarantee exactly reaching T (a tie) or gamble to exceed it.
# tie=1 (ties count as a win) -> take the guaranteed tie; tie=0 -> must gamble.
# Chance-only, dice sum already = T-Snow so 'keep all' guarantees the tie.
Snow, T, dice = 100, 118, [6, 5, 4, 2, 1]  # sum 18 -> exactly T with keep-all
s_tie1 = Solver(tie_value=1.0)
s_tie0 = Solver(tie_value=0.0)
g = GameState(["Chance"], 0, False, Snow, dice, 1)
k1 = s_tie1.best_play(g, OpponentInfo(finished_score=T), objective="win").best_keep
k0 = s_tie0.best_play(g, OpponentInfo(finished_score=T), objective="win").best_keep
# tie=1: the tie is guaranteed-or-better -> 100% attainable (and it then maximises
# score among the 100% moves). tie=0: the tie is worthless, so it must gamble and
# cannot reach 100%.
check("tie=1 can guarantee at-least-a-tie (100%)",
      abs(k1.value - 1.0) < 1e-9, "kept %s win %.3f" % (k1.keep, k1.value))
check("tie=0 cannot guarantee a win -- must gamble (<100%)",
      k0.value < 1.0 - 1e-9, "kept %s win %.3f" % (k0.keep, k0.value))


print("\n--- Yahtzee bonus / joker forcing ---")
# (small open sets so the exact win-prob guard stays happy)
# Yahtzee box already filled with 50 (bonus active); roll five 4s; Fours open.
gj = GameState(["Fours", "Sixes", "Chance"], upper_total=0, bonus_active=True,
               current_score=0, dice=[4, 4, 4, 4, 4], rerolls_left=0)
dj = solver.best_play(gj, OpponentInfo(finished_score=1000), objective="win")
legal_names = [o.name for o in dj.score_options]
check("joker: five 4s with bonus active + Fours open -> forced to Fours",
      legal_names == ["Fours"], "legal: %s" % legal_names)
check("joker: forced Fours scores 20 + 100 bonus",
      dj.best_score.points == 120, "points %d" % dj.best_score.points)

# Yahtzee box filled with 0 (bonus NOT active): same roll -> no +100.
gj0 = GameState(["Fours", "Sixes", "Chance"], 0, False, 0, [4, 4, 4, 4, 4], 0)
dj0 = solver.best_play(gj0, OpponentInfo(finished_score=1000), objective="win")
check("no bonus when Yahtzee box holds 0: Fours scores only 20",
      dj0.best_score.points == 20, "points %d" % dj0.best_score.points)

# Matching upper (Fours) AND Yahtzee filled -> joker to any open lower box;
# Full House pays 25 (+100 bonus) even though five-of-a-kind isn't a full house.
gj2 = GameState(["Full House", "Small Straight", "Chance"], 0, True, 0,
                [4, 4, 4, 4, 4], 0)
dj2 = solver.best_play(gj2, OpponentInfo(finished_score=1000), objective="win")
fh = [o for o in dj2.score_options if o.name == "Full House"][0]
check("joker: Full House with five-of-a-kind pays 25 + 100 bonus",
      fh.points == 125, "points %d" % fh.points)


print("\n--- Upper bonus: play to cross 63 when it wins ---")
# Only Sixes open, at upper 48. Scoring three 6s -> +18 crosses 63 -> +35 = +53.
# Need exactly that bonus to win.
gub = GameState(["Sixes"], upper_total=48, bonus_active=False,
                current_score=150, dice=[6, 6, 6, 2, 1], rerolls_left=0)
dub = solver.best_play(gub, OpponentInfo(finished_score=200), objective="win")
check("upper bonus counted: Sixes here is worth 53 (18 + 35 bonus)",
      dub.best_score.points == 53, "points %d" % dub.best_score.points)
check("that reaches 150+53=203 > 200 -> guaranteed win",
      abs(dub.best_score.value - 1.0) < 1e-9, "win %.3f" % dub.best_score.value)


print("\n--- Structural sanity ---")
# Last category is forced (only one legal option).
last = GameState(["Chance"], 0, False, 100, [3, 3, 3, 1, 1], 0)
dl = solver.best_play(last, OpponentInfo(finished_score=90), objective="win")
check("single open category returns exactly one option",
      len(dl.score_options) == 1 and dl.best_score.name == "Chance")

# Reroll is never worse than committing to the best immediate score.
base = dict(open_categories=["Sixes", "Yahtzee", "Chance"], upper_total=40,
            bonus_active=False, current_score=150, dice=[6, 6, 3, 2, 1])
opp = OpponentInfo(finished_score=185)
d0 = solver.best_play(GameState(rerolls_left=0, **base), opp)
d1 = solver.best_play(GameState(rerolls_left=1, **base), opp)
check("reroll option >= best immediate score (keeping all is always allowed)",
      d1.best_keep.value >= d0.best_score.value - 1e-12,
      "reroll %.4f vs score-now %.4f" % (d1.best_keep.value, d0.best_score.value))

# Win probabilities are always in [0, 1].
allvals = [o.value for o in d1.keep_options] + [o.value for o in d0.score_options]
check("all win values within [0,1]", all(-1e-12 <= v <= 1 + 1e-12 for v in allvals))


print("\n================  %d passed, %d failed  ================"
      % (len(PASS), len(FAIL)))
if FAIL:
    print("FAILED:", ", ".join(FAIL))
    sys.exit(1)
