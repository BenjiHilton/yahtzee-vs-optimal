# Optimal 2-roll Yahtzee solver (2-player)

Given a Yahtzee game state — your open categories, your running score, the dice
on the table, whether you may reroll, and the opponent's situation — this engine
returns the **win-probability-optimal** move: which dice to keep, or which
category to score in.

This is the **1-reroll variant** (2 rolls per turn total), with standard
(Hasbro) scoring and the extra-Yahtzee joker rule.

## Rules implemented

| Category | Score |
|---|---|
| Ones … Sixes | sum of matching dice; **+35 bonus** when the upper total reaches **63** |
| Three / Four of a kind | sum of all five dice (needs ≥3 / ≥4 alike) |
| Full house | 25 |
| Small / Large straight | 30 / 40 |
| Yahtzee | 50 |
| Chance | sum of all five dice |

**Extra Yahtzee (joker):** once the Yahtzee box is filled with 50, every further
Yahtzee scores **+100**, and the roll is then placed by the joker rule — into the
matching upper box if open; otherwise any open lower box (where Full House /
Small / Large straight pay their full 25 / 30 / 40 even though five-of-a-kind
doesn't "really" form them); otherwise a forced 0 in an open upper box. (If the
Yahtzee box was filled with **0**, no +100 bonus applies.)

## What "optimal" means here

The engine maximises **your probability of winning**, not just your expected
score. Concretely it plays to maximise `P(your final total beats the
opponent's)`, which is what actually wins a 2-player game — e.g. it will lock a
guaranteed win, or gamble for variance when behind, where an expected-score
maximiser would not.

Two opponent cases:

* **Opponent has finished** → their score is a known target. Play is then
  **exactly optimal**: maximise `P(final total > target)`.
* **Opponent still playing** → we model them as playing the expected-score-optimal
  strategy and compute the exact distribution of their final total. This is the
  single modelling assumption in the engine. (Fully solving the game where *both*
  players continuously adapt to the score gap is computationally intractable; a
  strong fixed opponent policy is the standard, principled substitute.)

Ties are configurable (`--tie`): `0.5` counts a draw as half a win (default),
`1.0` gives ties to you, `0.0` to the opponent.

### What's precomputed, and what isn't

* **Expected-score play is a solved lookup table.** Its value-to-go depends only
  on the scorecard `(open categories, upper total, Yahtzee-bonus flag)` —
  **591,360 reachable states**, one number each. `precompute_ev.py` builds the
  exact table in about **3 minutes** and saves it to `ev_table.pkl` (~12 MB);
  the solver auto-loads it, after which every expected-score decision is an
  instant lookup. (Headline number: optimal expected final score from an empty
  card is **190.13** with one reroll, versus ~254 for standard 3-roll Yahtzee.)
  Without the file, states still solve lazily — instantly near the end, slowly
  from early positions.

* **Win-probability is not a single static table**, because the best move also
  depends on your running score relative to the target (and, against a live
  opponent, on their state too). That extra dimension multiplies the state space
  by orders of magnitude — inherent, not an implementation gap. So win-probability
  is solved on demand:
  * **Opponent finished** (known target): exact, and fast for the endgame — the
    regime where win-probability actually differs from expected-score play.
  * **Opponent still playing**: their exact final-score distribution is computed
    with a vectorised DP (sub-second up to ~5 open categories, a few seconds at
    6). Beyond `max_opp_dist_open` (default 6) open categories the opponent is
    approximated by their expected final total (fast, mean only).

  A guard refuses exact win-probability queries when **you** have more than
  `max_exact_open` (default 7) categories open — that far out, use
  `--objective ev` (instant, near-optimal for win rate with many turns to go) or
  pass `--allow-slow`.

## Usage

### Command line

```bash
# Endgame, must score now, opponent already finished on 205:
python -m yahtzee --open Sixes,Yahtzee,Chance --upper 50 --score 178 \
    --dice 6,6,6,2,1 --rerolls 0 --opp-final 205

# Reroll available, opponent still playing:
python -m yahtzee --open Fours,Full-House,Small-Straight,Yahtzee \
    --upper 40 --score 120 --dice 3,3,3,4,4 --rerolls 1 \
    --opp-open Sixes,Chance,Yahtzee --opp-upper 55 --opp-current 130

# Maximise your own expected score instead (no opponent needed, any game length):
python -m yahtzee --open Threes,Fours,Fives,Sixes,Yahtzee --upper 20 \
    --score 90 --dice 5,5,5,2,1 --rerolls 1 --objective ev
```

Category names are case-insensitive; spaces may be written with `-`
(`Full-House`) and common aliases work (`3kind`, `4kind`, `sm`, `lg`, `fh`).
`--rerolls 1` means the first roll is on the table and you may reroll once;
`--rerolls 0` means you must score now.

### Python API

```python
from yahtzee.solver import Solver, GameState, OpponentInfo

solver = Solver(tie_value=0.5)
decision = solver.best_play(
    GameState(open_categories=["Sixes", "Yahtzee", "Chance"],
              upper_total=50, bonus_active=False, current_score=178,
              dice=[6, 6, 6, 2, 1], rerolls_left=0),
    opponent=OpponentInfo(finished_score=205))

print(decision.describe())
print(decision.best_score.name, decision.best_score.value)  # 'Sixes' 1.0
```

For a reroll decision, `decision.best_keep` gives the dice to hold
(`.keep` / `.reroll`) and its win probability (`.value`).

## How it works

* `dice.py` — the 252 distinct rolls, reroll transition probabilities, keep
  enumeration, and index tables for the hot path.
* `scoring.py` — all categories, the upper bonus, and the joker/+100 rules.
* `state.py` — a scorecard as `(open-categories bitmask, upper total capped at
  63, Yahtzee-bonus flag)` plus the legal-placement enumerator.
* `engine.py` — the turn DP (optimise keep, then category) and two backward
  inductions: `ExpectedScoreSolver` (exact expected score) and
  `WinProbabilitySolver` (maximise `E[reward(final total)]`, with an exact
  monotone-saturation prune).
* `fastcore.py` — NumPy vectorisation of the per-turn keep-expectation.
* `opponent.py` — the opponent's final-score distribution and the "beat them"
  reward.
* `solver.py` / `cli.py` — the top-level API and CLI.

## Tests

```bash
python tests/test_engine.py        # or: python -m pytest -q
```

The expected-score DP is cross-checked against closed-form values (Chance =
21.25, Ones = 55/36) and a brute-force Yahtzee computation; the joker and
upper-bonus rules and the win-probability edge cases are covered directly.

## Requirements

Python 3.7+ and NumPy (used to accelerate the inner loop; the engine falls back
to pure Python if NumPy is absent).
```
