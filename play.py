"""Interactive Yahtzee — play a game with the optimal-move advisor.

    python play.py

Two modes:
  * auto   — the program rolls the dice for you; you choose what to keep and
             where to score (the advisor shows the optimal move at each step).
  * manual — you type the dice you rolled on real dice (use it as a live advisor
             while playing a physical game).

At any prompt just press Enter to accept the advisor's recommendation.
"""

import random
import sys

from yahtzee import scoring as S
from yahtzee.dice import counts_to_dice, dice_to_counts
from yahtzee.solver import GameState, OpponentInfo, Solver, category_index
from yahtzee.state import (State, categories_open, enumerate_placements,
                           is_terminal, new_game_state)


# ----- small input helpers --------------------------------------------------

def ask(prompt):
    try:
        return input(prompt).strip()
    except EOFError:
        print()
        sys.exit(0)


def parse_dice(text):
    vals = [int(t) for t in text.replace(",", " ").split()]
    for v in vals:
        if not 1 <= v <= 6:
            raise ValueError("dice must be 1..6")
    return vals


def roll(n):
    return sorted(random.randint(1, 6) for _ in range(n))


# ----- display --------------------------------------------------------------

def show_card(state, total):
    open_cats = set(categories_open(state))
    print("\n  Scorecard   (total so far: %d, upper: %d%s)" %
          (total, state.upper_total,
           ", Yahtzee bonus active" if state.bonus_active else ""))
    line = []
    for c in range(S.NUM_CATEGORIES):
        tag = S.CATEGORY_NAMES[c]
        mark = "open" if c in open_cats else " -- "
        line.append("   [%-15s] %s" % (tag, mark))
        if len(line) == 2:
            print("".join(line))
            line = []
    if line:
        print("".join(line))


def advise(solver, state, total, dice, rerolls_left, opponent, objective):
    """Return a Decision, falling back to expected-score when exact
    win-probability is intractable this early."""
    gs = GameState(open_categories=[S.CATEGORY_NAMES[c] for c in categories_open(state)],
                   upper_total=state.upper_total, bonus_active=state.bonus_active,
                   current_score=total, dice=dice, rerolls_left=rerolls_left)
    obj = objective
    try:
        return solver.best_play(gs, opponent, objective=obj), obj
    except ValueError:
        # too many categories open for exact win-prob -> use expected score
        return solver.best_play(gs, opponent, objective="ev"), "ev"


# ----- one turn -------------------------------------------------------------

def play_turn(solver, state, total, opponent, objective, auto):
    # --- first roll ---
    if auto:
        dice = roll(5)
        print("\n  You rolled:  %s" % " ".join(map(str, dice)))
    else:
        while True:
            try:
                dice = parse_dice(ask("\n  Enter your roll (5 dice): "))
                if len(dice) != 5:
                    raise ValueError("need exactly 5 dice")
                break
            except ValueError as e:
                print("   ! %s" % e)

    # --- keep / reroll advice ---
    decision, used = advise(solver, state, total, dice, 1, opponent, objective)
    rec = decision.best_keep
    print("   Advisor (%s): keep [%s], reroll [%s]   %s" % (
        "win%" if used == "win" else "exp.score",
        " ".join(map(str, rec.keep)) or "-", " ".join(map(str, rec.reroll)) or "-",
        _val(rec.value, used)))

    # --- decide what to keep, then reroll ---
    keep = _choose_keep(dice, rec.keep, auto)
    n_reroll = 5 - len(keep)
    if n_reroll == 0:
        final = sorted(dice)
    else:
        if auto:
            new = roll(n_reroll)
            print("   Rerolled %d: %s" % (n_reroll, " ".join(map(str, new))))
            final = sorted(list(keep) + new)
        else:
            while True:
                try:
                    final = parse_dice(ask("   Enter dice after rerolling %d: " % n_reroll))
                    if len(final) != 5:
                        raise ValueError("need exactly 5 dice")
                    break
                except ValueError as e:
                    print("   ! %s" % e)
    print("   Final dice:  %s" % " ".join(map(str, final)))

    # --- category advice ---
    decision, used = advise(solver, state, total, final, 0, opponent, objective)
    print("   Best box (%s): %s  (+%d, %s)" % (
        "win%" if used == "win" else "exp.score",
        decision.best_score.name, decision.best_score.points,
        _val(decision.best_score.value, used)))
    print("     next best: " + "; ".join(
        "%s +%d" % (o.name, o.points) for o in decision.score_options[1:4]))

    # --- choose the box ---
    cat = _choose_category(state, final, decision.best_score.category, auto)

    # --- apply it ---
    counts = dice_to_counts(final)
    for p in enumerate_placements(state, counts):
        if p.category == cat:
            print("   -> scored %s for %d points." % (S.CATEGORY_NAMES[cat], p.gain))
            return p.child, total + p.gain
    raise AssertionError("chosen category was not legal")


def _choose_keep(dice, recommended, auto):
    if auto:
        while True:
            ans = ask("   Keep which dice? (Enter = accept, faces e.g. '6 6 3'): ")
            if ans == "":
                return list(recommended)
            try:
                keep = parse_dice(ans)
            except ValueError as e:
                print("   ! %s" % e)
                continue
            if _is_submultiset(keep, dice):
                return keep
            print("   ! you can only keep dice you actually rolled (%s)" %
                  " ".join(map(str, dice)))
    return list(recommended)  # manual mode: you physically rerolled already


def _choose_category(state, final, recommended, auto):
    legal = set(S.legal_categories(state.open_mask, dice_to_counts(final),
                                   state.bonus_active))
    while True:
        ans = ask("   Score in which box? (Enter = accept '%s'): " %
                  S.CATEGORY_NAMES[recommended])
        if ans == "":
            return recommended
        try:
            cat = category_index(ans)
        except ValueError:
            print("   ! unknown box name")
            continue
        if cat not in legal:
            print("   ! not a legal box here. Legal: %s" %
                  ", ".join(S.CATEGORY_NAMES[c] for c in sorted(legal)))
            continue
        return cat


def _is_submultiset(keep, dice):
    from collections import Counter
    kc, dc = Counter(keep), Counter(dice)
    return all(kc[f] <= dc[f] for f in kc)


def _val(v, used):
    return ("win %.1f%%" % (100 * v)) if used == "win" else ("exp %.1f" % v)


# ----- game -----------------------------------------------------------------

def main():
    print("=" * 60)
    print(" Yahtzee (1 reroll) — optimal-move advisor")
    print("=" * 60)
    mode = ask("Roll dice automatically or enter them manually? (a/m) [a]: ").lower()
    auto = not mode.startswith("m")

    obj = ask("Play for (w)in probability or (e)xpected score? (w/e) [e]: ").lower()
    objective = "win" if obj.startswith("w") else "ev"

    opponent = OpponentInfo(finished_score=0)
    if objective == "win":
        t = ask("Opponent's final score to beat (blank = expected-score play): ").strip()
        if t == "":
            objective = "ev"
        else:
            opponent = OpponentInfo(finished_score=int(t))

    print("\nLoading solver...")
    solver = Solver()

    state = new_game_state()
    total = 0
    turn = 0
    while not is_terminal(state):
        turn += 1
        print("\n" + "-" * 60 + "\n  TURN %d of 13" % turn)
        show_card(state, total)
        state, total = play_turn(solver, state, total, opponent, objective, auto)

    print("\n" + "=" * 60)
    print("  GAME OVER — final score: %d" % total)
    if objective == "win" and opponent.finished_score is not None:
        verdict = ("WIN" if total > opponent.finished_score else
                   "TIE" if total == opponent.finished_score else "LOSS")
        print("  Opponent had %d  ->  %s" % (opponent.finished_score, verdict))
    print("=" * 60)


if __name__ == "__main__":
    main()
