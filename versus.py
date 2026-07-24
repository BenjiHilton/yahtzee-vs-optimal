"""Play a full game of Yahtzee (1 reroll) against the optimal algorithm.

    python versus.py

You and the algorithm ("Optimal") alternate turns.  You roll, choose what to
keep, reroll once, and pick a box; then the algorithm takes its turn, playing to
maximise its probability of beating you.  Both scorecards are shown throughout.

The algorithm plays true win-probability in the endgame (the last few turns,
where the score gap changes the right move) and expected-score-optimal earlier
(where the two agree and exact win-probability is intractable).
"""

import random
import sys
from collections import Counter

from yahtzee import scoring as S
from yahtzee.dice import dice_to_counts
from yahtzee.solver import GameState, OpponentInfo, Solver, category_index
from yahtzee.state import State, categories_open, enumerate_placements

FULL_MASK = (1 << S.NUM_CATEGORIES) - 1


# ----- a player's scorecard -------------------------------------------------

class Player:
    def __init__(self, name):
        self.name = name
        self.filled = {}          # category index -> points shown in that box
        self.upper_total = 0      # capped at 63
        self.bonus_active = False
        self.yz_bonus = 0         # accumulated +100 extra-Yahtzee bonuses
        self.upper_bonus_earned = False
        self.total = 0

    def state(self):
        mask = FULL_MASK
        for c in self.filled:
            mask &= ~(1 << c)
        return State(mask, self.upper_total, self.bonus_active)

    def open_cats(self):
        return categories_open(self.state())

    def done(self):
        return len(self.filled) == S.NUM_CATEGORIES

    def apply(self, cat, counts):
        pts = S.score_placement(cat, counts, self.bonus_active)
        new_upper, upper_bonus = S.upper_delta(cat, counts, self.upper_total)
        yz = S.YAHTZEE_BONUS if (S.is_yahtzee(counts) and self.bonus_active) else 0
        self.filled[cat] = pts - yz
        self.yz_bonus += yz
        self.total += pts + upper_bonus
        self.upper_total = new_upper
        if upper_bonus:
            self.upper_bonus_earned = True
        if cat == S.YAHTZEE and S.is_yahtzee(counts):
            self.bonus_active = True
        return pts + upper_bonus


# ----- display --------------------------------------------------------------

def _cell(v):
    return "%4d" % v if v is not None else "   ."


def show_boards(you, ai):
    print("\n  %-17s %6s %6s" % ("", "YOU", "OPTIMAL"))
    print("  " + "-" * 33)
    for c in range(S.NUM_CATEGORIES):
        if c == S.THREE_KIND:
            print("  %-17s %6s %6s" % ("--- upper bonus",
                  ("  35" if you.upper_bonus_earned else "%2d/63" % you.upper_total),
                  ("  35" if ai.upper_bonus_earned else "%2d/63" % ai.upper_total)))
        print("  %-17s %6s %6s" % (S.CATEGORY_NAMES[c],
                                   _cell(you.filled.get(c)), _cell(ai.filled.get(c))))
    print("  %-17s %6s %6s" % ("Yahtzee bonus", _cell(you.yz_bonus or None),
                               _cell(ai.yz_bonus or None)))
    print("  " + "-" * 33)
    print("  %-17s %6d %6d" % ("TOTAL", you.total, ai.total))


# ----- helpers --------------------------------------------------------------

def roll(n):
    return sorted(random.randint(1, 6) for _ in range(n))


def parse_dice(text):
    vals = [int(t) for t in text.replace(",", " ").split()]
    for v in vals:
        if not 1 <= v <= 6:
            raise ValueError("dice must be 1..6")
    return vals


def ask(prompt):
    try:
        return input(prompt).strip()
    except EOFError:
        print()
        sys.exit(0)


def opp_info(player):
    """OpponentInfo describing `player` from the other side's perspective."""
    if player.done():
        return OpponentInfo(finished_score=player.total)
    return OpponentInfo(
        open_categories=[S.CATEGORY_NAMES[c] for c in player.open_cats()],
        upper_total=player.upper_total, bonus_active=player.bonus_active,
        current_score=player.total)


def advise(solver, me, opponent, dice, rerolls_left):
    gs = GameState(open_categories=[S.CATEGORY_NAMES[c] for c in me.open_cats()],
                   upper_total=me.upper_total, bonus_active=me.bonus_active,
                   current_score=me.total, dice=dice, rerolls_left=rerolls_left)
    try:
        return solver.best_play(gs, opp_info(opponent), objective="win"), "win"
    except ValueError:
        return solver.best_play(gs, opp_info(opponent), objective="ev"), "ev"


def is_submultiset(keep, dice):
    kc, dc = Counter(keep), Counter(dice)
    return all(kc[f] <= dc[f] for f in kc)


# ----- turns ----------------------------------------------------------------

def human_turn(solver, you, ai, auto, hints):
    if auto:
        dice = roll(5)
    else:
        while True:
            try:
                dice = parse_dice(ask("  Enter your roll (5 dice): "))
                if len(dice) != 5:
                    raise ValueError("need exactly 5 dice")
                break
            except ValueError as e:
                print("   ! %s" % e)
    print("\n  YOUR TURN — you rolled:  %s" % " ".join(map(str, dice)))

    rec_keep = None
    if hints:
        dec, used = advise(solver, you, ai, dice, 1)
        rec_keep = dec.best_keep.keep
        print("   hint: keep [%s]" % " ".join(map(str, rec_keep)))

    # choose dice to keep
    while True:
        msg = "   Keep which dice? (faces, 'all', 'none'"
        msg += ", Enter=hint): " if hints else ", Enter=all): "
        ans = ask(msg).lower()
        if ans == "" and hints:
            keep = list(rec_keep)
            break
        if ans in ("", "all"):
            keep = list(dice)
            break
        if ans == "none":
            keep = []
            break
        if ans == "hint":
            dec, _ = advise(solver, you, ai, dice, 1)
            print("   hint: keep [%s]" % " ".join(map(str, dec.best_keep.keep)))
            continue
        try:
            keep = parse_dice(ans)
        except ValueError as e:
            print("   ! %s" % e)
            continue
        if is_submultiset(keep, dice):
            break
        print("   ! you can only keep dice you rolled (%s)" % " ".join(map(str, dice)))

    n = 5 - len(keep)
    if n == 0:
        final = sorted(dice)
    elif auto:
        new = roll(n)
        final = sorted(list(keep) + new)
        print("   rerolled %d -> %s" % (n, " ".join(map(str, new))))
    else:
        while True:
            try:
                final = parse_dice(ask("   Enter dice after rerolling %d: " % n))
                if len(final) != 5:
                    raise ValueError("need exactly 5 dice")
                break
            except ValueError as e:
                print("   ! %s" % e)
    print("   final dice:  %s" % " ".join(map(str, final)))

    counts = dice_to_counts(final)
    legal = set(S.legal_categories(you.state().open_mask, counts, you.bonus_active))
    rec_cat = None
    if hints:
        dec, used = advise(solver, you, ai, final, 0)
        rec_cat = dec.best_score.category
        print("   hint: score in %s (+%d)" %
              (S.CATEGORY_NAMES[rec_cat], dec.best_score.points))

    while True:
        prompt = "   Score in which box?"
        prompt += " (Enter=hint): " if hints else " (%s): " % ", ".join(
            S.CATEGORY_NAMES[c] for c in sorted(legal))
        ans = ask(prompt)
        if ans == "" and hints:
            cat = rec_cat
            break
        if ans == "hint":
            dec, _ = advise(solver, you, ai, final, 0)
            print("   hint: %s (+%d)" %
                  (S.CATEGORY_NAMES[dec.best_score.category], dec.best_score.points))
            continue
        try:
            cat = category_index(ans)
        except ValueError:
            print("   ! unknown box")
            continue
        if cat not in legal:
            print("   ! not legal here. Legal: %s" %
                  ", ".join(S.CATEGORY_NAMES[c] for c in sorted(legal)))
            continue
        break

    gain = you.apply(cat, counts)
    print("   >> you scored %s for %d." % (S.CATEGORY_NAMES[cat], gain))


def ai_turn(solver, ai, you):
    dice = roll(5)
    dec, used = advise(solver, ai, you, dice, 1)
    keep = dec.best_keep.keep
    n = 5 - len(keep)
    final = sorted(keep + roll(n)) if n else sorted(dice)

    dec2, used2 = advise(solver, ai, you, final, 0)
    cat = dec2.best_score.category
    counts = dice_to_counts(final)
    gain = ai.apply(cat, counts)

    print("\n  OPTIMAL'S TURN — rolled %s, kept [%s], final %s" % (
        " ".join(map(str, dice)), " ".join(map(str, keep)), " ".join(map(str, final))))
    tag = ("est. win %.0f%%" % (100 * dec2.best_score.value)) if used2 == "win" \
        else "playing for score"
    print("   >> Optimal scored %s for %d.  (%s)" %
          (S.CATEGORY_NAMES[cat], gain, tag))


# ----- game -----------------------------------------------------------------

def main():
    print("=" * 50)
    print("  YOU  vs  OPTIMAL   —   Yahtzee (1 reroll)")
    print("=" * 50)
    auto = not ask("Roll your dice automatically? (a/m) [a]: ").lower().startswith("m")
    first = not ask("Do you go first or second? (f/s) [f]: ").lower().startswith("s")
    hints = ask("Show optimal-move hints on your turns? (y/n) [n]: ").lower().startswith("y")

    print("\nLoading optimal solver...")
    # Exact win-probability once a side has <=4 boxes open (expected-score before
    # that); a still-playing opponent is modelled by their expected final total
    # so each AI turn stays responsive (~1-2s). Play vs a finished opponent -- the
    # decisive last turn -- remains exact.
    solver = Solver(max_exact_open=4, max_opp_dist_open=0)

    you, ai = Player("You"), Player("Optimal")
    order = [(you, "human"), (ai, "ai")] if first else [(ai, "ai"), (you, "human")]

    for rnd in range(1, S.NUM_CATEGORIES + 1):
        print("\n" + "=" * 50 + "\n  ROUND %d of 13" % rnd)
        for player, kind in order:
            show_boards(you, ai)
            if kind == "human":
                human_turn(solver, you, ai, auto, hints)
            else:
                ai_turn(solver, ai, you)

    print("\n" + "=" * 50)
    show_boards(you, ai)
    print("=" * 50)
    if you.total > ai.total:
        print("  YOU WIN  %d - %d !" % (you.total, ai.total))
    elif you.total < ai.total:
        print("  OPTIMAL WINS  %d - %d." % (ai.total, you.total))
    else:
        print("  TIE  %d - %d." % (you.total, ai.total))


if __name__ == "__main__":
    main()
