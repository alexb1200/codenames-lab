"""Removing card hubness (HubCorrected: similarity minus the card's mean similarity to its 10
nearest clue words). Raw w2v against w2v-hub:
  concentration: share of clue words whose top card is the board's most popular card (100 boards)
  K*: best same-model clue size over all legal clue words (300 boards)
  transfer: one clue (SimSpymaster) read by a ppmi-cos SimGuesser (300 boards)
  games: RiskSpymaster + SoftGuesser at T 0.5 and 0.2, max risk 0.2, 150 seat-swapped games."""
import random
from collections import Counter

from codenames_lab import lab
from codenames_lab.players import SimGuesser, SimSpymaster, flat
from experiments.e09_vocab import clue_sizes

OUT = lab.results_name(__file__)

if __name__ == "__main__":
    lab.fresh(OUT)
    words_all = lab.clue_words()
    for name in ["w2v", "w2v-hub"]:
        m = lab.models()[name]
        spy = SimSpymaster(m, clue_words=words_all)
        for seed in range(lab.n(300)):
            rules = lab.Rules(lab.board(seed), lab.TEAMS)
            view = rules.view("red", spymaster=True)
            tokens, owners, *_ = flat(view)
            legal = [w for w in words_all if SimSpymaster.legal(w, tokens)]
            rec = {"experiment": "hub", "model": name, "seed": seed,
                   "k_star": int(clue_sizes(m, legal, tokens, owners).max())}
            if seed < lab.n(100):
                top = m.sims(legal, tokens).argmax(axis=1)
                rec["top_card_share"] = Counter(top.tolist()).most_common(1)[0][1] / len(legal)
            clue = spy.give_clue(view)
            owners_seq = lab.one_turn(rules, clue, SimGuesser(lab.models()["ppmi-cos"], random.Random(seed)))
            rec.update(clue=str(clue.token), count=clue.count, transfer_owners=owners_seq,
                       transfer_found=lab.run_of_own(owners_seq))
            lab.write(OUT, rec)
    for t in [0.5, 0.2]:
        cfg = lambda mm: {"model": mm, "spy": "risk", "T": t, "max_risk": 0.2, "guesser": "soft"}
        lab.play_match(OUT, "hub_games", f"T {t}: w2v-hub vs w2v", cfg("w2v-hub"), cfg("w2v"), lab.n(150), T=t)
