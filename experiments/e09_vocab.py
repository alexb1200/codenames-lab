"""Vocabulary scaling. Part 1, shared model: a clue's size is the number of my cards above
every non-own card; K* is the best over a random subset of M legal clue words. 20 subsets per
board and size, 300 boards; the full vocabulary once per board. One record per (model, M,
board) with the list of K* values. Part 2, noisy guesser: RiskSpymaster at T 0.5, max risk
0.2, first clue from a random vocabulary of size M; the expected own cards it plans for
(re-estimated with 4,000 samples). w2v, 100 boards; one record per (M, board)."""
import numpy as np

from codenames_lab import lab
from codenames_lab.game import ASSASSIN, NEUTRAL
from codenames_lab.players import RiskSpymaster, SimSpymaster, flat

OUT = lab.results_name(__file__)
SIZES = [100, 300, 1000, 3000, 10000]


def clue_sizes(model, words, tokens, owners, team="red"):
    sims = model.sims(words, tokens)
    mine = np.array([o == team for o in owners])
    return (sims[:, mine] > sims[:, ~mine].max(axis=1, keepdims=True)).sum(axis=1)


if __name__ == "__main__":
    lab.fresh(OUT)
    rng = np.random.default_rng(0)
    for name in ["w2v", "ppmi", "ppmi-cos"]:
        model = lab.models()[name]
        vocab = model.clue_words
        sizes = [m for m in SIZES if m <= len(vocab) // 2]
        for seed in range(lab.n(300)):
            rules = lab.Rules(lab.board(seed), lab.TEAMS)
            tokens, owners, *_ = flat(rules.view("red", spymaster=True))
            legal = [w for w in vocab if SimSpymaster.legal(w, tokens)]
            k = clue_sizes(model, legal, tokens, owners)
            for m in sizes:
                lab.write(OUT, {"experiment": "vocab_shared", "model": name, "M": m, "seed": seed,
                                "k_star": [int(k[rng.choice(len(legal), m, replace=False)].max())
                                           for _ in range(lab.n(20))]})
            lab.write(OUT, {"experiment": "vocab_shared", "model": name, "M": len(vocab), "full": True,
                            "seed": seed, "k_star": [int(k.max())]})
    w2v = lab.models()["w2v"]
    for m in SIZES + [len(w2v.clue_words)]:
        for seed in range(lab.n(100)):
            rules = lab.Rules(lab.board(seed), lab.TEAMS)
            sub = w2v.clue_words if m >= len(w2v.clue_words) else \
                [w2v.clue_words[i] for i in rng.choice(len(w2v.clue_words), m, replace=False)]
            spy = RiskSpymaster(w2v, 0.5, 0.2, clue_words=sub, rng=np.random.default_rng(seed))
            view = rules.view("red", spymaster=True)
            clue = spy.give_clue(view)
            tokens, owners, *_ = flat(view)
            cat = np.array([0 if o == "red" else 3 if o == ASSASSIN else 1 if o == NEUTRAL else 2 for o in owners])
            found, risk = spy.estimate([str(clue.token)], tokens, list(range(len(tokens))), cat, np.arange(1, 10), 4000)
            lab.write(OUT, {"experiment": "vocab_noisy", "model": "w2v", "M": m, "seed": seed,
                            "clue": str(clue.token), "count": clue.count,
                            "expected_found": float(found[0, clue.count - 1]), "risk": float(risk[0, clue.count - 1])})
