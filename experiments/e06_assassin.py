"""Assassin risk under a soft guesser, against the exact Plackett-Luce forms.
Random legal clue words (not spymaster picks), w2v. For each board, clue, temperature and
count, the hit rate is estimated from sampled guess orders (Gumbel trick, the sampler
RiskSpymaster uses). One record per (board, clue, T, count) with the measured rates and the
geometry: s = w_A / (w_A + sum of other non-own w), first = w_A / sum of all w, how many
non-own cards rank above the assassin, and two gaps (in units of T)."""
import numpy as np

from codenames_lab import lab
from codenames_lab.game import ASSASSIN, NEUTRAL
from codenames_lab.players import SimSpymaster, flat, zscores

OUT = lab.results_name(__file__)
TEMPS = [0.2, 0.5, 1.0]
COUNTS = [1, 2, 3, 5, 9]
SAMPLES = 4000

if __name__ == "__main__":
    lab.fresh(OUT)
    w2v = lab.models()["w2v"]
    rng = np.random.default_rng(0)
    for seed in range(lab.n(200)):
        rules = lab.Rules(lab.board(seed), lab.TEAMS)
        tokens, owners, *_ = flat(rules.view("red", spymaster=True))
        cat = np.array([0 if o == "red" else 3 if o == ASSASSIN else 1 if o == NEUTRAL else 2 for o in owners])
        legal = [w for w in lab.clue_words() if SimSpymaster.legal(w, tokens)]
        words = [legal[i] for i in rng.choice(len(legal), lab.n(20), replace=False)]
        z = zscores(w2v.sims(words, tokens))
        a = int(np.flatnonzero(cat == 3)[0])
        other_bad = (cat != 0) & (np.arange(len(tokens)) != a)
        for t in TEMPS:
            g = rng.gumbel(size=(SAMPLES, len(words), len(tokens)))
            ranked = cat[np.argsort(-(z[None] / t + g), axis=-1)]
            pos = (ranked != 0).argmax(-1)                                   # first non-own pick
            first_cat = np.take_along_axis(ranked, pos[..., None], -1)[..., 0]
            w = np.exp(z / t)
            for j, word in enumerate(words):
                own_z = np.sort(z[j, cat == 0])[::-1]
                base = {"experiment": "assassin", "seed": seed, "clue": word, "T": t,
                        "s": float(w[j, a] / (w[j, a] + w[j, other_bad].sum())),
                        "first": float(w[j, a] / w[j].sum()),
                        "assassin_first_miss": float((first_cat[:, j] == 3).mean()),
                        "above": int((z[j, cat != 0] > z[j, a]).sum()),
                        "gap_bad": float((z[j, a] - z[j, other_bad].max()) / t)}
                for k in COUNTS:
                    lab.write(OUT, {**base, "count": k,
                                    "hit": float(((first_cat[:, j] == 3) & (pos[:, j] < k)).mean()),
                                    "gap_own": float((own_z[min(k, len(own_z)) - 1] - z[j, a]) / t)})
