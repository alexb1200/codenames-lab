"""Can every one of my cards be reached by a clean clue? A real word graph against random controls.
A clue touches a card when their w2v cosine is at least the (1 - p) quantile of all
clue-card cosines over the board pool. Board: N cards, t = round(0.36 N) mine. Solvable =
every card of mine is touched by some clean clue (touches a card of mine and no bad card).
M = 4,000 random w2v clue words with no substring relation to any pool word; board words
from the 1,840 w2v-known concrete nouns. 100 boards per (N, p). Per board:
  real graph; clue-degree shuffle (each clue keeps its degree, cards redrawn);
  two-sided random graph (Chung-Lu: edge prob d_c k_i / E, both degree sequences kept);
  per-clue threshold (each clue touches its own top max(1, round(pN)) cards);
  predictions: equal density, per-clue binomial density, exact fixed degree.
One record per (N, p, board)."""
import random
from math import lgamma

import numpy as np

from codenames_lab import lab
from codenames_lab.words import nltk_pool

OUT = lab.results_name(__file__)
TAU, M = 0.36, 4000
DENSITIES = [0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.03, 0.06, 0.12, 0.2, 0.3, 0.4]


def log_comb(n, k):
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)


def solvable(adj, mine):
    clean = adj[:, mine].any(1) & ~adj[:, ~mine].any(1)
    reach = adj[clean][:, mine].any(0)
    return bool(reach.all()), reach


def clue_vocab_and_pool(rng):
    w2v = lab.models()["w2v"]
    big_pool = nltk_pool(top=5000, vocab=w2v.index)
    allowed = [w for w in w2v.clue_words if not any(w in b or b in w for b in big_pool)]
    clues = [allowed[i] for i in rng.choice(len(allowed), M, replace=False)]
    return clues, big_pool


if __name__ == "__main__":
    lab.fresh(OUT)
    rng = np.random.default_rng(0)
    clues, big_pool = clue_vocab_and_pool(rng)
    sims_pool = lab.models()["w2v"].sims(clues, big_pool)
    col = {w: i for i, w in enumerate(big_pool)}
    thresholds = {p: float(np.quantile(sims_pool, 1 - p)) for p in DENSITIES}
    for n in [25, 50, 100, 200]:
        t = round(TAU * n)
        q_of_degree = np.array([0.0] + [np.exp(log_comb(t - 1, d - 1) - log_comb(n, d)) if d <= t else 0.0
                                        for d in range(1, n + 1)])
        for p in DENSITIES:
            for seed in range(lab.n(100)):
                r = random.Random(seed * 7919 + n)
                words = r.sample(big_pool, n)
                mine = np.zeros(n, bool)
                mine[r.sample(range(n), t)] = True
                s = sims_pool[:, [col[w] for w in words]]
                adj = s >= thresholds[p]
                ok, reach = solvable(adj, mine)
                d, k = adj.sum(1), adj.sum(0)
                shuffled = np.argsort(rng.random(adj.shape), axis=1) < d[:, None]
                chung_lu = rng.random(adj.shape) < np.minimum(1.0, np.outer(d, k) / max(int(adj.sum()), 1))
                m = max(1, round(p * n))
                per_clue = np.zeros_like(adj)
                np.put_along_axis(per_clue, np.argsort(-s, axis=1)[:, :m], True, axis=1)
                pc = d / n
                lab.write(OUT, {
                    "experiment": "solvability", "N": n, "t": t, "p": p, "threshold": thresholds[p], "seed": seed,
                    "real": ok, "unreachable": int((~reach).sum()),
                    "deg_unreachable": k[mine][~reach].tolist(), "deg_reachable": k[mine][reach].tolist(),
                    "clue_shuffle": solvable(shuffled, mine)[0], "two_sided": solvable(chung_lu, mine)[0],
                    "per_clue_threshold": solvable(per_clue, mine)[0],
                    "pred_fixed_degree": float(np.exp(-t * np.exp(-q_of_degree[d].sum()))),
                    "pred_per_clue": float(np.exp(-t * np.exp(-(pc * (1 - pc) ** (n - t)).sum()))),
                    "pred_equal": float(np.exp(-t * np.exp(-M * p * (1 - p) ** (n - t))))})
