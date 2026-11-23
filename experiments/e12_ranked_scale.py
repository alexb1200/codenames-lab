"""Do ranked clues scale with board size? Same clue words and pool as e11. A clue's size is
the number of my cards ranked above every non-own card (same-model reading); K* is the
best. Coverage: share of my cards that are the top card of at least one clue word.
N up to 400, t = round(0.36 N), 100 boards per size. One record per (N, board)."""
import random

import numpy as np

from codenames_lab import lab
from experiments.e11_solvability import TAU, clue_vocab_and_pool

OUT = lab.results_name(__file__)

if __name__ == "__main__":
    lab.fresh(OUT)
    rng = np.random.default_rng(0)
    clues, big_pool = clue_vocab_and_pool(rng)
    sims_pool = lab.models()["w2v"].sims(clues, big_pool)
    col = {w: i for i, w in enumerate(big_pool)}
    for n in [25, 50, 100, 200, 400]:
        t = round(TAU * n)
        for seed in range(lab.n(100)):
            r = random.Random(seed * 7919 + n)
            words = r.sample(big_pool, n)
            mine = np.zeros(n, bool)
            mine[r.sample(range(n), t)] = True
            s = sims_pool[:, [col[w] for w in words]]
            k_star = int((s[:, mine] > s[:, ~mine].max(axis=1, keepdims=True)).sum(axis=1).max())
            covered = np.zeros(n, bool)
            covered[s.argmax(axis=1)] = True
            lab.write(OUT, {"experiment": "ranked_scale", "N": n, "t": t, "M": len(clues), "seed": seed,
                            "k_star": k_star, "covered_share": float(covered[mine].mean()),
                            "all_covered": bool(covered[mine].all())})
