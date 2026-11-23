"""Exact facts the experiments rely on, checked on a small synthetic word model."""
import random
import unittest

import numpy as np

from codenames_lab.game import ASSASSIN, NEUTRAL, Board, Rules
from codenames_lab.models import Model
from codenames_lab.players import RiskSpymaster, SoftGuesser, flat, zscores

WORDS = [f"w{i}" for i in range(60)]
CLUES = [f"c{i}" for i in range(40)]


class RandomVectors(Model):
    """Random unit vectors for every word: a stand-in model with no downloads."""
    name = "random-vectors"

    def __init__(self, dim: int = 16, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        words = WORDS + CLUES
        m = rng.normal(size=(len(words), dim))
        self.matrix = m / np.linalg.norm(m, axis=1, keepdims=True)
        self.index = {w: i for i, w in enumerate(words)}
        self.clue_words = CLUES

    def knows(self, word: str) -> bool:
        return word in self.index

    def sims(self, clues, words):
        left = self.matrix[[self.index[c] for c in (clues or self.clue_words)]]
        return left @ self.matrix[[self.index[w] for w in words]].T


class TestPlackettLuce(unittest.TestCase):
    """For picks in softmax order with weights w = exp(z/T):
    P(first pick is the assassin) = w_A / sum(w), and
    P(assassin is the first non-own pick) = w_A / (w_A + sum of other non-own w)."""

    def test_identities(self):
        rng = np.random.default_rng(5)
        cat = np.array([0] * 9 + [2] * 8 + [1] * 7 + [3])
        for _ in range(5):
            z, t = rng.normal(size=25), rng.choice([0.2, 0.5, 1.0])
            ranked = cat[np.argsort(-(z / t + rng.gumbel(size=(200_000, 25))), axis=-1)]
            pos = (ranked != 0).argmax(-1)
            first_miss = ranked[np.arange(len(ranked)), pos]
            w = np.exp(z / t)
            self.assertAlmostEqual((first_miss == 3).mean(), w[24] / w[cat != 0].sum(), delta=0.005)
            self.assertAlmostEqual(((first_miss == 3) & (pos == 0)).mean(), w[24] / w.sum(), delta=0.005)


class TestRiskEstimate(unittest.TestCase):
    def test_estimate_matches_soft_guesser(self):
        """RiskSpymaster's sampled prediction should match a real SoftGuesser at the same T."""
        model = RandomVectors()
        for t in (0.3, 1.0):
            rules0 = Rules(Board(5, 5, ["red", "blue"], WORDS, random.Random(2)), ["red", "blue"])
            view = rules0.view("red", spymaster=True)
            spy = RiskSpymaster(model, t, 0.3, rng=np.random.default_rng(0))
            clue = spy.give_clue(view)
            tokens, owners, *_ = flat(view)
            cat = np.array([0 if o == "red" else 3 if o == ASSASSIN else 1 if o == NEUTRAL else 2 for o in owners])
            found, risk = spy.estimate([str(clue.token)], tokens, list(range(25)), cat, np.arange(1, 10), 20_000)
            sim_found, sim_risk = [], []
            for trial in range(3000):
                rules = Rules(Board(5, 5, ["red", "blue"], WORDS, random.Random(2)), ["red", "blue"])
                rules.give_clue(clue)
                guesser = SoftGuesser(model, t, 0.0, random.Random(trial))
                seq = []
                while not rules.done and rules.current == "red" and rules.clue is not None:
                    pos = guesser.guess(rules.view("red", spymaster=False), [])
                    if pos is not None:
                        seq.append(rules.board.grid[pos[0]][pos[1]].owner)
                    rules.guess(pos)
                run = next((i for i, o in enumerate(seq) if o != "red"), len(seq))
                sim_found.append(run)
                sim_risk.append(any(o not in ("red", NEUTRAL) for o in seq))
            self.assertAlmostEqual(np.mean(sim_found), found[0, clue.count - 1], delta=0.08)
            self.assertAlmostEqual(np.mean(sim_risk), risk[0, clue.count - 1], delta=0.04)

    def test_zscores(self):
        z = zscores(np.array([[1.0, 2.0, 3.0]]))
        self.assertAlmostEqual(float(z.mean()), 0.0)
        self.assertAlmostEqual(float(z.std()), 1.0)


if __name__ == "__main__":
    unittest.main()
