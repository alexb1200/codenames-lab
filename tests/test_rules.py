"""Rules invariants, using random players (no word models needed)."""
import random
import unittest

from codenames_lab.game import ASSASSIN, Board, Clue, Game, RandomPlayer, Rules, Team

WORDS = [f"w{i}" for i in range(100)]


def random_game(teams: list[str], seed: int, size: int = 6) -> Rules:
    rng = random.Random(seed)
    game = Game([Team(t, RandomPlayer(rng), [RandomPlayer(rng)]) for t in teams])
    return game.run(Board(size, size, teams, WORDS, rng))


class TestRules(unittest.TestCase):
    def test_card_counts(self):
        board = Board(5, 5, ["red", "blue"], WORDS, random.Random(0))
        owners = [cell.owner for _, cell in board.cells()]
        self.assertEqual([owners.count(o) for o in ["red", "blue", "NEUTRAL", ASSASSIN]], [9, 8, 7, 1])

    def test_board_too_small(self):
        with self.assertRaises(AssertionError):
            Board(2, 2, ["red", "blue"], WORDS, random.Random(0), assassins=3)

    def test_games_end_and_stay_consistent(self):
        for teams in (["red", "blue"], ["red", "blue", "green"], ["red", "blue", "green", "gold"]):
            for seed in range(200):
                r = random_game(teams, seed)
                guessed = [e.pos for e in r.history if e.kind == "guess"]
                self.assertEqual(len(guessed), len(set(guessed)), "a cell was revealed twice")
                self.assertEqual(len(guessed), sum(c.revealed for _, c in r.board.cells()))
                if r.reason == "found":
                    self.assertEqual(r.board.remaining(r.winner), 0)

    def test_turn_after_elimination(self):
        teams = ["red", "blue", "green"]
        r = Rules(Board(5, 5, teams, WORDS, random.Random(1)), teams)
        r.turn = 2                                   # green to play, last in order
        assassin = next(p for p, c in r.board.cells() if c.owner == ASSASSIN)
        r.give_clue(Clue("x", 1))
        r.guess(assassin)
        self.assertEqual(r.alive, ["red", "blue"])
        self.assertEqual(r.current, "red")           # wraps to the first team

    def test_clue_checks(self):
        r = Rules(Board(5, 5, ["red", "blue"], WORDS, random.Random(0)), ["red", "blue"])
        word = str(r.board.grid[0][0].token)
        for bad in (Clue(word, 1), Clue(word.upper(), 1), Clue("two words", 1), Clue("", 1), Clue("ok", -1)):
            self.assertIsNotNone(r.check_clue(bad))
        self.assertIsNone(r.check_clue(Clue("ok", 2)))
        with self.assertRaises(ValueError):
            r.give_clue(Clue(word, 1))
        self.assertIsNone(r.clue)

    def test_must_guess_once_before_passing(self):
        r = Rules(Board(5, 5, ["red", "blue"], WORDS, random.Random(0)), ["red", "blue"])
        r.give_clue(Clue("x", 2))
        with self.assertRaises(ValueError):
            r.guess(None)
        own = next(p for p, c in r.board.cells() if c.owner == "red")
        r.guess(own)
        r.guess(None)
        self.assertEqual(r.current, "blue")


if __name__ == "__main__":
    unittest.main()
