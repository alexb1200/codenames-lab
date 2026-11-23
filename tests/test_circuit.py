"""The circuit compiler on the repo's engine. Needs Yosys (pip install yowasp-yosys);
skipped without it."""
import shutil
import unittest

from codenames_lab.game import Board, Game
from codenames_lab.circuit.compile import DESIGNS, compile_shared, synth
from codenames_lab.circuit.life import life_ref, wrong_pixels
from codenames_lab.circuit.play import CircuitBoard, CircuitGame, play, read_port, revealed_cards, summary, teams

HAVE_YOSYS = shutil.which("yowasp-yosys") is not None


@unittest.skipUnless(HAVE_YOSYS, "yowasp-yosys not installed")
class TestCircuit(unittest.TestCase):
    adder: dict

    @classmethod
    def setUpClass(cls) -> None:
        cls.adder = synth(f"{DESIGNS}/adder.v", "adder4")

    def test_adder_all_inputs(self) -> None:
        for a in range(16):
            for b in range(16):
                c = compile_shared(self.adder, {"a": a, "b": b}, fillers_per_team=200)
                rules = play(c, seed=a * 16 + b)
                self.assertEqual(read_port(c, revealed_cards(rules), "sum"), (a + b, 0))
                self.assertEqual(summary(c, rules)["leaks"], 0)

    def test_small_life(self) -> None:
        """4x4 torus, 2 generations, four starting states."""
        mod = synth(f"{DESIGNS}/life.v", "life", {"W": 4, "H": 4, "T": 2})
        for s0 in (0b0000011100000000, 0b0110011000000000, 0b1010010110100101, 0xFFFF):
            c = compile_shared(mod, {"s0": s0}, fillers_per_team=1000)
            rules = play(c, seed=s0)
            ref = life_ref(s0, 2, 4, 4)
            self.assertEqual(read_port(c, revealed_cards(rules), "hist"), (ref[0] | ref[1] << 16, 0))
            self.assertEqual(wrong_pixels(c, revealed_cards(rules), s0, 2, 4, 4), 0)

    def test_fast_rules_match_plain_rules(self) -> None:
        """CircuitRules only changes storage: the same players give the same game on plain Rules."""

        class PlainBoard(CircuitBoard):
            remaining = Board.remaining   # count by scanning, as the base Board does

        c = compile_shared(self.adder, {"a": 11, "b": 6}, fillers_per_team=100)
        fast = play(c, seed=5)
        board = PlainBoard(c)
        slow = Game(teams(c, board.width, seed=5)).run(board)
        self.assertEqual(fast.history, slow.history)
        self.assertEqual((fast.winner, fast.reason), (slow.winner, slow.reason))
        self.assertIsInstance(CircuitGame(teams(c, board.width)).reset(CircuitBoard(c)), type(fast))


if __name__ == "__main__":
    unittest.main()
