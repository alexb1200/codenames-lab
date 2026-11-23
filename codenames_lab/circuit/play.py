"""Play a compiled circuit on the repo's own engine.

The rules are game.Rules, unchanged: clue for n allows n + 1 guesses, a wrong card ends
the turn, the assassin loses, a team whose cards are all revealed wins. The subclasses
here only change how state is stored, because a Life board has 63,155 cards and a game
takes about 40,000 steps:
  * CircuitBoard lays the cards out row by row and keeps a count of hidden cards per
    owner, so remaining() is O(1).
  * CircuitRules hands out views that share the live grids instead of copying them, and
    checks clue legality against a set of hidden card names. Players must not mutate a view.

Players (ported from game_shared.py):
  RankSpymaster 'max' (honest): among words whose top unrevealed cards are its own, gives
    the one with the longest run of its own cards (count n), ties at random. With no
    such word it gives the pass word for 1, which hits a bystander.
  RankSpymaster 'any': gives any safe word at random, with its full run as n.
  RankGuesser: takes the top-ranked unrevealed card of the clue word, n times, then
    passes. With probability eps it takes the second-ranked card instead.
Both work from the View alone: the spymaster from the spymaster view and the history,
the guesser from the revealed grid, the clue and the history.
"""
import random
from dataclasses import dataclass

from ..game import ASSASSIN, NEUTRAL, Board, Cell, Clue, Game, Guesser, Pos, Rules, Spymaster, Team, View
from .compile import ASS, BYS, B, R, Circuit

OWNER = {R: "red", B: "blue", BYS: NEUTRAL, ASS: ASSASSIN}
TEAMS = ["red", "blue"]


@dataclass(frozen=True)
class Card:
    """A card token: printed as c<id>, so it can never clash with a clue word w<id>."""
    id: int
    label: str

    def __str__(self) -> str:
        return f"c{self.id}"


class CircuitBoard(Board):
    """The compiled cards in a near-square grid, card id i at divmod(i, width). The few
    cells past the last card are inert neutral padding: no word ranks them."""

    def __init__(self, circuit: Circuit) -> None:
        n = len(circuit.colour)
        self.width = int(n ** 0.5) + (int(n ** 0.5) ** 2 < n)
        self.height = -(-n // self.width)
        cards = [Cell(Card(i, circuit.label[i]), OWNER[circuit.colour[i]]) for i in range(n)]
        cards += [Cell(Card(i, "pad"), NEUTRAL) for i in range(n, self.height * self.width)]
        self.grid = [cards[r * self.width:(r + 1) * self.width] for r in range(self.height)]
        self.hidden = {o: 0 for o in OWNER.values()}
        for cell in cards:
            self.hidden[cell.owner] += 1

    def pos(self, card: int) -> Pos:
        r, c = divmod(card, self.width)
        return (r, c)

    def remaining(self, owner: str) -> int:
        return self.hidden[owner]


class CircuitRules(Rules):
    board: CircuitBoard

    def __init__(self, board: CircuitBoard, teams: list[str]) -> None:
        super().__init__(board, teams)
        grid = board.grid
        self.tokens = [[cell.token for cell in row] for row in grid]
        self.owners = [[cell.owner for cell in row] for row in grid]
        self.public: list[list[str | None]] = [[None for _ in row] for row in grid]   # owners shown to guessers
        self.revealed_grid = [[False for _ in row] for row in grid]
        self.hidden_names = {str(cell.token) for _, cell in board.cells()}

    def view(self, team: str, spymaster: bool) -> View:
        owners = self.owners if spymaster else self.public
        return View(team, self.tokens, owners, self.revealed_grid, self.clue, self.history)  # type: ignore[arg-type]

    def check_clue(self, clue: Clue) -> str | None:
        token = str(clue.token)
        if not token or any(ch.isspace() for ch in token):
            return "clue must be one word"
        if token.lower() in self.hidden_names:
            return f"{token!r} is a word on the board"
        if clue.count < 0:
            return "count must be 0 or more"
        return None

    def reveal(self, pos: Pos) -> str:
        owner = super().reveal(pos)
        r, c = pos
        self.board.hidden[owner] -= 1
        self.public[r][c] = owner
        self.revealed_grid[r][c] = True
        self.hidden_names.discard(str(self.board.grid[r][c].token))
        return owner


class CircuitGame(Game):
    def reset(self, board: Board) -> Rules:
        assert isinstance(board, CircuitBoard)
        self.rules = CircuitRules(board, list(self.teams))
        self.renderer.draw(self.rules)
        return self.rules


def word_id(clue: Clue) -> int:
    return int(str(clue.token)[1:])


class Tracker:
    """Per-word pointer to the top unrevealed card, kept up to date from the history."""

    def __init__(self, circuit: Circuit, width: int) -> None:
        self.circuit = circuit
        self.width = width
        n = len(circuit.colour)
        self.revealed = [False] * n
        self.ptr = [0] * len(circuit.words)
        self.by_card: list[list[int]] = [[] for _ in range(n)]
        for wi, lst in enumerate(circuit.words):
            for c in lst:
                self.by_card[c].append(wi)
        self.seen = 0

    def sync(self, view: View) -> list[int]:
        """Apply guesses made since the last call. Returns the words whose top card moved.
        A shorter history than before means a new game: start over."""
        if len(view.history) < self.seen:
            self.__init__(self.circuit, self.width)  # type: ignore[misc]
        touched = []
        for e in view.history[self.seen:]:
            if e.kind == "guess" and e.pos is not None:
                card = e.pos[0] * self.width + e.pos[1]
                self.revealed[card] = True
                touched.extend(self.by_card[card])
        self.seen = len(view.history)
        return touched

    def advance(self, wi: int) -> int:
        """Move word wi's pointer past revealed cards; return its top card."""
        lst = self.circuit.words[wi]
        while self.ptr[wi] < len(lst) - 1 and self.revealed[lst[self.ptr[wi]]]:
            self.ptr[wi] += 1
        return lst[self.ptr[wi]]

    def ranked(self, wi: int, k: int) -> list[int]:
        """The top k unrevealed cards of word wi."""
        lst = self.circuit.words[wi]
        self.advance(wi)
        out: list[int] = []
        for c in lst[self.ptr[wi]:]:
            if not self.revealed[c]:
                out.append(c)
                if len(out) == k:
                    break
        return out


class IndexedSet:
    """A set with O(1) add, discard and uniform random choice."""

    def __init__(self) -> None:
        self.items: list[int] = []
        self.pos: dict[int, int] = {}

    def add(self, x: int) -> None:
        if x not in self.pos:
            self.pos[x] = len(self.items)
            self.items.append(x)

    def discard(self, x: int) -> None:
        i = self.pos.pop(x, None)
        if i is None:
            return
        last = self.items.pop()
        if i < len(self.items):
            self.items[i] = last
            self.pos[last] = i

    def __len__(self) -> int:
        return len(self.items)

    def choice(self, rng: random.Random) -> int:
        return self.items[rng.randrange(len(self.items))]


class RankSpymaster(Spymaster):
    """Gives the word that scores the most own cards ('max') or any safe word ('any').
    A word is safe for a team when its top unrevealed card is that team's. Its score is
    the run of that team's cards from the top, skipping revealed cards."""

    def __init__(self, circuit: Circuit, width: int, rng: random.Random, mode: str = "max") -> None:
        assert mode in ("max", "any")
        self.circuit, self.width, self.rng, self.mode = circuit, width, rng, mode
        self.start()

    def start(self) -> None:
        self.track = Tracker(self.circuit, self.width)
        self.owner: list[int | None] = [None] * len(self.circuit.words)
        self.run = [0] * len(self.circuit.words)
        self.buckets: dict[int, dict[int, IndexedSet]] = {R: {}, B: {}}
        for wi in range(len(self.circuit.words)):
            self.refresh(wi)

    def refresh(self, wi: int) -> None:
        col, track = self.circuit.colour, self.track
        lst = self.circuit.words[wi]
        if self.owner[wi] is not None:
            self.buckets[self.owner[wi]][self.run[wi]].discard(wi)  # type: ignore[index]
        t = col[track.advance(wi)]
        if t in (R, B) and wi != self.circuit.pass_word:
            n = 0
            for c in lst[track.ptr[wi]:]:
                if track.revealed[c]:
                    continue
                if col[c] != t:
                    break
                n += 1
            self.owner[wi], self.run[wi] = t, n
            self.buckets[t].setdefault(n, IndexedSet()).add(wi)
        else:
            self.owner[wi], self.run[wi] = None, 0

    def give_clue(self, view: View) -> Clue:
        if len(view.history) < self.track.seen:
            self.start()
        for wi in set(self.track.sync(view)):
            self.refresh(wi)
        d = self.buckets[TEAMS.index(view.team)]
        live = [n for n, s in d.items() if n > 0 and len(s)]
        if not live:
            return Clue(f"w{self.circuit.pass_word}", 1)
        if self.mode == "max":
            n = max(live)
            wi = d[n].choice(self.rng)
        else:
            r = self.rng.randrange(sum(len(d[k]) for k in live))
            for n in live:
                if r < len(d[n]):
                    wi = d[n].items[r]
                    break
                r -= len(d[n])
        return Clue(f"w{wi}", n)


class RankGuesser(Guesser):
    """Takes the clue word's top-ranked unrevealed card, `count` times, then passes."""

    def __init__(self, circuit: Circuit, width: int, rng: random.Random, eps: float = 0.0) -> None:
        self.circuit, self.width, self.rng, self.eps = circuit, width, rng, eps
        self.track = Tracker(circuit, width)

    def guess(self, view: View, said: list[Pos | None]) -> Pos | None:
        assert view.clue is not None
        self.track.sync(view)
        made = 0
        for e in reversed(view.history):
            if e.kind == "clue":
                break
            made += 1
        if made == view.clue.count:
            return None
        ranked = self.track.ranked(word_id(view.clue), 2)
        pick = ranked[1] if (self.eps and len(ranked) > 1 and self.rng.random() < self.eps) else ranked[0]
        r, c = divmod(pick, self.width)
        return (r, c)


def teams(circuit: Circuit, width: int, seed: int = 0, mode: str = "max", eps: float = 0.0) -> list[Team]:
    """Red and Blue, one spymaster and one guesser each, with their own random streams."""
    out = []
    for k, name in enumerate(TEAMS):
        out.append(Team(name, RankSpymaster(circuit, width, random.Random(4 * seed + 2 * k), mode),
                        [RankGuesser(circuit, width, random.Random(4 * seed + 2 * k + 1), eps)]))
    return out


def play(circuit: Circuit, seed: int = 0, mode: str = "max", eps: float = 0.0,
         renderer=None, max_steps: int = 10**7) -> CircuitRules:
    board = CircuitBoard(circuit)
    game = CircuitGame(teams(circuit, board.width, seed, mode, eps), renderer)
    rules = game.reset(board)
    for _ in range(max_steps):
        if rules.done:
            break
        game.step()
    return rules  # type: ignore[return-value]


def revealed_cards(rules: CircuitRules) -> list[bool]:
    return [cell.revealed for row in rules.board.grid for cell in row]


def read_port(circuit: Circuit, revealed: list[bool], port: str) -> tuple[int, int]:
    """The whole readout: bit i is 1 if its '1' card is revealed and its '0' card is not.
    Returns (value, number of unreadable bits: both or neither revealed)."""
    value = bad = 0
    for (name, i), (c0, c1) in circuit.outputs.items():
        if name != port:
            continue
        if revealed[c1] and not revealed[c0]:
            value |= 1 << i
        elif revealed[c0] == revealed[c1]:
            bad += 1
    return value, bad


def summary(circuit: Circuit, rules: CircuitRules) -> dict:
    """Turns, clues by kind, count histogram and leaks, from the history alone.
    A reveal is a leak if the card is a signal card taken from a word where it is not
    part of the word's target (the last three cards before the assassin); sweep excluded."""
    label = circuit.label
    is_signal = [not (lb.startswith(("spacer", "filler", "pass")) or lb == "assassin") for lb in label]
    # turn_number counts finished turns; the game ends inside one more
    n = dict(clues=0, passes=0, filler_clues=0, reveals=0, leaks=0)
    counts: dict[int, int] = {}
    wi = None
    width = rules.board.width
    for e in rules.history:
        if e.kind == "clue":
            wi = word_id(Clue(e.token, e.count or 0))
            if wi == circuit.pass_word:
                n["passes"] += 1
            else:
                n["clues"] += 1
                counts[e.count or 0] = counts.get(e.count or 0, 0) + 1
                n["filler_clues"] += circuit.word_label[wi].startswith("filler")
        elif e.kind == "guess" and e.pos is not None and wi is not None:
            n["reveals"] += 1
            card = e.pos[0] * width + e.pos[1]
            if (is_signal[card] and card not in circuit.words[wi][-4:-1]
                    and not circuit.word_label[wi].startswith("sweep")):
                n["leaks"] += 1
    return dict(turns=rules.turn_number + 1, **n, winner=rules.winner, how=rules.reason, counts=counts)
