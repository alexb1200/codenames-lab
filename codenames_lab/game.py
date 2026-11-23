"""The game: board, rules, teams and the loop that runs them.

Rules holds the state and the legal moves and never sees a player. Game holds the teams
and asks the current one for a single action per step(). A human seat is a Human marker;
the GUI feeds its moves to Rules directly.
"""
import random
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import product


NEUTRAL, ASSASSIN = "NEUTRAL", "ASSASSIN"
type Pos = tuple[int, int]


@dataclass
class Cell:
    token: object
    owner: str
    revealed: bool = False


@dataclass(frozen=True)
class Clue:
    token: object
    count: int


@dataclass(frozen=True)
class Event:
    turn: int                 # turn number, counts up across the game
    team: str
    kind: str                 # "clue", "guess" or "pass"
    token: object = None      # clue token or guessed token
    count: int | None = None  # clue count
    pos: Pos | None = None    # guessed cell
    owner: str | None = None  # owner revealed by the guess


@dataclass(frozen=True)
class View:
    team: str
    tokens: list[list[object]]
    owners: list[list[str | None]]  # None = hidden from this player
    revealed: list[list[bool]]
    clue: Clue | None
    history: list[Event]


class Board:
    """A height x width grid of cells. Every team gets n // (teams + 1) cards, the first
    team one more (it moves first), then the assassins; the rest are neutral. For 5x5 and
    two teams that is the real game's 9 / 8 / 7 / 1. Tokens can be any objects."""

    def __init__(self, height: int, width: int, teams: list[str], pool: Sequence[object],
                 rng: random.Random, assassins: int = 1) -> None:
        self.height, self.width = height, width
        n = height * width
        per_team = n // (len(teams) + 1)
        owners = [teams[0]]
        owners += [t for t in teams for _ in range(per_team)]
        owners += [ASSASSIN] * assassins
        assert len(owners) <= n, "board too small"
        owners += [NEUTRAL] * (n - len(owners))
        rng.shuffle(owners)
        tokens = rng.sample(pool, n)
        self.grid = [[Cell(tokens[r * width + c], owners[r * width + c]) for c in range(width)]
                     for r in range(height)]

    def cells(self):
        for r, c in product(range(self.height), range(self.width)):
            yield (r, c), self.grid[r][c]

    def remaining(self, owner: str) -> int:
        return sum(1 for _, cell in self.cells() if cell.owner == owner and not cell.revealed)


# ---------- rules: state and legal moves, no players ----------

class Rules:
    """Game state and legal moves for any number of teams. A clue for count n allows up to
    n + 1 guesses; the turn ends on a wrong card, after the guesses, or on a pass, and a
    team must guess at least once per turn. A team that reveals the assassin is out; the
    last team in wins. A team whose cards are all revealed wins, whoever revealed them.
    One difference from the box rules: a count of 0 here allows one guess, not unlimited."""

    def __init__(self, board: Board, teams: list[str]) -> None:
        self.board = board
        self.alive = list(teams)
        self.turn = 0          # index into alive
        self.turn_number = 0   # counts up, for the log
        self.clue: Clue | None = None
        self.guesses_left = 0
        self.history: list[Event] = []
        self.winner: str | None = None
        self.reason: str | None = None

    @property
    def current(self) -> str:
        return self.alive[self.turn]

    @property
    def done(self) -> bool:
        return self.winner is not None

    def view(self, team: str, spymaster: bool) -> View:
        grid = self.board.grid
        tokens = [[cell.token for cell in row] for row in grid]
        owners = [[cell.owner if spymaster or cell.revealed else None for cell in row] for row in grid]
        revealed = [[cell.revealed for cell in row] for row in grid]
        return View(team, tokens, owners, revealed, self.clue, list(self.history))

    def check_clue(self, clue: Clue) -> str | None:
        """Why this clue is illegal, or None if it is fine."""
        token = str(clue.token)
        if not token or any(ch.isspace() for ch in token):
            problem = "clue must be one word"
        elif any(str(cell.token).lower() == token.lower() for _, cell in self.board.cells() if not cell.revealed):
            problem = f"{token!r} is a word on the board"
        elif clue.count < 0:
            problem = "count must be 0 or more"
        else:
            problem = None
        return problem

    def give_clue(self, clue: Clue) -> None:
        assert not self.done and self.clue is None, "not time for a clue"
        problem = self.check_clue(clue)
        if problem:
            raise ValueError(problem)
        self.clue = clue
        self.guesses_left = clue.count + 1
        self.history.append(Event(self.turn_number, self.current, "clue", clue.token, clue.count))

    def guess(self, pos: Pos | None) -> None:
        assert not self.done and self.clue is not None, "not time for a guess"
        if pos is None and self.guesses_left == self.clue.count + 1:
            raise ValueError("you must make at least one guess before passing")
        if pos is None:
            self.history.append(Event(self.turn_number, self.current, "pass"))
            self.end_turn()
        else:
            self.resolve(self.reveal(pos))

    def reveal(self, pos: Pos) -> str:
        r, c = pos
        cell = self.board.grid[r][c]
        assert not cell.revealed, "already revealed"
        cell.revealed = True
        self.history.append(Event(self.turn_number, self.current, "guess", cell.token, pos=pos, owner=cell.owner))
        return cell.owner

    def resolve(self, owner: str) -> None:
        if owner == ASSASSIN:
            self.eliminate(self.current)
        elif owner in self.alive and self.board.remaining(owner) == 0:
            self.finish(owner, "found")
        else:
            self.guesses_left -= 1
            if owner != self.current or self.guesses_left == 0:
                self.end_turn()

    def end_turn(self) -> None:
        self.clue = None
        self.turn_number += 1
        self.turn = (self.turn + 1) % len(self.alive)

    def eliminate(self, team: str) -> None:
        self.alive.remove(team)  # the next team slides into this index
        self.clue = None
        self.turn_number += 1
        if len(self.alive) == 1:
            self.finish(self.alive[0], "assassin")
        else:
            self.turn %= len(self.alive)

    def finish(self, winner: str, reason: str) -> None:
        self.winner, self.reason = winner, reason
        self.clue = None


# ---------- players and teams ----------

class Spymaster:
    def give_clue(self, view: View) -> Clue:
        raise NotImplementedError


class Guesser:
    def guess(self, view: View, said: list[Pos | None]) -> Pos | None:
        """said = what teammates before me proposed this round, in order."""
        raise NotImplementedError


class Human(Spymaster, Guesser):
    """Marks a seat whose moves come from outside step(): GUI clicks or a terminal."""


class RandomPlayer(Spymaster, Guesser):
    """Random clue words and random hidden cards. Useful for testing the rules."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    def give_clue(self, view: View) -> Clue:
        return Clue(f"clue{self.rng.randrange(10**6)}", self.rng.randint(1, 3))

    def guess(self, view: View, said: list[Pos | None]) -> Pos | None:
        hidden = [(r, c) for r, row in enumerate(view.revealed) for c, rev in enumerate(row) if not rev]
        return self.rng.choice(hidden)


def majority(said: list[Pos | None]) -> Pos | None:
    pos, _ = Counter(said).most_common(1)[0]
    return pos


@dataclass
class Team:
    name: str
    spymaster: Spymaster
    guessers: list[Guesser]
    decide: Callable[[list[Pos | None]], Pos | None] = majority

    def guess(self, view: View) -> Pos | None:
        said: list[Pos | None] = []
        for g in self.guessers:
            said.append(g.guess(view, list(said)))
        return self.decide(said)


# ---------- rendering ----------

class Renderer:
    def draw(self, rules: Rules) -> None:
        pass


class TextRenderer(Renderer):
    def __init__(self, spymaster: bool = False) -> None:
        self.spymaster = spymaster

    def draw(self, rules: Rules) -> None:
        view = rules.view(rules.current, self.spymaster)
        for tok_row, own_row in zip(view.tokens, view.owners):
            print(" ".join(f"{str(t) + (f'[{o[0]}]' if o else ''):<12}" for t, o in zip(tok_row, own_row)))
        print(f"turn={rules.turn_number} team={rules.current} clue={rules.clue} winner={rules.winner}\n")


# ---------- game: runs teams against the rules ----------

class Game:
    """Runs teams against the Rules. step() asks the current team for one action: a clue
    if none is out, otherwise one guess. A renderer is drawn after every step."""

    def __init__(self, teams: list[Team], renderer: Renderer | None = None) -> None:
        self.teams = {t.name: t for t in teams}
        self.renderer = renderer or Renderer()

    def reset(self, board: Board) -> Rules:
        self.rules = Rules(board, list(self.teams))
        self.renderer.draw(self.rules)
        return self.rules

    def waiting_on_human(self) -> bool:
        team = self.teams[self.rules.current]
        seat = [team.spymaster] if self.rules.clue is None else team.guessers
        return any(isinstance(p, Human) for p in seat)

    def step(self) -> Rules:
        assert not self.waiting_on_human(), "a human must move"
        rules = self.rules
        team = self.teams[rules.current]
        if rules.clue is None:
            rules.give_clue(team.spymaster.give_clue(rules.view(team.name, spymaster=True)))
        else:
            rules.guess(team.guess(rules.view(team.name, spymaster=False)))
        self.renderer.draw(rules)
        return rules

    def run(self, board: Board) -> Rules:
        self.reset(board)
        while not self.rules.done:
            self.step()
        return self.rules
