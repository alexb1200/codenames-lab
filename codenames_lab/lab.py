"""Shared pieces for experiments: build models and players from plain config dicts,
play games and single turns, and append raw results to JSONL files under results/.

A team config looks like
    {"model": "w2v", "spy": "risk", "T": 0.5, "max_risk": 0.2, "guesser": "soft"}
Spymaster kinds: sim (margin, bias), risk (T, max_risk), learn (max_risk),
secret (T, max_risk, kappa, secret_read). Guesser kinds: sim, soft (T), eaves (T, read, weight),
bayes (T, read, read_T). The guesser's T defaults to the spymaster's T."""
import json
import os
import random
from functools import cache

import numpy as np

from .game import ASSASSIN, NEUTRAL, Board, Clue, Game, Rules, Team
from .models import Cooc, HubCorrected, Model, PPMIVectors, Vectors
from .players import (BayesEavesdropGuesser, EavesdropGuesser, LearningSpymaster, RiskSpymaster,
                      SecretSpymaster, SimGuesser, SimSpymaster, SoftGuesser)
from .words import nltk_pool

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
MAX_STEPS = 2000  # a game that runs past this is a bug, not a draw
TEAMS = ["red", "blue"]


# ---------- models, vocabulary, boards ----------

@cache
def models() -> dict[str, Model]:
    cooc = Cooc()
    w2v = Vectors()
    out: dict[str, Model] = {"w2v": w2v, "ppmi": cooc, "ppmi-cos": PPMIVectors(cooc)}
    out["w2v-hub"] = HubCorrected(w2v, shared_clue_words_from(out), k=10)
    return out


def shared_clue_words_from(ms: dict[str, Model]) -> list[str]:
    cooc = ms["ppmi"]
    return [w for w in cooc.clue_words if ms["w2v"].knows(w) and cooc.knows(w)]


@cache
def clue_words() -> list[str]:
    """The 4,321 clue words that every model knows."""
    return shared_clue_words_from(models())


@cache
def pool() -> list[str]:
    """Board words every model knows (372 common concrete nouns)."""
    w2v, cooc = models()["w2v"], models()["ppmi"]
    assert isinstance(w2v, Vectors) and isinstance(cooc, Cooc)
    known = set(w2v.index) & set(cooc.index)
    return nltk_pool(vocab=known)


def board(seed: int, height: int = 5, width: int = 5, words: list[str] | None = None) -> Board:
    return Board(height, width, TEAMS, words or pool(), random.Random(seed))


# ---------- players from configs ----------

def make_spymaster(cfg: dict, seed: int):
    m = models()[cfg["model"]]
    kind = cfg["spy"]
    words = cfg.get("clue_words_list") or clue_words()
    rng = np.random.default_rng(seed)
    if kind == "sim":
        return SimSpymaster(m, cfg.get("margin", 0.0), cfg.get("bias", 0.0), clue_words=words)
    if kind == "risk":
        return RiskSpymaster(m, cfg["T"], cfg["max_risk"], clue_words=words, rng=rng)
    if kind == "learn":
        return LearningSpymaster(m, cfg["max_risk"], clue_words=words, rng=rng)
    if kind == "secret":
        return SecretSpymaster(m, cfg["T"], cfg["max_risk"], cfg["kappa"], models()[cfg.get("secret_read", cfg["model"])], cfg["T"],
                               clue_words=words, rng=rng)
    raise ValueError(f"unknown spymaster {kind}")


def make_guesser(cfg: dict, seed: int):
    m = models()[cfg.get("guess_model", cfg["model"])]
    kind = cfg.get("guesser", "sim")
    t = float(cfg.get("guess_T", cfg.get("T", 0.0)))
    rng = random.Random(seed)
    if kind == "sim":
        return SimGuesser(m, rng)
    if kind == "soft":
        return SoftGuesser(m, t, cfg.get("stop", 0.0), rng)
    if kind == "eaves":
        return EavesdropGuesser(m, t, models()[cfg["read"]], cfg["weight"], 0.0, rng)
    if kind == "bayes":
        return BayesEavesdropGuesser(m, t, models()[cfg["read"]], float(cfg.get("read_T", t)), 0.0, rng)
    raise ValueError(f"unknown guesser {kind}")


def public(cfg: dict) -> dict:
    """The config as logged: drop bulky entries."""
    return {k: v for k, v in cfg.items() if k != "clue_words_list"}


# ---------- playing ----------

def outcome_counts(history, team: str) -> dict[str, int]:
    c = {"own": 0, "opp": 0, "neutral": 0, "assassin": 0}
    for e in history:
        if e.kind == "guess" and e.team == team:
            o = e.owner
            c["own" if o == team else "assassin" if o == ASSASSIN else "neutral" if o == NEUTRAL else "opp"] += 1
    return c


def play_game(cfg_red: dict, cfg_blue: dict, seed: int, height: int = 5, width: int = 5) -> dict:
    """One full game. Player seeds are 2*seed (red) and 2*seed+1 (blue); the board uses seed."""
    teams = [Team(color, make_spymaster(cfg, 2 * seed + i), [make_guesser(cfg, 2 * seed + i)])
             for i, (color, cfg) in enumerate([("red", cfg_red), ("blue", cfg_blue)])]
    game = Game(teams)
    game.reset(board(seed, height, width))
    steps = 0
    while not game.rules.done:
        game.step()
        steps += 1
        assert steps < MAX_STEPS, f"game {seed} did not finish"
    r = game.rules
    return {
        "seed": seed, "winner": r.winner, "reason": r.reason, "turns": r.turn_number,
        "clues": {c: [e.count for e in r.history if e.kind == "clue" and e.team == c] for c in TEAMS},
        "guesses": {c: outcome_counts(r.history, c) for c in TEAMS},
        "events": [[e.turn, e.team, e.kind, None if e.token is None else str(e.token), e.count,
                    list(e.pos) if e.pos else None, e.owner] for e in r.history],
        "cells": [[str(cell.token), cell.owner] for _, cell in r.board.cells()],
    }


def play_match(path: str, experiment: str, row: str, cfg_a: dict, cfg_b: dict, games: int, **extra) -> None:
    """Seat-swapped match: A is red (moves first) on even seeds, blue on odd seeds.
    Writes one line per game with both configs, A's colour and the game record."""
    for seed in range(games):
        a_red = seed % 2 == 0
        rec = play_game(cfg_a if a_red else cfg_b, cfg_b if a_red else cfg_a, seed)
        write(path, {"experiment": experiment, "row": row, "a": public(cfg_a), "b": public(cfg_b),
                     "a_color": "red" if a_red else "blue", **extra, **rec})


def one_turn(rules: Rules, clue: Clue, guesser, team: str = "red") -> list[str]:
    """Give `clue` for `team`, let `guesser` play the turn out. Returns owners of the picks, in order."""
    rules.give_clue(clue)
    owners = []
    while not rules.done and rules.current == team and rules.clue is not None:
        pos = guesser.guess(rules.view(team, spymaster=False), [])
        if pos is not None:
            owners.append(rules.board.grid[pos[0]][pos[1]].owner)
        rules.guess(pos)
    return owners


def run_of_own(owners: list[str], team: str = "red") -> int:
    n = 0
    for o in owners:
        if o != team:
            break
        n += 1
    return n


# ---------- output ----------

def write(path: str, record: dict) -> None:
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, path), "a") as f:
        f.write(json.dumps(record, default=_default) + "\n")


def _default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (tuple, set)):
        return list(x)
    return str(x)


def fresh(path: str) -> None:
    """Start a results file from empty."""
    full = os.path.join(RESULTS, path)
    if os.path.exists(full):
        os.remove(full)


def read(path: str) -> list[dict]:
    with open(os.path.join(RESULTS, path)) as f:
        return [json.loads(line) for line in f]


def n(full: int, quick: int = 2) -> int:
    """Sample size: `full`, or `quick` when the environment sets QUICK=1 (for smoke tests)."""
    return quick if os.environ.get("QUICK") == "1" else full


def results_name(script_file: str) -> str:
    """results/<experiment name>.jsonl, e.g. e01_transfer.jsonl for experiments/e01_transfer.py."""
    base = os.path.splitext(os.path.basename(script_file))[0]
    return base + (".quick" if os.environ.get("QUICK") == "1" else "") + ".jsonl"
