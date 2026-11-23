"""Players. Every spymaster and guesser here works with any word model.

SimSpymaster          largest clue whose cards all beat the closest bad card (plus margin)
SimGuesser            takes the closest hidden card, up to the clue count
SoftGuesser           picks in softmax order at a temperature (a noisy reader)
EavesdropGuesser      SoftGuesser that avoids cards near opponent clues (crude)
BayesEavesdropGuesser SoftGuesser that tracks how many of each opponent clue's targets remain
RiskSpymaster         most expected cards under a risk limit, against an imagined SoftGuesser
LearningSpymaster     RiskSpymaster that learns its guesser's temperature from its picks
SecretSpymaster       RiskSpymaster that also pays for what a clue tells the other team
"""
import random

import numpy as np

from .game import ASSASSIN, NEUTRAL, Clue, Guesser, Pos, Spymaster, View
from .models import Model


def flat(view: View) -> tuple[list[str], list[str | None], list[bool], list[Pos]]:
    """The view's grid as flat lists, row by row: tokens, owners, revealed flags, positions."""
    tokens = [str(t) for row in view.tokens for t in row]
    owners = [o for row in view.owners for o in row]
    revealed = [x for row in view.revealed for x in row]
    width = len(view.tokens[0])
    positions = [(i // width, i % width) for i in range(len(tokens))]
    return tokens, owners, revealed, positions


class SimSpymaster(Spymaster):
    """Pick the clue that is close to as many of my words as possible, with a safety gap
    over the closest bad word. The assassin counts as closer than it is, by `assassin_bias`.
    `margin` and `assassin_bias` are in the model's own units."""

    def __init__(self, model: Model, margin: float = 0.0, assassin_bias: float = 0.0,
                 clue_words: list[str] | None = None) -> None:
        self.model = model
        self.clue_words = clue_words if clue_words is not None else model.clue_words
        self.margin = margin
        self.assassin_bias = assassin_bias
        self.name = f"spy[{model.name},m={margin},a={assassin_bias}]"
        self.unsafe_clues = 0  # clues given with no card safely past the danger

    def give_clue(self, view: View) -> Clue:
        tokens, owners, revealed, _ = flat(view)
        hidden = [i for i, r in enumerate(revealed) if not r]
        mine = [i for i in hidden if owners[i] == view.team]
        bad = [i for i in hidden if owners[i] != view.team]

        sims = self.model.sims(self.clue_words, [tokens[i] for i in hidden])   # (clue words, hidden)
        col = {i: j for j, i in enumerate(hidden)}
        mine_sims = sims[:, [col[i] for i in mine]]
        if bad:
            bias = np.array([self.assassin_bias if owners[i] == ASSASSIN else 0.0 for i in bad])
            danger = (sims[:, [col[i] for i in bad]] + bias).max(axis=1)  # closest bad word, per clue
        else:
            danger = np.full(len(sims), -np.inf, dtype=np.float32)

        count = (mine_sims > danger[:, None] + self.margin).sum(axis=1)  # my words safely closer
        kth = -np.sort(-mine_sims, axis=1)                                # my sims, high to low
        kth_sim = kth[np.arange(len(count)), np.maximum(count, 1) - 1]
        gap = kth_sim - danger

        words = self.clue_words
        legal = np.array([self.legal(w, tokens) for w in words])
        score = np.where(legal, count + 0.5 * np.tanh(gap), -np.inf)  # count first, gap breaks ties
        best = int(np.argmax(score))
        if count[best] == 0:
            # No clue gets any of my cards past the danger plus margin. The tie-break has already
            # picked the clue whose closest card of mine has the largest gap, so give it for 1.
            self.unsafe_clues += 1
            return Clue(words[best], 1)
        return Clue(words[best], int(count[best]))

    @staticmethod
    def legal(word: str, board: list[str]) -> bool:
        return all(word not in b and b not in word for b in board)


class SimGuesser(Guesser):
    """Guess the hidden word closest to the clue. Stop after `count` guesses this turn.
    The rules require one guess per turn, so if the model cannot score the clue (it does
    not know the clue word, or any hidden word), it guesses a random hidden card and
    counts it in `blind_guesses`."""

    def __init__(self, model: Model, rng: random.Random | None = None) -> None:
        self.model = model
        self.rng = rng or random.Random(0)
        self.name = f"guess[{model.name}]"
        self.blind_guesses = 0

    def guess(self, view: View, said: list[Pos | None]) -> Pos | None:
        assert view.clue is not None
        clue = str(view.clue.token)
        last_clue = max(i for i, e in enumerate(view.history) if e.kind == "clue")
        made = sum(1 for e in view.history[last_clue:] if e.kind == "guess")
        tokens, _, revealed, positions = flat(view)
        hidden = [i for i, r in enumerate(revealed) if not r]
        known = [i for i in hidden if self.model.knows(tokens[i])]
        if made >= view.clue.count:
            choice = None   # out of guesses for this clue: pass
        elif self.model.knows(clue) and known:
            scores = self.model.sims([clue], [tokens[i] for i in known])[0]
            choice = positions[known[int(np.argmax(scores))]]
        else:
            self.blind_guesses += 1
            choice = positions[self.rng.choice(hidden)]
        return choice


def zscores(sims: np.ndarray) -> np.ndarray:
    """Standardise each row (one clue) across the board cards, so a temperature
    means the same thing for every model, whatever its units."""
    mean = sims.mean(axis=-1, keepdims=True)
    std = sims.std(axis=-1, keepdims=True)
    return (sims - mean) / np.maximum(std, 1e-9)


class SoftGuesser(Guesser):
    """Picks a hidden card at random with probability softmax(z / temperature), where z is
    the clue's similarity to each card, standardised over the cards in play when the clue
    was given (so its sense of the clue does not shift as cards are revealed). Low
    temperature means it nearly always takes the closest card; high means it spreads its
    picks. After its first guess it passes when its most likely card has probability below
    `stop_threshold`. It never takes more than `count` guesses. If its model does not know
    the clue word, it must still guess once, so it picks a random hidden card and counts it
    in `blind_guesses`."""

    def __init__(self, model: Model, temperature: float, stop_threshold: float = 0.0,
                 rng: random.Random | None = None) -> None:
        self.model = model
        self.temperature = temperature
        self.stop_threshold = stop_threshold
        self.rng = rng or random.Random(0)
        self.name = f"soft[{model.name},T={temperature},stop={stop_threshold}]"
        self.blind_guesses = 0
        self.early_stops = 0

    def guess(self, view: View, said: list[Pos | None]) -> Pos | None:
        assert view.clue is not None
        clue = str(view.clue.token)
        last_clue = max(i for i, e in enumerate(view.history) if e.kind == "clue")
        made = sum(1 for e in view.history[last_clue:] if e.kind == "guess")
        tokens, _, revealed, positions = flat(view)
        hidden = [i for i, r in enumerate(revealed) if not r]
        known = [i for i in hidden if self.model.knows(tokens[i])]
        # cards in play when the clue was given: still hidden, or guessed since the clue
        guessed_since = {e.pos for e in view.history[last_clue:] if e.kind == "guess"}
        at_clue = [i for i, p in enumerate(positions) if (i in known or p in guessed_since)
                   and self.model.knows(tokens[i])]
        if made >= view.clue.count:
            choice = None
        elif not self.model.knows(clue) or not known:
            self.blind_guesses += 1
            choice = positions[self.rng.choice(hidden)]
        else:
            # standardise over the cards in play at clue time, so the guesser's sense of the
            # clue does not shift as cards are revealed; then pick among the still-hidden ones
            z_all = zscores(self.model.sims([clue], [tokens[i] for i in at_clue]))[0]
            z = np.array([z_all[at_clue.index(i)] for i in known])
            logits = z / self.temperature - self.penalty(view, tokens, known)
            p = np.exp(logits - logits.max())
            p /= p.sum()
            if made >= 1 and p.max() < self.stop_threshold:
                self.early_stops += 1
                choice = None
            else:
                pick = self.rng.choices(range(len(known)), weights=p.tolist())[0]
                choice = positions[known[pick]]
        return choice

    def penalty(self, view: View, tokens: list[str], known: list[int]) -> np.ndarray:
        """Subtracted from the logits of the cards in `known`. A plain guesser has none."""
        return np.zeros(len(known))


class EavesdropGuesser(SoftGuesser):
    """A SoftGuesser that also listens to the other teams' clues. A card close to an
    opponent clue is probably theirs, so it lowers that card's logit by
    `weight` x (its largest positive z-score for any opponent clue so far). It reads
    opponent clues with `read_model`, which may differ from its own model: reading with
    the opponent's model means it knows their language. Opponent clue words or cards the
    read model does not know add no penalty."""

    def __init__(self, model: Model, temperature: float, read_model: Model, weight: float,
                 stop_threshold: float = 0.0, rng: random.Random | None = None) -> None:
        super().__init__(model, temperature, stop_threshold, rng)
        self.read_model = read_model
        self.weight = weight
        self.name = f"eaves[{model.name},T={temperature},read={read_model.name},w={weight}]"

    def penalty(self, view: View, tokens: list[str], known: list[int]) -> np.ndarray:
        opp_clues = [str(e.token) for e in view.history
                     if e.kind == "clue" and e.team != view.team and self.read_model.knows(str(e.token))]
        readable = [j for j, i in enumerate(known) if self.read_model.knows(tokens[i])]
        pen = np.zeros(len(known))
        if opp_clues and len(readable) > 1:
            z = zscores(self.read_model.sims(opp_clues, [tokens[known[j]] for j in readable]))  # (clues, cards)
            pen[readable] = np.maximum(z, 0).max(axis=0)
        return self.weight * pen


class RiskSpymaster(Spymaster):
    """Plans against an imagined guesser that picks like a SoftGuesser with temperature
    `assumed_temperature` (the real guesser's may differ). For every clue word and count k,
    it samples `samples` imagined turns. In each, the guesser picks cards in a random order
    drawn from the softmax (Gumbel trick: sort z / T + Gumbel noise), keeps going while it
    hits my cards, and stops at k or the first miss. From those samples:
        found(k) = mean number of my cards revealed
        risk(k)  = chance the turn hits an opponent card or the assassin
    It picks the (clue, k) with the most expected cards among those with risk(k) <= max_risk,
    then re-estimates that clue with `check_samples` fresh samples and lowers k until the
    fresh risk fits (picking the best of many noisy estimates favours underestimated risk).
    If even k = 1 does not fit, it gives the clue for 1 anyway and counts it in `unsafe_clues`.
    The imagined guesser never stops early, so the risk estimate errs on the cautious side."""

    MAX_COUNT = 9

    def __init__(self, model: Model, assumed_temperature: float, max_risk: float,
                 samples: int = 64, clue_words: list[str] | None = None,
                 rng: np.random.Generator | None = None, chunk: int = 1000,
                 check_samples: int = 2000) -> None:
        self.model = model
        self.assumed_temperature = assumed_temperature
        self.max_risk = max_risk
        self.samples = samples
        self.clue_words = clue_words if clue_words is not None else model.clue_words
        self.rng = rng or np.random.default_rng(0)
        self.chunk = chunk
        self.check_samples = check_samples
        self.name = f"risk[{model.name},T~{assumed_temperature},max={max_risk}]"
        self.unsafe_clues = 0

    def objective(self, found: np.ndarray, words: list[str], tokens: list[str], hidden: list[int],
                  cat: np.ndarray, ks: np.ndarray) -> np.ndarray:
        """What the spymaster maximises, shape (len(words), len(ks)). Plain: expected own cards."""
        return found

    def temperatures(self, samples: int) -> np.ndarray:
        """The imagined guesser's temperature in each sampled turn, shape (samples, 1, 1)."""
        return np.full((samples, 1, 1), self.assumed_temperature)

    def estimate(self, words: list[str], tokens: list[str], hidden: list[int], cat: np.ndarray,
                 ks: np.ndarray, samples: int) -> tuple[np.ndarray, np.ndarray]:
        """Expected cards found and risk, shape (len(words), len(ks))."""
        found_all, risk_all = [], []
        n = len(hidden)
        for start in range(0, len(words), self.chunk):
            part = words[start:start + self.chunk]
            z = zscores(self.model.sims(part, [tokens[i] for i in hidden]))             # (W, n)
            gumbel = self.rng.gumbel(size=(samples, len(part), n))
            order = np.argsort(-(z[None] / self.temperatures(samples) + gumbel), axis=-1)
            ranked = cat[order]                                                         # (S, W, n)
            miss = ranked != 0
            run = np.where(miss.any(-1), miss.argmax(-1), n)                            # my cards before first miss
            first_miss = np.take_along_axis(ranked, np.minimum(run, n - 1)[..., None], -1)[..., 0]
            bad_miss = (first_miss >= 2) & (run < n)
            found_all.append(np.minimum(run[..., None], ks).mean(0))
            risk_all.append(((run[..., None] < ks) & bad_miss[..., None]).mean(0))
        return np.concatenate(found_all), np.concatenate(risk_all)

    def give_clue(self, view: View) -> Clue:
        tokens, owners, revealed, _ = flat(view)
        hidden = [i for i, r in enumerate(revealed) if not r]
        # category of each hidden card: 0 mine, 1 neutral, 2 opponent, 3 assassin
        cat = np.array([0 if owners[i] == view.team else 3 if owners[i] == ASSASSIN
                        else 1 if owners[i] == NEUTRAL else 2 for i in hidden])
        legal = [w for w in self.clue_words if SimSpymaster.legal(w, tokens)]
        ks = np.arange(1, min(self.MAX_COUNT, int((cat == 0).sum())) + 1)
        found, risk = self.estimate(legal, tokens, hidden, cat, ks, self.samples)
        score = self.objective(found, legal, tokens, hidden, cat, ks)

        ok = risk <= self.max_risk
        if ok.any():
            value = np.where(ok, score - 1e-3 * risk, -np.inf)   # best score; lower risk breaks ties
            w, k = (int(x) for x in np.unravel_index(int(np.argmax(value)), value.shape))
        else:
            w, k = int(np.argmin(risk[:, 0])), 0
        # The pick is the best of thousands of noisy estimates, so its risk is likely
        # underestimated. Re-estimate it with fresh samples and lower the count until it fits.
        _, fresh_risk = self.estimate([legal[w]], tokens, hidden, cat, ks, self.check_samples)
        while k > 0 and fresh_risk[0, k] > self.max_risk:
            k -= 1
        if fresh_risk[0, k] > self.max_risk:
            self.unsafe_clues += 1
        return Clue(legal[w], int(ks[k]))


class LearningSpymaster(RiskSpymaster):
    """A RiskSpymaster that does not know its guesser's temperature. It keeps a posterior
    over a grid of temperatures and updates it from every guess its own guesser makes:
    the likelihood of a pick is its SoftGuesser probability at that temperature, computed
    exactly as SoftGuesser does (z standardised over the cards in play when the clue was
    given, softmax over the cards still hidden). When planning, each imagined turn draws
    its temperature from the posterior. Call observe(view) after each of your team's turns;
    the posterior carries over between games, as with a teammate you keep playing with."""

    def __init__(self, model: Model, max_risk: float, grid: np.ndarray | None = None, **kwargs) -> None:
        super().__init__(model, assumed_temperature=float("nan"), max_risk=max_risk, **kwargs)
        self.grid = grid if grid is not None else np.geomspace(0.1, 2.0, 13)
        self.log_post = np.zeros(len(self.grid))      # uniform prior over the grid
        self.seen = 0                                  # history events already used
        self.name = f"learn[{model.name},max={max_risk}]"

    def posterior(self) -> np.ndarray:
        p = np.exp(self.log_post - self.log_post.max())
        return p / p.sum()

    def mean_temperature(self) -> float:
        return float(self.posterior() @ self.grid)

    def temperatures(self, samples: int) -> np.ndarray:
        return self.rng.choice(self.grid, size=samples, p=self.posterior()).reshape(samples, 1, 1)

    def observe(self, view: View) -> None:
        tokens, _, _, positions = flat(view)
        index = {p: i for i, p in enumerate(positions)}
        revealed_before: set[int] = set()
        for e in view.history[:self.seen]:
            if e.kind == "guess":
                assert e.pos is not None
                revealed_before.add(index[e.pos])
        clue: str | None = None
        in_play: list[int] = []
        picked: set[int] = set()
        z_of: dict[int, float] = {}
        for e in view.history[self.seen:]:
            if e.kind == "clue":
                clue = str(e.token) if e.team == view.team and self.model.knows(str(e.token)) else None
                in_play = [i for i in range(len(tokens)) if i not in revealed_before]
                picked = set()
                if clue is not None:
                    z = zscores(self.model.sims([clue], [tokens[i] for i in in_play]))[0]
                    z_of = {j: float(x) for j, x in zip(in_play, z)}
            elif e.kind == "guess":
                assert e.pos is not None
                i = index[e.pos]
                if clue is not None and e.team == view.team:
                    choices = [j for j in in_play if j not in picked]
                    zc = np.array([z_of[j] for j in choices])
                    logits = zc[None, :] / self.grid[:, None]                   # (grid, choices)
                    logits -= logits.max(axis=1, keepdims=True)
                    log_p = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
                    self.log_post += log_p[:, choices.index(i)]
                picked.add(i)
                revealed_before.add(i)
        self.seen = len(view.history)


class BayesEavesdropGuesser(SoftGuesser):
    """A SoftGuesser that uses the other teams' clues as evidence about which cards are theirs.
    For each opponent clue (word, count n) it counts how many of their own cards they have
    found since, so remaining = n - found. It spreads that remaining count over the hidden
    cards that were in play at that clue, in proportion to softmax(z / read_temperature)
    under `read_model`: p_i = min(0.95, remaining x share_i), the chance card i is one of the
    targets still standing. A card's pick weight is multiplied by prod over clues of (1 - p_i),
    so a clue whose targets are all found stops counting. `read_model` may differ from the
    guesser's own model; reading with the opponent's model means knowing their language."""

    def __init__(self, model: Model, temperature: float, read_model: Model, read_temperature: float,
                 stop_threshold: float = 0.0, rng: random.Random | None = None) -> None:
        super().__init__(model, temperature, stop_threshold, rng)
        self.read_model = read_model
        self.read_temperature = read_temperature
        self.name = f"bayes-eaves[{model.name},T={temperature},read={read_model.name}]"

    def penalty(self, view: View, tokens: list[str], known: list[int]) -> np.ndarray:
        positions = flat(view)[3]
        index = {p: i for i, p in enumerate(positions)}
        hidden_now = set(known)
        log_keep = np.zeros(len(tokens))
        revealed: set[int] = set()
        events = view.history
        for n, e in enumerate(events):
            if e.kind == "guess":
                assert e.pos is not None
                revealed.add(index[e.pos])
            if e.kind != "clue" or e.team == view.team or e.count is None:
                continue
            clue = str(e.token)
            if not self.read_model.knows(clue):
                continue
            found = 0
            for later in events[n + 1:]:
                if later.kind == "clue":
                    break
                if later.kind == "guess" and later.owner == e.team:
                    found += 1
            remaining = max(e.count - found, 0)
            in_play = [i for i in range(len(tokens)) if i not in revealed and self.read_model.knows(tokens[i])]
            still = [i for i in in_play if i in hidden_now]
            if remaining == 0 or len(in_play) < 2 or not still:
                continue
            z = zscores(self.read_model.sims([clue], [tokens[i] for i in in_play]))[0]
            z_of = dict(zip(in_play, z))
            logits = np.array([z_of[i] for i in still]) / self.read_temperature
            share = np.exp(logits - logits.max())
            share /= share.sum()
            p = np.minimum(0.95, remaining * share)
            log_keep[still] += np.log1p(-p)
        return -log_keep[known]


class SecretSpymaster(RiskSpymaster):
    """A RiskSpymaster that also charges each clue for what it tells the other team.
    Leak of clue c for count k = k x (share of the opponent's reading of c that falls on my
    hidden cards), where their reading is softmax(z / read_temperature) under `read_model`.
    That is about how many of my cards they would mark as mine. It maximises
    expected own cards - kappa x leak, under the same risk limit."""

    def __init__(self, model: Model, assumed_temperature: float, max_risk: float, kappa: float,
                 read_model: Model, read_temperature: float, **kwargs) -> None:
        super().__init__(model, assumed_temperature, max_risk, **kwargs)
        self.kappa = kappa
        self.read_model = read_model
        self.read_temperature = read_temperature
        self.name = f"secret[{model.name},k={kappa},read={read_model.name}]"

    def objective(self, found: np.ndarray, words: list[str], tokens: list[str], hidden: list[int],
                  cat: np.ndarray, ks: np.ndarray) -> np.ndarray:
        z = zscores(self.read_model.sims(words, [tokens[i] for i in hidden]))   # (W, n)
        logits = z / self.read_temperature
        share = np.exp(logits - logits.max(axis=1, keepdims=True))
        share /= share.sum(axis=1, keepdims=True)
        mine_share = share[:, cat == 0].sum(axis=1)                          # (W,)
        leak = mine_share[:, None] * ks[None, :]
        return found - self.kappa * leak
