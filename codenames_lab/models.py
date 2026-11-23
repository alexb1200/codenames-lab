"""Word models. Each has a clue vocabulary and a similarity between words.

Vectors      NLTK's word2vec_sample, cosine
Cooc         positive PMI from Brown co-occurrence (first order)
PPMIVectors  cosine between rows of the PPMI matrix (second order)
HubCorrected a model with each card's general closeness removed
"""
from collections import Counter

import nltk
import numpy as np
from nltk.corpus import brown, stopwords


class Model:
    """A word similarity. `clue_words` is what a spymaster may say.
    `sims(clues, words)` is a (len(clues), len(words)) array: higher = closer."""
    name = "model"
    clue_words: list[str]

    def knows(self, word: str) -> bool:
        raise NotImplementedError

    def sims(self, clues: list[str] | None, words: list[str]) -> np.ndarray:
        """clues=None means all clue_words."""
        raise NotImplementedError


class Vectors(Model):
    """word2vec_sample from NLTK: 43,981 words x 300 dims. Rows are unit length, so dot = cosine."""
    name = "w2v"

    def __init__(self) -> None:
        path = nltk.data.find("models/word2vec_sample/pruned.word2vec.txt")
        words, rows = [], []
        with open(path) as f:
            next(f)  # header: "43981 300"
            for line in f:
                word, *nums = line.split()
                words.append(word)
                rows.append([float(x) for x in nums])
        m = np.array(rows, dtype=np.float32)
        self.matrix = m / np.linalg.norm(m, axis=1, keepdims=True)
        self.index = {w: i for i, w in enumerate(words)}
        self.clue_words = [w for w in words if w.isalpha() and w.islower() and len(w) >= 3]
        self.clue_matrix = self.matrix[[self.index[w] for w in self.clue_words]]

    def knows(self, word: str) -> bool:
        return word in self.index

    def sims(self, clues: list[str] | None, words: list[str]) -> np.ndarray:
        left = self.clue_matrix if clues is None else self.matrix[[self.index[c] for c in clues]]
        return left @ self.matrix[[self.index[w] for w in words]].T


class Cooc(Model):
    """Word co-occurrence in the Brown corpus. Two words co-occur when they are within
    `window` tokens of each other in a sentence. Similarity is positive PMI:
    max(0, log(P(a, b) / (P(a) P(b)))). Vocabulary: lowercase words seen `min_count`+
    times, 3+ letters, not NLTK stopwords."""
    name = "ppmi"

    def __init__(self, window: int = 4, min_count: int = 20) -> None:
        stop = set(stopwords.words("english"))
        sents = [[w.lower() for w in s if w.isalpha()] for s in brown.sents()]
        freq = Counter(w for s in sents for w in s)
        self.words = sorted(w for w, c in freq.items() if c >= min_count and len(w) >= 3 and w not in stop)
        self.index = {w: i for i, w in enumerate(self.words)}
        n = len(self.words)

        # one id stream, sentences separated by `window` gaps so no pair crosses a sentence
        ids: list[int] = []
        for s in sents:
            ids += [self.index.get(w, -1) for w in s] + [-1] * window
        stream = np.array(ids)

        counts = np.zeros((n, n), dtype=np.float32)
        for d in range(1, window + 1):
            a, b = stream[:-d], stream[d:]
            keep = (a >= 0) & (b >= 0)
            np.add.at(counts, (a[keep], b[keep]), 1)
        counts += counts.T  # symmetric: a before b or b before a

        total = counts.sum()
        p_word = counts.sum(axis=1) / total
        with np.errstate(divide="ignore"):
            pmi = np.log(counts / total) - np.log(p_word)[:, None] - np.log(p_word)[None, :]
        self.ppmi = np.maximum(pmi, 0).astype(np.float32)   # log(0) = -inf becomes 0
        self.clue_words = self.words
        self.window, self.min_count = window, min_count

    def knows(self, word: str) -> bool:
        return word in self.index

    def sims(self, clues: list[str] | None, words: list[str]) -> np.ndarray:
        rows = self.ppmi if clues is None else self.ppmi[[self.index[c] for c in clues]]
        return rows[:, [self.index[w] for w in words]]


class PPMIVectors(Model):
    """Second-order co-occurrence. Each word's row of the PPMI matrix is its vector:
    which words it tends to appear near. Two words are similar when they appear near
    the same words, measured by cosine between rows. Built from a Cooc, so the counts
    are shared."""
    name = "ppmi-cos"

    def __init__(self, cooc: Cooc) -> None:
        norms = np.linalg.norm(cooc.ppmi, axis=1, keepdims=True)
        self.rows = cooc.ppmi / np.maximum(norms, 1e-12)   # a word with no positive PMI stays all zero
        self.index = cooc.index
        self.clue_words = cooc.clue_words

    def knows(self, word: str) -> bool:
        return word in self.index

    def sims(self, clues: list[str] | None, words: list[str]) -> np.ndarray:
        left = self.rows if clues is None else self.rows[[self.index[c] for c in clues]]
        return left @ self.rows[[self.index[w] for w in words]].T


class HubCorrected(Model):
    """A base model with each card's hubness removed (the card side of CSLS).
    hub(w) = mean similarity of w to its `k` most similar words in `reference` (the clue
    vocabulary). Similarity becomes sim(c, w) - hub(w): a card that is close to everything
    no longer outranks a card that is close only to this clue. Subtracting a per-clue term
    as well would change nothing here, since players z-score each clue across the board."""

    def __init__(self, base: Model, reference: list[str], k: int = 10) -> None:
        self.base = base
        self.reference = reference
        self.k = k
        self.clue_words = base.clue_words
        self.name = f"{base.name}-hub"
        self._hub: dict[str, float] = {}

    def knows(self, word: str) -> bool:
        return self.base.knows(word)

    def hub(self, words: list[str]) -> np.ndarray:
        missing = [w for w in words if w not in self._hub]
        if missing:
            s = self.base.sims(self.reference, missing)            # (reference, missing)
            top = -np.sort(-s, axis=0)[: self.k]
            for w, h in zip(missing, top.mean(axis=0)):
                self._hub[w] = float(h)
        return np.array([self._hub[w] for w in words])

    def sims(self, clues: list[str] | None, words: list[str]) -> np.ndarray:
        return self.base.sims(clues, words) - self.hub(words)[None, :]
