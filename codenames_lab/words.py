"""Board words from NLTK: common concrete nouns (Brown frequency, filtered by WordNet)."""
from collections import Counter

from nltk.corpus import brown, wordnet as wn

CONCRETE = {"noun.artifact", "noun.animal", "noun.food", "noun.plant",
            "noun.object", "noun.body", "noun.substance"}


def usage(word: str, pos: str | None = None) -> int:
    """How often WordNet's tagged corpus used this word, in this part of speech."""
    return sum(l.count() for s in wn.synsets(word, pos=pos) for l in s.lemmas() if l.name() == word)


def concrete_noun(word: str) -> bool:
    senses = wn.synsets(word, pos=wn.NOUN)
    return (bool(senses) and wn.morphy(word, wn.NOUN) == word
            and senses[0].lexname() in CONCRETE
            and usage(word, wn.NOUN) > usage(word) / 2)


def nltk_pool(top: int = 400, vocab: set[str] | dict[str, int] | None = None) -> list[str]:
    """Common concrete nouns: Brown frequency order, WordNet filters.
    Pass `vocab` to keep only words a model knows, e.g. the word2vec index."""
    freq = Counter(w.lower() for w in brown.words() if w.isalpha() and len(w) >= 3)
    return [w for w, _ in freq.most_common()
            if (vocab is None or w in vocab) and concrete_noun(w)][:top]
