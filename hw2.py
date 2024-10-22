import math
import random
from abc import abstractmethod
from collections import defaultdict, Counter
from math import log, prod
from pathlib import Path
from typing import Sequence, Iterable, Generator, TypeVar, Union, Generic, final

############################################################
# The following constants and function are provided as helpers.
# Do not modify them! The stubs for what you need to implement are later in the file.
# HW 2 stubs 1.0 10/1/2024

# DO NOT MODIFY
random.seed(0)

# DO NOT MODIFY
START_TOKEN = "<START>"
# DO NOT MODIFY
END_TOKEN = "<END>"
# DO NOT MODIFY
POS_INF = float("inf")
NEG_INF = float("-inf")
# DO NOT MODIFY (needed for copying code from HW 1)
T = TypeVar("T")


# DO NOT MODIFY
def load_tokenized_file(path: Union[Path, str]) -> Generator[Sequence[str], None, None]:
    """Yield sentences as sequences of tokens."""
    with open(path, encoding="utf8") as file:
        for line in file:
            line = line.rstrip("\n")
            tokens = line.split(" ")
            assert tokens, "Empty line in input"
            yield tuple(tokens)


# DO NOT MODIFY
def sample(probs: dict[str, float]) -> str:
    """Return a sample from a distribution."""
    # To avoid relying on the dictionary iteration order,
    # sort items before sampling. This is very slow and
    # should be avoided in general, but we do it in order
    # to get predictable results.
    items = sorted(probs.items())
    # Now split them back up into keys and values
    keys, vals = zip(*items)
    # Choose using the weights in the values
    return random.choices(keys, weights=vals)[0]


# DO NOT MODIFY
class ProbabilityDistribution(Generic[T]):
    """A generic probability distribution."""

    # DO NOT ADD AN __INIT__ METHOD HERE
    # You will implement this in subclasses

    # DO NOT MODIFY
    # You will implement this in subclasses
    @abstractmethod
    def prob(self, item: T) -> float:
        """Return a probability for the specified item."""
        raise NotImplementedError

############################################################
# The stubs below this are the ones that you should fill in.
# Do not modify anything above this line other than to add any needed imports.


def sample_bigrams(probs: dict[str, dict[str, float]]) -> list[str]:
    curr_token = START_TOKEN
    res = []
    while True:
        next_token = sample(probs[curr_token])
        if next_token == END_TOKEN:
            break
        else:
            res.append(next_token)
        curr_token = next_token

    return res


def sample_trigrams(probs: dict[tuple[str, str], dict[str, float]]) -> list[str]:
    curr_token = (START_TOKEN, START_TOKEN)
    res = []
    while True:
        next_token = sample(probs[curr_token])
        if next_token == END_TOKEN:
            break
        else:
            res.append(next_token)
        curr_token = (curr_token[1], next_token)

    return res


def unigram_counts(sentences: Iterable[Sequence[str]]) -> dict[str, int]:
    c = Counter()
    for sentence in sentences:
        c.update(sentence)

    return dict(c)


def bigram_counts(sentences: Iterable[Sequence[str]]) -> dict[str, dict[str, int]]:
    counting = defaultdict(Counter)

    for sentence in sentences:
        for first, second in bigrams(sentence):
            counting[first][second] += 1
    return {key: dict(counting[key]) for key in counting}


def trigram_counts(
    sentences: Iterable[Sequence[str]],
) -> dict[tuple[str, str], dict[str, int]]:
    counting = defaultdict(Counter)

    for sentence in sentences:
        for first, second, third in trigrams(sentence):
            counting[(first, second)][third] += 1
    return {key: dict(counting[key]) for key in counting}


class UnigramMLE(ProbabilityDistribution[str]):
    def __init__(self, probs: dict[str, int]) -> None:
        self.probs = probs
        self.total = Counter(probs).total()

    def prob(self, item: str) -> float:
        if self.probs.get(item, 0) == 0:
            return 0.0
        else:
            return self.probs[item] / self.total


class BigramMLE(ProbabilityDistribution[tuple[str, str]]):
    def __init__(self, probs: dict[str, dict[str, int]]) -> None:
        self.probs = probs
        self.total_dict = {key: Counter(probs[key]).total()  for key in probs}

    def prob(self, item: tuple[str, str]) -> float:
        first_token = item[0]
        second_token = item[1]
        if first_token not in self.probs:
            return 0.0
        else:
            if self.probs[first_token].get(second_token, 0) == 0:
                return 0.0
            else:
                return self.probs[first_token][second_token] / self.total_dict[first_token]


class TrigramMLE(ProbabilityDistribution[tuple[str, str, str]]):
    def __init__(self, probs: dict[tuple[str, str], dict[str, int]]) -> None:
        self.probs = probs
        self.total_dict = {key: Counter(probs[key]).total() for key in probs}

    def prob(self, item: tuple[str, str, str]) -> float:
        given_token = tuple([item[0], item[1]])
        actual = item[2]
        if given_token not in self.probs:
            return 0.0
        else:
            if self.probs[given_token].get(actual, 0) == 0:
                return 0.0
            else:
                return self.probs[given_token][actual] / self.total_dict[given_token]


class UnigramLidstoneSmoothed(ProbabilityDistribution[str]):
    def __init__(self, counts: dict[str, int], k: float) -> None:
        self.k = k
        self.counts = counts
        self.total = len(self.counts) * k + sum([self.counts[key] for key in counts])


    def prob(self, item: str) -> float:
        if item not in self.counts:
            return 0.0
        else:
            return (self.counts[item] + self.k) / self.total


class BigramInterpolation(ProbabilityDistribution[tuple[str, str]]):
    def __init__(
        self,
        uni_probs: ProbabilityDistribution[str],
        bi_probs: ProbabilityDistribution[tuple[str, str]],
        l_1: float,
        l_2: float,
    ) -> None:
        self.uni_probs = uni_probs
        self.bi_probs = bi_probs
        self.l_1 = l_1
        self.l_2 = l_2

    def prob(self, item: tuple[str, str]) -> float:
        return self.l_1 * self.uni_probs.prob(item[1]) + self.l_2 * self.bi_probs.prob(item)


def unigram_sequence_prob(
    sequence: Sequence[str], probs: ProbabilityDistribution[str]
) -> float:
    result = 0
    for token in sequence:
        result += math.log(probs.prob(token))

    return result



def bigram_sequence_prob(
    sequence: Sequence[str], probs: ProbabilityDistribution[tuple[str, str]]
) -> float:
    result = 0
    for bigram in bigrams(sequence):
        result += math.log(probs.prob(bigram))
    if result == 0.0:
        return NEG_INF
    else:
        return result


def trigram_sequence_prob(
    sequence: Sequence[str], probs: ProbabilityDistribution[tuple[str, str, str]]
) -> float:
    result = 0
    for trigram in trigrams(sequence):
        result += math.log(probs.prob(trigram))
    if result == 0.0:
        return NEG_INF
    else:
        return result


def unigram_perplexity(
    sentences: Iterable[Sequence[str]], probs: ProbabilityDistribution[str]
) -> float:
    final_prob = 1
    tokens = 0
    for sentence in sentences:
        for word in sentence:
            curr_prob = probs.prob(word)
            if curr_prob == 0:
                return POS_INF
            final_prob *= curr_prob
        tokens += len(sentence)

    return final_prob ** (-1/tokens)


def bigram_perplexity(
    sentences: Iterable[Sequence[str]], probs: ProbabilityDistribution[tuple[str, str]]
) -> float:
    final_prob = 1
    tokens = 0
    for sentence in sentences:
        for bigram in bigrams(sentence):
            curr_prob = probs.prob(bigram)
            if curr_prob == 0:
                return POS_INF
            final_prob *= curr_prob
            tokens += 1

    return final_prob ** (-1/tokens)


def trigram_perplexity(
    sentences: Iterable[Sequence[str]],
    probs: ProbabilityDistribution[tuple[str, str, str]],
) -> float:
    final_prob = 1
    tokens = 0
    for sentence in sentences:
        for trigram in trigrams(sentence):
            curr_prob = probs.prob(trigram)
            if curr_prob == 0:
                return POS_INF
            final_prob *= curr_prob
            tokens += 1

    return final_prob ** (-1/tokens)


############################################################
# From HW 1


def bigrams(sentence: Sequence[str]) -> list[tuple[str, str]]:
    sentence = (START_TOKEN,) + tuple(sentence) + (END_TOKEN,)
    res = list()

    for i in range(len(sentence) - 1):
        res.append(tuple([sentence[i], sentence[i + 1]]))
    return res


def trigrams(sentence: Sequence[str]) -> list[tuple[str, str, str]]:
    sentence = (START_TOKEN,) + (START_TOKEN,) + tuple(sentence) + (END_TOKEN,)
    res = list()

    for i in range(len(sentence) - 2):
        res.append(tuple([sentence[i], sentence[i + 1], sentence[i + 2]]))
    return res


# The following three functions should only be used to test your sampler


def bigram_probs(
    sentences: Iterable[Sequence[str]],
) -> dict[str, dict[str, float]]:
    bigram_counts = defaultdict(Counter)

    for sentence in sentences:
        for first, second in bigrams(sentence):
            bigram_counts[first][second] += 1

    bigram_counts = {key: counts_to_probs(bigram_counts[key]) for key in bigram_counts}
    return dict(bigram_counts)


def trigram_probs(
    sentences: Iterable[Sequence[str]],
) -> dict[tuple[str, str], dict[str, float]]:
    trigram_counts = defaultdict(Counter)

    for sentence in sentences:
        for first, second, third in trigrams(sentence):
            trigram_counts[(first, second)][third] += 1

    trigram_counts = {key: counts_to_probs(trigram_counts[key]) for key in trigram_counts}

    return dict(trigram_counts)


def counts_to_probs(counts: Counter[T]) -> dict[T, float]:
    summ = counts.total()
    dict_counts = dict(counts)
    dict_with_prob = {key: float(val / summ) for key, val in dict_counts.items()}
    return dict_with_prob



