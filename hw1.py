from collections import Counter, defaultdict

from typing import Iterable, TypeVar, Sequence

# DO NOT MODIFY
T = TypeVar("T")

# DO NOT MODIFY
START_TOKEN = "<START>"
# DO NOT MODIFY
END_TOKEN = "<END>"


def counts_to_probs(counts: Counter[T]) -> dict[T, float]:
    summ = counts.total()
    dict_counts = dict(counts)
    dict_with_prob = {key: float(val / summ) for key, val in dict_counts.items()}
    return dict_with_prob


def bigrams(sentence: Sequence[str]) -> list[tuple[str, str]]:
    res = padding_helper(2, sentence)
    for i in range(len(sentence)):
        item = []
        item.append(sentence[i])
        if i == len(sentence) - 1:
            item.append(END_TOKEN)
        else:
            item.append(sentence[i + 1])
        res.append(tuple(item))

    return res


def trigrams(sentence: Sequence[str]) -> list[tuple[str, str, str]]:
    res = padding_helper(3, sentence)
    for i in range(len(sentence) - 1):
        item = [sentence[i], sentence[i + 1]]
        if i == len(sentence) - 2:
            item.append(END_TOKEN)
        else:
            item.append(sentence[i + 2])
        res.append(tuple(item))

    return res


def count_unigrams(
    sentences: Iterable[Sequence[str]], lower: bool = False
) -> Counter[str]:
    sentence_copy = [sentence for sentence in sentences]
    if lower:
        sentence_copy = [[word.lower() for word in sentence] for sentence in sentence_copy]
    c = Counter()
    for sentence in sentence_copy:
        c.update(sentence)

    return c


def count_bigrams(
    sentences: Iterable[Sequence[str]], lower: bool = False
) -> Counter[tuple[str, str]]:
    sentence_copy = [sentence for sentence in sentences]
    if lower:
        sentence_copy = [word.lower() for sentence in sentence_copy for word in sentence]
    c = Counter()
    for sentence in sentence_copy:
        c.update(bigrams(sentence))

    return c


def count_trigrams(
    sentences: Iterable[Sequence[str]], lower: bool = False
) -> Counter[tuple[str, str, str]]:
    sentence_copy = [sentence for sentence in sentences]
    if lower:
        sentence_copy = [word.lower() for sentence in sentence_copy for word in sentence]
    c = Counter()
    for sentence in sentence_copy:
        c.update(trigrams(sentence))

    return c


def bigram_probs(sentences: Iterable[Sequence[str]]) -> dict[str, dict[str, float]]:
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


def padding_helper(ngram_size: int, sentence: Sequence[str]) -> list:
    res = []
    n = ngram_size

    while n > 1:
        n -= 1
        curr_tuple = []
        for i in range(n):
            curr_tuple.append(START_TOKEN)
        for i in range(ngram_size - n):
            curr_tuple.append(sentence[i])
        res.append(tuple(curr_tuple))
    return res