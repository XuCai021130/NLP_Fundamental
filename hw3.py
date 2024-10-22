from collections import defaultdict, Counter
from math import log
from multiprocessing.managers import Value
from typing import (
    Iterable,
    Sequence,
)

# Version 1.0.0
# 10/11/2024

############################################################
# The following classes and methods are provided as helpers.
# Do not modify them! The stubs for what you need to implement are later in the file.

# DO NOT MODIFY
START_TOKEN = "<START>"
# DO NOT MODIFY
END_TOKEN = "<END>"


# DO NOT MODIFY
class AirlineSentimentInstance:
    """Represents a single instance from the airline sentiment dataset.

    Each instance contains the sentiment label, the name of the airline,
    and the sentences of text. The sentences are stored as a tuple of
    tuples of strings. The outer tuple represents sentences, and each
    sentences is a tuple of tokens."""

    def __init__(
        self, label: str, airline: str, sentences: Sequence[Sequence[str]]
    ) -> None:
        self.label: str = label
        self.airline: str = airline
        # These are converted to tuples so they cannot be modified
        self.sentences: tuple[tuple[str, ...], ...] = tuple(
            tuple(sentence) for sentence in sentences
        )

    def __repr__(self) -> str:
        return f"<AirlineSentimentInstance: {str(self)}>"

    def __str__(self) -> str:
        return f"label={self.label}; airline={self.airline}; sentences={self.sentences}"


# DO NOT MODIFY
class SentenceSplitInstance:
    """Represents a potential sentence boundary in context.

    Each instance is labeled with whether it is ('y') or is not ('n') a sentence
    boundary, the characters to the left of the boundary token, the potential
    boundary token itself (punctuation that could be a sentence boundary), and
    the characters to the right of the boundary token."""

    def __init__(
        self, label: str, left_context: str, token: str, right_context: str
    ) -> None:
        self.label: str = label
        self.left_context: str = left_context
        self.token: str = token
        self.right_context: str = right_context

    def __repr__(self) -> str:
        return f"<SentenceSplitInstance: {str(self)}>"

    def __str__(self) -> str:
        return " ".join(
            [
                f"label={self.label};",
                f"left_context={repr(self.left_context)};",
                f"token={repr(self.token)};",
                f"right_context={repr(self.right_context)}",
            ]
        )


# DO NOT MODIFY
class ClassificationInstance:
    """Represents a label and features for classification."""

    def __init__(self, label: str, features: Iterable[str]) -> None:
        self.label: str = label
        # Features can be passed in as any iterable and they will be
        # stored in a tuple
        self.features: tuple[str, ...] = tuple(features)

    def __repr__(self) -> str:
        return f"<ClassificationInstance: {str(self)}>"

    def __str__(self) -> str:
        return f"label={self.label}; features={self.features}"


############################################################
# The stubs below this are the ones that you should fill in.
# Do not modify anything above this line other than to add any needed imports.

def calculate_confusion_metrics(
    predictions: Sequence[str], expected: Sequence[str], positive_label: str
) -> tuple[float, float, float, float]:
    if len(predictions) != len(expected) or len(predictions) == len(expected) == 0:
        raise ValueError("Predictions and expected sequences must be of the same length.")
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for predict, expect in zip(predictions, expected):
        if predict == positive_label:
            if expect == positive_label:
                tp += 1
            else:
                fp += 1
        else:
            if expect == positive_label:
                fn += 1
            else:
                tn += 1

    return tp, tn, fp, fn


def accuracy(predictions: Sequence[str], expected: Sequence[str]) -> float:
    if len(predictions) != len(expected) or len(predictions) == len(expected) == 0:
        raise ValueError("Predictions and expected sequences must be of the same length.")

    correct_prediction = sum(predict == expect for predict, expect in zip(predictions, expected))
    all_prediction = len(predictions)
    if all_prediction == 0:  # handle zero as denominator error
        return 0.0
    else:
        return correct_prediction / all_prediction


def recall(
    predictions: Sequence[str], expected: Sequence[str], positive_label: str
) -> float:
    tp, tn, fp, fn = calculate_confusion_metrics(predictions, expected, positive_label)
    if (tp + fn) == 0:
        return 0.0
    else:
        return tp / (tp + fn)


def precision(
    predictions: Sequence[str], expected: Sequence[str], positive_label: str
) -> float:
    tp, tn, fp, fn = calculate_confusion_metrics(predictions, expected, positive_label)
    if (tp + fp) == 0:
        return 0.0
    else:
        return tp / (tp + fp)


def f1(predictions: Sequence[str], expected: Sequence[str], positive_label: str) -> float:
    """Compute the F1-score of the provided predictions."""
    p = precision(predictions, expected, positive_label)
    r = recall(predictions, expected, positive_label)
    if (p + r) == 0:
        return 0.0
    else:
        return 2 * p * r / (p + r)


def bigrams(sentence: Sequence[str]) -> list[tuple[str, str]]: # helper function to generate normal bi-gram
    sentence = (START_TOKEN,) + tuple(sentence) + (END_TOKEN,)
    res = list()

    for i in range(len(sentence) - 1):
        res.append(tuple([sentence[i], sentence[i + 1]]))
    return res


class UnigramAirlineSentimentFeatureExtractor:
    @staticmethod
    def extract_features(instance: AirlineSentimentInstance) -> ClassificationInstance:
        uniq_unigram = set(word.lower() for sentence in instance.sentences for word in sentence)
        return ClassificationInstance(instance.label, uniq_unigram)


class BigramAirlineSentimentFeatureExtractor:
    @staticmethod
    def extract_features(instance: AirlineSentimentInstance) -> ClassificationInstance:
        lowered_sentences = [[word.lower() for word in sentence] for sentence in instance.sentences]
        uniq_bigram = set(str(bigram) for sentence in lowered_sentences for bigram in bigrams(sentence))
        return ClassificationInstance(instance.label, uniq_bigram)


class BaselineSegmentationFeatureExtractor:
    @staticmethod
    def extract_features(instance: SentenceSplitInstance) -> ClassificationInstance:
        pass


class InstanceCounter:
    def __init__(self) -> None:
        pass

    def count_instances(self, instances: Iterable[ClassificationInstance]) -> None:
        # You should fill in this loop. Do not try to store the instances!
        for instance in instances:
            pass

    def label_count(self, label: str) -> int:
        pass

    def total_labels(self) -> int:
        pass

    def feature_label_joint_count(self, feature: str, label: str) -> int:
        pass

    def unique_labels(self) -> list[str]:
        pass

    def feature_vocab_size(self) -> int:
        pass

    def feature_set(self) -> set[str]:
        pass

    def total_feature_count_for_label(self, label: str) -> int:
        pass


class NaiveBayesClassifier:
    # DO NOT MODIFY
    def __init__(self, k: float):
        self.k: float = k
        self.instance_counter: InstanceCounter = InstanceCounter()

    # DO NOT MODIFY
    def train(self, instances: Iterable[ClassificationInstance]) -> None:
        self.instance_counter.count_instances(instances)

    def prior_prob(self, label: str) -> float:
        pass

    def feature_prob(self, feature: str, label) -> float:
        pass

    def log_posterior_prob(self, features: Sequence[str], label: str) -> float:
        pass

    def classify(self, features: Sequence[str]) -> str:
        pass

    def test(
        self, instances: Iterable[ClassificationInstance]
    ) -> tuple[list[str], list[str]]:
        pass


# MODIFY THIS AND DO THE FOLLOWING:
# 1. Inherit from UnigramAirlineSentimentFeatureExtractor or BigramAirlineSentimentFeatureExtractor
#    (instead of object) to get an implementation for the extract_features method.
# 2. Set a value for self.k below based on your tuning experiments.
class TunedAirlineSentimentFeatureExtractor(object):
    def __init__(self) -> None:
        self.k = float("NaN")
