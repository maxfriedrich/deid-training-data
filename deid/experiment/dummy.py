import argparse
import collections
from typing import Sequence
import random

import numpy as np
from keras.preprocessing.sequence import pad_sequences

from .directory import experiment_directory
from .evaluation import evaluate_deid_performance
from ..data import TrainingSet, ValidationSet, Token
from ..env import env


class DummyDeidentifier:
    def guess(self, sentence: Sequence[str]):
        raise NotImplementedError

    def predict(self, X, **_):
        if len(X) == 2 and isinstance(X[0][0], list) and isinstance(X[0][0][0], Token):  # extra features provided
            X, _ = X
        y = [self.guess([token.text for token in sentence]) for sentence in X]
        y = pad_sequences(y)
        return y


class UpperBoundDeidentifier(DummyDeidentifier):
    def __init__(self, X, y):
        self.solutions = {}
        for sentence, labels in zip(X, y):
            self.solutions[' '.join([token.text for token in sentence])] = [l[0] for l in labels]

    def guess(self, sentence):
        return self.solutions[' '.join(sentence)]


class RandomGuessingDeidentifier(DummyDeidentifier):
    def __init__(self, X, y):
        label_counts = collections.defaultdict(int)
        for sentence, labels in zip(X, y):
            for label in labels:
                label_counts[label[0]] += 1
        n_labels = sum(label_counts.values())
        self.labels = sorted(label_counts.keys())
        self.probabilities = [label_counts[label] / n_labels for label in self.labels]

    def guess(self, sentence):
        return np.random.choice(self.labels, size=len(sentence), p=self.probabilities)


class WordListDeidentifier(DummyDeidentifier):
    def __init__(self, X, y):
        self.memory = collections.defaultdict(lambda: [1])
        for sentence, labels in zip(X, y):
            for word, label in zip(sentence, labels):
                self.memory[word.text].append(label[0])

    def guess(self, sentence):
        def most_common(lst):
            return max(set(lst), key=lst.count)

        return [most_common(self.memory[word]) for word in sentence]


def main():
    parser = argparse.ArgumentParser()
    parser.description = 'different dummy predictors'
    parser.add_argument('--upper-bound', help='the embeddings to use, either glove or fasttext', action='store_true')
    parser.add_argument('--random-guessing', help='the embeddings to use, either glove or fasttext',
                        action='store_true')
    parser.add_argument('--word-list', help='the embeddings to use, either glove or fasttext', action='store_true')
    args = parser.parse_args()

    if not any([args.upper_bound, args.random_guessing, args.word_list]):
        raise ValueError('please select at least one of --upper-bound, --random-guessing, --word-list')

    tr = TrainingSet(limit_documents=env.limit_training_documents)
    val = ValidationSet(tr.label2ind, limit_documents=env.limit_training_documents, validation_set='validation')

    if args.upper_bound:
        # needs its own special case because the model is initialized with the test set!
        test = ValidationSet(tr.label2ind, limit_documents=env.limit_training_documents, validation_set='test')
        experiment_dir = experiment_directory('upper_bound')
        model = UpperBoundDeidentifier(test.X, test.y)
        evaluate_deid_performance(model, embeddings=None, test_set='test', label2ind=tr.label2ind,
                                  ind2label=tr.ind2label,
                                  batch_size=8, experiment_dir=experiment_dir, require_argmax=False)

    if args.random_guessing:
        test_baseline('random_guessing', RandomGuessingDeidentifier, tr, val)

    if args.word_list:
        test_baseline('word_list', WordListDeidentifier, tr, val)


def test_baseline(identifier, model_class, tr, val):
    experiment_dir = experiment_directory(identifier)
    model = model_class(tr.X + val.X, tr.y + val.y)
    evaluate_deid_performance(model, embeddings=None, test_set='test', label2ind=tr.label2ind,
                              ind2label=tr.ind2label, batch_size=8, experiment_dir=experiment_dir,
                              require_argmax=False)


if __name__ == '__main__':
    main()
