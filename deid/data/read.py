import csv
import os
from typing import Sequence, Optional, Set

from .token import Token, HIPAA_TOKEN_TYPE, BINARY_LABEL
from ..env import env


def _add_data_dir_if_needed(path: str) -> str:
    """ Adds the data directory to a path if it's not already a sub-path.

    >>> _add_data_dir_if_needed('train') == os.path.join(env.data_dir, 'train')
    True

    :param path: the input path
    :return: a path containing the data directory
    """
    if os.path.realpath(env.data_dir) not in os.path.realpath(path):
        path = os.path.join(env.data_dir, path)
    return path


def full_text_for_csv(filename: str) -> str:
    """ Returns the full text for a csv file that is saved to a .txt file with the same stem name.

    :param filename: the csv filename
    :return: a string that is read from the corresponding txt file
    """
    filename = _add_data_dir_if_needed(filename)

    if not filename.endswith('.csv'):
        raise ValueError(f'{filename} is not a csv file')

    return open(filename[:-4] + '.txt').read()


def tokens_from_csv(file_or_dir: str,
                    limit: Optional[int] = None,
                    binary_classification: bool = False,
                    hipaa_only: bool = False) -> Sequence[Token]:
    """ Parses a directory of csv files or a single csv file for tokens.

    :param file_or_dir: the csv file or directory to parse
    :param limit: upper limit for the number of csv files to parse
    :param binary_classification: set to True to skip the classes and use only generic BIO labels
    :param hipaa_only: set to True to skip all non-HIPAA tags

    :return: a list of Token objects
    """

    def label_string(bio_string):
        if hipaa_only:
            if bio_string == 'O' or bio_string[2:] not in HIPAA_TOKEN_TYPE.keys():
                return 'O'

        if binary_classification:
            # Not really binary: there is still a B, I, and O label (and the padding label). I tried using true binary
            # labels and there was no real difference, so I'm deciding to keep it like this.
            return 'O' if bio_string == 'O' else f'{bio_string[0]}-{BINARY_LABEL}'
        return bio_string

    file_or_dir = _add_data_dir_if_needed(file_or_dir)

    if os.path.isdir(file_or_dir):
        filenames = sorted([os.path.join(file_or_dir, f) for f in os.listdir(file_or_dir) if f.endswith('.csv')])
        if len(filenames) == 0:
            raise ValueError(f'{file_or_dir} does not contain any csv files')
    elif file_or_dir.endswith('.csv'):
        filenames = [file_or_dir]
    else:
        raise ValueError(f'{file_or_dir} is not a csv file')

    tokens = []
    for i, filename in enumerate(filenames):
        with open(filename) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                tokens.append(Token(row[0],
                                    label_string(row[1]),
                                    *map(int, row[2:])))
        if i == limit:
            break

    return tokens


def split_sentences(tokens: Sequence[Token]) -> Sequence[Sequence[Token]]:
    """ Breaks a list of Token objects into sentence chunks delimited by sent_start and sent_end. Incomplete sentences
    are not included in the result.

    >>> len(split_sentences([Token.with_text(env.sent_start), Token.with_text('test'), Token.with_text(env.sent_end)]))
    1

    >>> len(split_sentences([Token.with_text(env.sent_start), Token.with_text('test')]))
    0

    :param tokens: the tokens to break into sentences
    :return: a list of sentences (i.e. a list of lists of tokens)
    """
    sents = []
    current_sent = []
    for token in tokens:
        if token.text not in [env.sent_start, env.sent_end]:
            current_sent.append(token)
        if token.text == env.sent_end:
            if len(current_sent) > 0:
                sents.append(current_sent)
            current_sent = []
    return sents


def vocab_from_tokens(tokens: Sequence[Token]) -> Set[str]:
    """ Returns a set of words from a token sequence, excluding the special sent_start and sent_end tokens.

    >>> sorted(vocab_from_tokens([Token.with_text(env.sent_start), Token.with_text('test'), \
    Token.with_text('some'), Token.with_text('test'), Token.with_text('words'), Token.with_text(env.sent_end)]))
    ['some', 'test', 'words']

    :param tokens: the tokens to convert to a vocabulary set
    :return: a set of words
    """
    return set(token.text for token in tokens) - {env.sent_start, env.sent_end}
