import itertools
import os
import pickle

import numpy as np


def _create_cache(embeddings_class, lookup_sentences):
    from ..data import DataSet, TrainingSet, ValidationSet, TestSet

    def sentences_from_dataset(dataset: DataSet):
        return [[token.text for token in sentence] for sentence in dataset.X]

    def words_from_dataset(dataset: DataSet):
        return list(itertools.chain.from_iterable(sentences_from_dataset(dataset)))

    read_dataset = sentences_from_dataset if lookup_sentences else words_from_dataset

    print('Loading the vocabulary...')
    tr = TrainingSet()
    vocab = read_dataset(tr)
    vocab += read_dataset(ValidationSet(validation_set='validation', embeddings=None, label2ind=tr.label2ind))
    vocab += read_dataset(ValidationSet(validation_set='test', embeddings=None, label2ind=tr.label2ind))

    print('Loading embeddings...')
    embeddings_class(vocab)
    print('Done.')


def create_fasttext_cache():
    from ..embeddings import CachedFastTextEmbeddings
    _create_cache(CachedFastTextEmbeddings, lookup_sentences=False)


def create_elmo_cache():
    from ..embeddings import CachedElmoEmbeddings
    _create_cache(CachedElmoEmbeddings, lookup_sentences=True)


def convert_precomputed_fasttext_embeddings():
    from ..embeddings.fasttext import fasttext_dir, fasttext_embeddings_name

    print('Loading precomputed embeddings...')
    vec_filename = os.path.join(fasttext_dir, fasttext_embeddings_name + '.vec')

    precomputed_vocab = np.loadtxt(vec_filename, usecols=0, dtype=object, skiprows=2, comments=None)
    precomputed_word2ind = {word: i for i, word in enumerate(precomputed_vocab)}

    # make sure there are no duplicate words
    assert len(precomputed_vocab) == len(precomputed_word2ind)

    precomputed_matrix = np.loadtxt(vec_filename, usecols=range(1, 301), skiprows=2, comments=None)

    print('L2 normalizing the embedding matrix...')
    normalized_matrix = precomputed_matrix / np.sqrt((precomputed_matrix ** 2).sum(-1))[..., np.newaxis]

    print('Saving the dictionary...')
    pickle.dump(precomputed_word2ind, open(vec_filename + '.vocab.pickle', 'wb'))
    print('Saving the matrix...')
    np.save(vec_filename + '.matrix.npy', normalized_matrix)
    print('Done.')


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--fasttext-cache', help='Initialize a fasttext embeddings cache with the i2b2 vocabulary',
                        action='store_true')
    parser.add_argument('--fasttext-precomputed', help='Convert precomputed fasttext embeddings to matrix/dict',
                        action='store_true')
    parser.add_argument('--elmo-cache', help='Initialize an elmo embeddings cache with the i2b2 vocabulary',
                        action='store_true')
    args = parser.parse_args()

    if not any([args.fasttext_cache, args.fasttext_precomputed, args.elmo_cache]):
        print('Specify at least one of --fasttext-cache, --fasttext-precomputed, --elmo-cache')

    if args.fasttext_cache:
        create_fasttext_cache()

    if args.elmo_cache:
        create_elmo_cache()

    if args.fasttext_precomputed:
        convert_precomputed_fasttext_embeddings()


if __name__ == '__main__':
    main()
