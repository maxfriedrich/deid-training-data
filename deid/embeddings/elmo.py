import math
import os
import pickle
from typing import Sequence

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm

from . import Embeddings
from .util import pad_string_sequences, unpad_sequences, chunks
from ..env import env

elmo_dir = os.path.join(env.resources_dir, 'elmo')


class ElmoEmbeddings(Embeddings):
    def __new__(cls, *args, **kwargs):
        if env.embeddings_cache:
            return CachedElmoEmbeddings(*args, **kwargs)
        return TensorFlowElmoEmbeddings(*args, **kwargs)

    def __init__(self, *_, **__):
        raise NotImplementedError('this should not happen')

    @property
    def size(self) -> int:
        raise NotImplementedError

    @property
    def std(self):
        raise NotImplementedError

    def lookup(self, word: str) -> np.ndarray:
        raise NotImplementedError

    def is_unknown(self, word: str) -> bool:
        raise NotImplementedError


class ElmoEmbeddingsImpl(Embeddings):
    @property
    def size(self) -> int:
        return 1024

    @property
    def std(self) -> float:
        return 0.47

    def lookup(self, word: str) -> np.ndarray:
        raise RuntimeError("Don't lookup single words in ELMo")

    def is_unknown(self, word: str):
        return False


class TensorFlowElmoEmbeddings(ElmoEmbeddingsImpl):
    def __init__(self, *_, **__):
        graph = tf.Graph()
        with graph.as_default():
            self.tokens = tf.placeholder(tf.string, shape=[None, None])
            self.sequence_len = tf.placeholder(tf.int32, shape=[None])
            self.elmo = hub.Module('https://tfhub.dev/google/elmo/2')
            self.embed = self.elmo({'tokens': self.tokens, 'sequence_len': self.sequence_len}, signature='tokens',
                                   as_dict=True)
            init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
        graph.finalize()
        self.sess = tf.Session(graph=graph)
        self.sess.run(init_op)

    def lookup_sentence(self, words: Sequence[str]) -> Sequence[np.ndarray]:
        return self.sess.run(self.embed, {self.tokens: [words], self.sequence_len: [len(words)]})['elmo'][0]

    def lookup_sentences(self, sentences: Sequence[Sequence[str]]) -> Sequence[Sequence[np.ndarray]]:
        sentences, seq_length = pad_string_sequences(sentences)
        result = self.sess.run(self.embed, {self.tokens: sentences, self.sequence_len: seq_length})['elmo']
        return unpad_sequences(result, seq_length)


class CachedElmoEmbeddings(ElmoEmbeddingsImpl):
    def __init__(self, sentences=None, lookup_batch_size=64, *_, **__):
        if sentences is None:
            self.sent2vec = {}
            for chunk_name in [filename for filename in os.listdir(elmo_dir) if 'chunk' in filename]:
                self.sent2vec.update(pickle.load(open(os.path.join(elmo_dir, chunk_name), 'rb')))
        else:
            if not os.path.isdir(elmo_dir):
                os.mkdir(elmo_dir)

            embeddings = TensorFlowElmoEmbeddings()
            self.sent2vec = {}
            sentence_chunks = chunks(sentences, lookup_batch_size)
            for i, sentence_chunk in tqdm(enumerate(sentence_chunks), desc='Looking up sentence batches',
                                          total=math.ceil(len(sentences) / lookup_batch_size)):
                chunk_sent2vec = {}
                result = embeddings.lookup_sentences(sentence_chunk)
                for j, sentence in enumerate(sentence_chunk):
                    chunk_sent2vec[' '.join(sentence)] = result[j]
                    self.sent2vec[' '.join(sentence)] = result[j]
                chunk_filename = os.path.join(elmo_dir, f'elmo_chunk{i:04}.pickle')
                pickle.dump(chunk_sent2vec, open(chunk_filename, 'wb'))

    def lookup_sentence(self, words: Sequence[str]):
        result = self.sent2vec.get(' '.join(words))
        if result is not None:
            return result
        raise RuntimeError(f'Cache lookup failed for "{words}". Please rebuild the embedding cache.')
