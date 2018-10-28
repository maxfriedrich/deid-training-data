from ..data.augment import *
from ..data.batch import StratifiedSampling, fake_sentences_batch
from ..data.dataset import TrainingSet, is_phi_sentence
from ..embeddings import FastTextEmbeddings
from ..embeddings import Matrix

emb = FastTextEmbeddings()
matrix = Matrix(emb, precomputed_word2ind=emb.precomputed_word2ind, precomputed_matrix=emb.precomputed_matrix)
augment = Augment(emb, strategy=MoveToNeighbor(matrix, 100), digit_strategy=RandomDigits(matrix))
tr = TrainingSet(emb, augment=augment)
gen = StratifiedSampling(tr.X, tr.y, 32, lambda x, y: any(label[0] > 1 for label in y))
batch_X, batch_y, batch_ind = next(gen)
fake_sentences_batch(batch_X, batch_y, batch_ind, tr.augmented if tr.augmented is not None else {},
                     split_condition=is_phi_sentence)
