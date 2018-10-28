from deid.data.augment.strategy import RandomDigits, MoveToNeighbor
from ..data import TrainingSet
from deid.data.augment.augment import *
from ..embeddings import FastTextEmbeddings, TensorFlowElmoEmbeddings, Matrix

if __name__ == '__main__':
    emb = FastTextEmbeddings()
    matrix = Matrix(emb, precomputed_word2ind=emb.precomputed_word2ind, precomputed_matrix=emb.precomputed_matrix)
    augment = Augment(emb, strategy=MoveToNeighbor(matrix, 100), digit_strategy=None)
    TrainingSet(emb, train_set='train', augment=augment)
