from ..embeddings import *

for i in range(2):
    if i == 0:
        emb = FastTextEmbeddings()
    else:
        emb = GloveEmbeddings()
    print(emb)

    mat = Matrix(emb, precomputed_matrix=emb.precomputed_matrix, precomputed_word2ind=emb.precomputed_word2ind)
    for word in ['Smith', 'Fox', 'Wolf']:
        print(', '.join([sim.word for sim in mat.most_similar_cosine(word, n=100)]))

