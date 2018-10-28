from ..embeddings import TensorFlowElmoEmbeddings

emb = TensorFlowElmoEmbeddings()
result = emb.lookup_sentence(['what', 'is', 'up'])
