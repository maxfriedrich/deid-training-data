import argparse
import itertools
import random
import numpy as np

from ..data import TrainingSet
from ..embeddings import FastTextEmbeddings, GloveEmbeddings, Matrix
from ..env import env


def top_perc(n, ranks):
    return len([rank for rank in ranks if rank <= n]) / len(ranks)


def sim_perc(threshold, similarities):
    return len([sim for sim in similarities if sim >= threshold]) / len(similarities)


def main():
    parser = argparse.ArgumentParser()
    parser.description = 'try different amounts of noise to find a balance'
    parser.add_argument('embeddings', type=str, help='the embeddings to use, either glove or fasttext')
    parser.add_argument('noises', nargs='+', type=float, help='the noises to try')
    args = parser.parse_args()

    noises = args.noises
    if len(noises) == 0:
        raise argparse.ArgumentTypeError('Please provide a list of noises')

    if args.embeddings == 'fasttext':
        emb = FastTextEmbeddings()
        lower = False
    elif args.embeddings == 'glove':
        emb = GloveEmbeddings()
        lower = True
    else:
        raise argparse.ArgumentTypeError(f'Unknown embeddings: {args.embeddings}')

    mat = Matrix(lookup_embeddings=emb, precomputed_word2ind=emb.precomputed_word2ind,
                 precomputed_matrix=emb.precomputed_matrix)

    tr = TrainingSet(limit_documents=env.limit_training_documents)

    phi_tokens = set([token.text for token in itertools.chain.from_iterable(tr.X) if token.type != 'O'])
    phi_tokens = [word.lower() if lower else word for word in phi_tokens
                  if len(word) > 2
                  and (word.lower() if lower else word) in emb.precomputed_word2ind.keys()
                  and not any([c.isdigit() for c in word])]

    tokens_to_check = random.sample(phi_tokens, 1_000)
    # print(tokens_to_check)

    print('Similarity to closest neighbors:')
    closest_neighbor_similarities = []
    for token in random.sample(tokens_to_check, 10):
        closest_neighbor_similarities.append(mat.most_similar_cosine(token, n=2)[1].similarity)

    print(f'closest neighbor similarity mean: {np.mean(closest_neighbor_similarities)}',
          f'std: {np.std(closest_neighbor_similarities)}')

    for noise in noises:
        ranks = []
        similarities = []
        closest_neighbor_similarities = []

        for token in tokens_to_check:
            looked_up = emb.lookup(token)
            noisy = looked_up + np.random.normal(0., noise, emb.size)
            ranks.append(mat.cosine_distance_rank(noisy, token))
            similarities.append(mat.cosine_distance(noisy, token))
            closest_neighbor_similarities.append(mat.most_similar_cosine(noisy, n=1)[0].similarity)

        print('---')
        print(f'Report for scale {noise}:')
        print(f'rank mean: {np.mean(ranks)},',
              f'std: {np.std(ranks)},',
              f'%top1: {top_perc(1, ranks)},',
              f'%top5: {top_perc(5, ranks)},',
              f'%top10: {top_perc(10, ranks)}')
        print(f'similarity with original mean: {np.mean(similarities)}',
              f'std: {np.std(similarities)}',
              f'%0.9+: {sim_perc(0.9, similarities)}',
              f'%0.8+: {sim_perc(0.8, similarities)}',
              f'%0.7+: {sim_perc(0.7, similarities)}',
              f'%0.6+: {sim_perc(0.6, similarities)}')
        print(f'closest neighbor similarity mean: {np.mean(closest_neighbor_similarities)}',
              f'std: {np.std(closest_neighbor_similarities)}',
              f'%0.9+: {sim_perc(0.9, closest_neighbor_similarities)}',
              f'%0.8+: {sim_perc(0.8, closest_neighbor_similarities)}',
              f'%0.7+: {sim_perc(0.7, closest_neighbor_similarities)}',
              f'%0.6+: {sim_perc(0.6, closest_neighbor_similarities)}')

        print()


if __name__ == '__main__':
    main()
