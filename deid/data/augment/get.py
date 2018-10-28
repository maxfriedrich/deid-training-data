from typing import Optional

from .strategy import AugmentStrategy, Zeros, RandomEmbedding, RandomDigits, AdditiveNoise, MoveToNeighbor


def get(identifier: Optional[str], *args, **kwargs) -> Optional[AugmentStrategy]:
    if identifier is None:
        return None
    elif identifier == 'zeros':
        return Zeros()
    elif identifier.startswith('random_embedding'):
        if '-' in identifier:
            scale = float(identifier.split('-')[1])
            return RandomEmbedding(scale, l2_normalize='l2' in identifier)
        else:
            return RandomEmbedding()
    elif identifier == 'random_digits':
        return RandomDigits(*args, **kwargs)
    elif identifier.startswith('additive_noise'):
        scale = float(identifier.split('-')[1])
        return AdditiveNoise(scale)
    elif identifier.startswith('move_to_neighbor'):
        n_neighbors = int(identifier.split('-')[1])
        return MoveToNeighbor(n_neighbors=n_neighbors, *args, **kwargs)  # type: ignore
    else:
        raise ValueError('unknown identifier:', identifier)
