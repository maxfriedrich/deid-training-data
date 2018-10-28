import itertools


def get(identifier):
    if identifier == 'balanced':
        return balanced

    raise ValueError(f'Unknown class weight: {identifier}')


def balanced(output_size, y):
    y = list(itertools.chain.from_iterable([[label[0] for label in sent] for sent in y]))

    o_weight = len(y) / y.count(1)
    phi_weight = len(y) / (len(y) - y.count(1))

    class_weight = [0, o_weight]
    for i in range(2, output_size):
        class_weight.append(phi_weight)

    return class_weight
