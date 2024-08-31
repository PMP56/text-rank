import numpy as np

def symmetrize(x):
    return x + x.T - np.diag(x.diagonal())

def build_matrix(vocab, token_pairs):
    n = len(vocab)
    matrix = np.zeros([n, n])

    for pair in token_pairs:
        v1, v2 = pair
        v1_index, v2_index = vocab.lookup_token(v1), vocab.lookup_token(v2)
        matrix[v1_index, v2_index] = 1

    symmetric_matrix = symmetrize(matrix)

    sum = np.sum(symmetric_matrix, axis=0)
    norm_matrix = np.divide(symmetric_matrix, sum, where=sum!=0)

    return norm_matrix