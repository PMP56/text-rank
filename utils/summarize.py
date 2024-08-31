import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

def train_w2v(segments):
    return Word2Vec(segments, vector_size=1, min_count=1)

def get_similarity_matrix(embeddings, size):
    similarity_matrix = np.zeros([size, size])
    
    for i, row_embedding in enumerate(embeddings):
        for j, column_embedding in enumerate(embeddings):
            row_embedding = np.array(row_embedding).reshape(1, -1)
            column_embedding = np.array(column_embedding).reshape(1, -1)
            similarity_matrix[i][j]= 1 - cosine_similarity(row_embedding, column_embedding)[0,0]

    return similarity_matrix