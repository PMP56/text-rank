import random
import spacy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from models.Vocab import Vocab
from utils.sentence import segment_sentences, token_pair
from utils.adjacency_matrix import build_matrix

from utils.summarize import train_w2v, get_similarity_matrix

class TextRank:
    def __init__(self, epoch=10, damping_factor=0.85, threshold=0.001):
        """
        TextRank algorithm for keyword extraction and summarization.

        Args:
            epoch (int): Number of iterations for the PageRank algorithm.
            damping_factor (float): Damping factor for the PageRank algorithm.
            threshold (float): Convergence threshold for the PageRank algorithm.
        """
        self.epoch = epoch
        self.damping_factor = damping_factor
        self.threshold = threshold
        self.nlp = spacy.load("en_core_web_sm")
    
    def keyword(self, text, keyword_count=10,
                 plot=False):
        """
        Extract keywords from the given text using the PageRank algorithm and optionally plot the graph.

        Args:
            text (str): The input text from which to extract keywords.
            plot (bool): Whether to plot the keyword graph.

        Returns:
            list: A list of tuples containing keywords and their ranks.
            np.ndarray: The normalized graph as an adjacency matrix.
        """
        doc = self.nlp(text)
        sentences = segment_sentences(doc, lower=True)

        vocab = Vocab(sentences)
        pairs = token_pair(sentences, window=5)
        norm_graph = build_matrix(vocab, pairs)

        ranks = np.array([1] * len(vocab))
        previous_rank = 0
        
        for i in range(self.epoch):
            print(f"Epoch: {i}")
            ranks = (1 - self.damping_factor) + self.damping_factor * np.dot(norm_graph, ranks)
            sum_ranks = np.sum(ranks)

            if abs(previous_rank - sum_ranks) <= self.threshold:
                break
            else:
                previous_rank = sum_ranks

        res = [(word, ranks[i], i) for (word, i) in vocab.stoi.items()]
        ordered = sorted(res, key=lambda x:x[1], reverse=True)[:keyword_count]
        
        if plot:
            self._plot_graph(ordered, norm_graph)

        return ordered, norm_graph
    
    def _plot_graph(self, output, matrix):
        """
        Visualize the keywords and their relationships as a graph (private method).

        Args:
            output (list): A list of tuples containing keywords and their ranks.
            matrix (np.ndarray): The adjacency matrix representing relationships between keywords.
        """
        G = nx.Graph()
        for word, rank, _ in output:
            G.add_node(word, rank=rank, label=f'{word}')

        for i, segment in enumerate(matrix):
            for j, word in enumerate(segment):
                if word != 0:
                    from_word = [o[0] for o in output if o[2] == i]
                    to_word = [o[0] for o in output if o[2] == j]

                    if from_word and to_word:
                        G.add_edge(from_word[0], to_word[0])
        
        plt.figure(figsize=(8, 5))

        ranks = nx.get_node_attributes(G, 'rank')
        min_rank, max_rank = min(ranks.values()), max(ranks.values())
        
        node_sizes = [3000 * ((rank - min_rank) / (max_rank - min_rank) + 0.5) for rank in ranks.values()]
        node_colors = [f'#{random.randint(0, 0xFFFFFF):06x}' for _ in G.nodes]
        
        nx.draw(G, with_labels=True, node_size=node_sizes, node_color=node_colors, font_size=8, font_weight='bold', edge_color='gray')
        
        plt.title('TextRank Output Visualization')
        plt.savefig("app/static/graph.png")
        # plt.show()

    def summarize(self, text, sentence_count=3):
        """
        Summarize the given text by extracting the top sentences.

        Args:
            text (str): The input text to summarize.
            sentence_count (int): The number of sentences to include in the summary.

        Returns:
            str: A summary of the input text.
        """
        doc = self.nlp(text)
        segments = segment_sentences(doc, pos=["NOUN", "PROPN", "PRN", "AUX", "VERB", "ADP", "NUM"], lower=True)

        w2v = train_w2v(segments)
        embeddings = [[w2v.wv[word][0] for word in segment] for segment in segments]
        
        max_length = max([len(emb) for emb in embeddings])
        embeddings_padded = [np.pad(emb, (0, max_length - len(emb))) for emb in embeddings]

        similarity_matrix = get_similarity_matrix(embeddings_padded, len(segments))
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph)
        
        top_sentence = {sentence.text: scores[index] for index, sentence in enumerate(doc.sents)}
        top = dict(sorted(top_sentence.items(), key=lambda x: x[1], reverse=True)[:sentence_count])

        return " ".join(list(top.keys())[:sentence_count])