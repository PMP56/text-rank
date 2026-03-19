import spacy
import random
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
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

    def keyword(self, text, keyword_count=10, plot=False,
                epoch=None, damping_factor=None):
        """
        Extract keywords from the given text using the PageRank algorithm.

        Args:
            text (str): The input text from which to extract keywords.
            keyword_count (int): Number of top keywords to return.
            plot (bool): Whether to save a static graph image (legacy).
            epoch (int): Override instance epoch for this call.
            damping_factor (float): Override instance damping_factor for this call.

        Returns:
            list: A list of tuples (word, rank, vocab_index) for the top keywords.
            np.ndarray: The normalized adjacency matrix.
        """
        epoch          = epoch          if epoch          is not None else self.epoch
        damping_factor = damping_factor if damping_factor is not None else self.damping_factor

        doc = self.nlp(text)
        sentences = segment_sentences(doc, lower=True)

        vocab = Vocab(sentences)
        pairs = token_pair(sentences, window=5)
        norm_graph = build_matrix(vocab, pairs)

        ranks = np.array([1.0] * len(vocab))
        previous_rank = 0.0

        for i in range(epoch):
            ranks = (1 - damping_factor) + damping_factor * np.dot(norm_graph, ranks)
            sum_ranks = float(np.sum(ranks))
            if abs(previous_rank - sum_ranks) <= self.threshold:
                break
            previous_rank = sum_ranks

        res = [(word, ranks[idx], idx) for (word, idx) in vocab.stoi.items()]
        ordered = sorted(res, key=lambda x: x[1], reverse=True)[:keyword_count]

        if plot:
            self._plot_graph(ordered, norm_graph)

        return ordered, norm_graph

    def get_graph_data(self, text, keyword_count=10,
                       epoch=None, damping_factor=None):
        """
        Return keyword graph data as a JSON-serialisable dict for interactive rendering.

        Args:
            text (str): Input text.
            keyword_count (int): Number of top keywords.
            epoch (int): Override instance epoch for this call.
            damping_factor (float): Override instance damping_factor for this call.

        Returns:
            dict: { "nodes": [...], "edges": [...] }
                  nodes: { id, label, rank, size }
                  edges: { source, target, weight }
        """
        ordered, norm_graph = self.keyword(
            text, keyword_count=keyword_count,
            epoch=epoch, damping_factor=damping_factor
        )

        # Normalise ranks to [0,1] for sizing
        ranks_vals = [r for (_, r, _) in ordered]
        min_r = min(ranks_vals)
        max_r = max(ranks_vals)
        rng = max_r - min_r if max_r != min_r else 1.0

        nodes = []
        for word, rank, idx in ordered:
            nodes.append({
                "id": idx,
                "label": word,
                "rank": round(float(rank), 4),
                "size": round((rank - min_r) / rng, 4),   # 0–1
            })

        # Only look at edges between the top-K words — O(k²) not O(n²)
        output_indices = {idx for (_, _, idx) in ordered}
        edges = []
        seen = set()

        for _, _, i in ordered:
            for _, _, j in ordered:
                if i == j:
                    continue
                key = (min(i, j), max(i, j))
                if key in seen:
                    continue
                weight = float(norm_graph[i][j])
                if weight > 0:
                    edges.append({"source": i, "target": j, "weight": round(weight, 4)})
                    seen.add(key)

        return {"nodes": nodes, "edges": edges}

    def _plot_graph(self, output, matrix):
        """
        Save a static matplotlib keyword graph (legacy / kept for compatibility).

        Args:
            output (list): Top keywords as (word, rank, vocab_index).
            matrix (np.ndarray): Full normalised adjacency matrix.
        """
        G = nx.Graph()
        for word, rank, _ in output:
            G.add_node(word, rank=rank)

        # Efficient: only check pairs within the output set
        output_by_idx = {idx: word for (word, _, idx) in output}
        for (word_i, _, i) in output:
            for (word_j, _, j) in output:
                if i >= j:
                    continue
                if matrix[i][j] != 0:
                    G.add_edge(word_i, word_j)

        plt.figure(figsize=(8, 5))
        ranks = nx.get_node_attributes(G, 'rank')
        min_rank = min(ranks.values())
        max_rank = max(ranks.values())
        denom = max_rank - min_rank if max_rank != min_rank else 1.0

        node_sizes = [3000 * ((r - min_rank) / denom + 0.5) for r in ranks.values()]
        node_colors = [f'#{random.randint(0, 0xFFFFFF):06x}' for _ in G.nodes]

        nx.draw(
            G, with_labels=True,
            node_size=node_sizes, node_color=node_colors,
            font_size=8, font_weight='bold', edge_color='gray'
        )
        plt.title('TextRank Output Visualization')
        plt.savefig("app/static/graph.png")
        plt.close()

    def summarize(self, text, sentence_count=3):
        """
        Summarize the given text by extracting the top sentences.

        Args:
            text (str): The input text to summarize.
            sentence_count (int): Number of sentences in the summary.

        Returns:
            str: A summary of the input text.
        """
        doc = self.nlp(text)
        segments = segment_sentences(
            doc,
            pos=["NOUN", "PROPN", "PRN", "AUX", "VERB", "ADP", "NUM"],
            lower=True
        )

        w2v = train_w2v(segments)
        embeddings = [[w2v.wv[word][0] for word in segment] for segment in segments]

        max_length = max(len(e) for e in embeddings)
        embeddings_padded = [np.pad(e, (0, max_length - len(e))) for e in embeddings]

        similarity_matrix = get_similarity_matrix(embeddings_padded, len(segments))
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph)

        top_sentence = {
            sentence.text: scores[index]
            for index, sentence in enumerate(doc.sents)
        }
        top = dict(sorted(top_sentence.items(), key=lambda x: x[1], reverse=True)[:sentence_count])

        return " ".join(list(top.keys())[:sentence_count])