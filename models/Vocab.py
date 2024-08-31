import numpy as np

class Vocab:
    def __init__(self, sentence_segments):
        self.stoi = {}
        self.itos = {}
        np_segments = list(set(np.concatenate(sentence_segments)))
        for i, word in enumerate(np_segments):
            self.stoi[word] = i
            self.itos[i] = word        

    def __len__(self):
        return len(self.stoi)
    
    def lookup_token(self, token):
        return self.stoi[token]

    def lookup_index(self, idx):
        return self.itos[idx]
