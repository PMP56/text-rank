def segment_sentences(doc, pos=["NOUN", "PROPN"], lower=False):
    res = []
    for sentence in doc.sents:
        words = []
        for word in sentence:
            if word.pos_ in pos and not word.is_stop:
                words.append(word.text.lower() if lower else word.text)

        res.append(words)

    return res

def token_pair(sentence_segments, window=5):
    pairs = []
    for segment in sentence_segments:
        for i, token in enumerate(segment):
            for j in range(i+1, window):
                if (j >= len(segment)):
                    break
                pair = (token, segment[j])
                if pair not in pairs:
                    pairs.append(pair)

    return pairs