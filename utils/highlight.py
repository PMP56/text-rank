import re

def highlight_keywords(text: str, keywords: list) -> str:
    
    if not keywords:
        return text

    ranks = [rank for (_, rank, _) in keywords]
    min_r = min(ranks)
    max_r = max(ranks)
    rng   = max_r - min_r if max_r != min_r else 1.0

    def strength(rank):
        norm = (rank - min_r) / rng   # 0 → 1
        if norm >= 0.66:
            return 'high'
        elif norm >= 0.33:
            return 'mid'
        return 'low'

    # Sort longest keyword first so multi-word substrings don't get
    # partially matched before longer ones (future-proofing).
    sorted_kw = sorted(keywords, key=lambda x: len(x[0]), reverse=True)

    patterns = [re.escape(word) for (word, _, _) in sorted_kw]
    combined = re.compile(
        r'(' + '|'.join(r'\b' + p + r'\b' for p in patterns) + r')',
        re.IGNORECASE
    )

    rank_map = {word.lower(): (rank, strength(rank)) for (word, rank, _) in keywords}

    parts = combined.split(text)
    html_parts = []

    for part in parts:
        key = part.lower()
        if key in rank_map:
            rank, cls = rank_map[key]
            html_parts.append(
                f'<mark class="kw kw-{cls}" data-rank="{rank:.4f}">{_escape(part)}</mark>'
            )
        else:
            html_parts.append(_escape(part))

    return ''.join(html_parts)


def _escape(s: str) -> str:
    """Minimal HTML escaping for plain text segments."""
    return (
        s.replace('&', '&amp;')
         .replace('<', '&lt;')
         .replace('>', '&gt;')
    )