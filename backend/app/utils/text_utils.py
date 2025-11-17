import re
from collections import Counter
from typing import Iterable, List

STOPWORDS = {
    "the", "a", "an", "to", "for", "of", "and", "in", "on",
    "this", "that", "with", "how", "what", "is", "are", "vs",
    "you", "your", "from", "new", "best", "top",
}


def _tokenize(text: str) -> List[str]:
    words = re.findall(r"[a-zA-Z0-9#+]+", (text or "").lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def extract_top_keywords(texts: Iterable[str], top_n: int = 5) -> List[str]:
    counter = Counter()
    for t in texts:
        counter.update(_tokenize(t))
    return [w for w, _ in counter.most_common(top_n)]
