"""Small, inspectable BM25 retriever suitable for local demos and tests."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    source: str


@dataclass(frozen=True)
class Hit:
    document: Document
    score: float


class BM25Retriever:
    def __init__(self, documents: list[Document], k1: float = 1.5, b: float = 0.75) -> None:
        if not documents:
            raise ValueError("At least one document is required")
        self.documents = documents
        self.k1 = k1
        self.b = b
        self._tokens = [tokenize(doc.text) for doc in documents]
        self._frequencies = [Counter(tokens) for tokens in self._tokens]
        self._avg_length = sum(map(len, self._tokens)) / len(self._tokens)
        self._document_frequency = Counter(
            token for tokens in self._tokens for token in set(tokens)
        )

    def search(self, query: str, limit: int = 3) -> list[Hit]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scored: list[Hit] = []
        corpus_size = len(self.documents)
        for document, tokens, frequencies in zip(
            self.documents, self._tokens, self._frequencies, strict=True
        ):
            score = 0.0
            for token in query_tokens:
                frequency = frequencies[token]
                if not frequency:
                    continue
                docs_with_token = self._document_frequency[token]
                inverse_frequency = math.log(
                    1 + (corpus_size - docs_with_token + 0.5) / (docs_with_token + 0.5)
                )
                length_normalizer = frequency + self.k1 * (
                    1 - self.b + self.b * len(tokens) / self._avg_length
                )
                score += inverse_frequency * frequency * (self.k1 + 1) / length_normalizer
            if score > 0:
                scored.append(Hit(document=document, score=score))
        return sorted(scored, key=lambda hit: (-hit.score, hit.document.id))[:limit]

