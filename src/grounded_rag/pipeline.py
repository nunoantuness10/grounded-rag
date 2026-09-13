"""RAG orchestration with sentence-level citations and abstention."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from .retrieval import BM25Retriever, Document, Hit, tokenize


class Generator(Protocol):
    def generate(self, question: str, evidence: list[Hit]) -> str: ...


class ExtractiveGenerator:
    """Offline baseline that selects the strongest evidence sentence."""

    def generate(self, question: str, evidence: list[Hit]) -> str:
        query_terms = set(tokenize(question))
        candidates: list[tuple[int, str, str]] = []
        for hit in evidence:
            for sentence in re.split(r"(?<=[.!?])\s+", hit.document.text):
                overlap = len(query_terms.intersection(tokenize(sentence)))
                candidates.append((overlap, sentence.strip(), hit.document.id))
        if not candidates:
            return "I do not have enough evidence to answer."
        _, sentence, document_id = max(candidates, key=lambda item: (item[0], len(item[1])))
        return f"{sentence} [{document_id}]"


@dataclass(frozen=True)
class Citation:
    document_id: str
    source: str
    score: float


@dataclass(frozen=True)
class RAGResponse:
    answer: str
    citations: tuple[Citation, ...]
    abstained: bool


class GroundedRAG:
    def __init__(
        self,
        documents: list[Document],
        generator: Generator | None = None,
        minimum_score: float = 0.15,
    ) -> None:
        self.retriever = BM25Retriever(documents)
        self.generator = generator or ExtractiveGenerator()
        self.minimum_score = minimum_score

    def ask(self, question: str, limit: int = 3) -> RAGResponse:
        evidence = self.retriever.search(question, limit=limit)
        if not evidence or evidence[0].score < self.minimum_score:
            return RAGResponse(
                answer="I do not have enough evidence to answer.", citations=(), abstained=True
            )
        answer = self.generator.generate(question, evidence)
        citations = tuple(
            Citation(hit.document.id, hit.document.source, round(hit.score, 4))
            for hit in evidence
        )
        return RAGResponse(answer=answer, citations=citations, abstained=False)

