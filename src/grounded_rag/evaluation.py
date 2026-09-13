"""Reproducible retrieval metrics for a small labelled question set."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .retrieval import BM25Retriever


@dataclass(frozen=True)
class EvaluationResult:
    recall_at_k: float
    mean_reciprocal_rank: float
    queries: int


def evaluate(retriever: BM25Retriever, dataset_path: Path, k: int = 3) -> EvaluationResult:
    rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    recalled = 0
    reciprocal_rank = 0.0
    for row in rows:
        ids = [hit.document.id for hit in retriever.search(row["question"], limit=k)]
        expected = row["document_id"]
        if expected in ids:
            recalled += 1
            reciprocal_rank += 1 / (ids.index(expected) + 1)
    count = len(rows)
    return EvaluationResult(recalled / count, reciprocal_rank / count, count)

