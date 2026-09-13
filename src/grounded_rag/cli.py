from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .evaluation import evaluate
from .pipeline import GroundedRAG
from .retrieval import Document


def load_documents(path: Path) -> list[Document]:
    return [Document(**row) for row in json.loads(path.read_text(encoding="utf-8"))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions against a local evidence corpus")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--documents", type=Path, default=Path("data/documents.json"))
    parser.add_argument("--evaluate", type=Path)
    args = parser.parse_args()
    rag = GroundedRAG(load_documents(args.documents))
    if args.evaluate:
        print(json.dumps(asdict(evaluate(rag.retriever, args.evaluate)), indent=2))
        return
    if not args.question:
        parser.error("provide a question or --evaluate DATASET")
    print(json.dumps(asdict(rag.ask(args.question)), indent=2))


if __name__ == "__main__":
    main()

