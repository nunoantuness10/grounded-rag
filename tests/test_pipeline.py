from grounded_rag.pipeline import GroundedRAG
from grounded_rag.retrieval import Document

DOCUMENTS = [
    Document("leave", "Employees receive 25 paid vacation days each year.", "handbook.md"),
    Document("remote", "Remote employees receive a 500 euro equipment budget.", "remote.md"),
]


def test_returns_grounded_answer_with_citation() -> None:
    response = GroundedRAG(DOCUMENTS).ask("What is the vacation allowance?")
    assert not response.abstained
    assert "25 paid vacation days" in response.answer
    assert "[leave]" in response.answer
    assert response.citations[0].document_id == "leave"


def test_abstains_when_corpus_has_no_evidence() -> None:
    response = GroundedRAG(DOCUMENTS).ask("What is the office wifi password?")
    assert response.abstained
    assert response.citations == ()

