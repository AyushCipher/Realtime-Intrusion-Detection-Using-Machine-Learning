import pytest

from ids_tier2.retrieval import TechniqueRetriever
from ids_tier2.knowledge_base import TechniqueEntry


def test_retrieve_returns_relevant_technique_for_exact_category_name():
    retriever = TechniqueRetriever()
    results = retriever.retrieve("Brute Force", top_k=1)
    assert len(results) == 1
    assert results[0].entry.technique_id == "T1110"
    assert 0.0 <= results[0].score <= 1.0


def test_retrieve_by_category_matches_plain_retrieve():
    retriever = TechniqueRetriever()
    assert retriever.retrieve_by_category("Botnet", top_k=1)[0].entry.technique_id == "T1071.001"


def test_retrieve_respects_top_k():
    retriever = TechniqueRetriever()
    results = retriever.retrieve("denial of service flood traffic", top_k=2)
    assert len(results) <= 2


def test_retrieve_respects_min_score_threshold():
    retriever = TechniqueRetriever()
    # An unrelated query should score low against every entry.
    results = retriever.retrieve("unrelated gardening advice", top_k=5, min_score=0.9)
    assert results == []


def test_results_sorted_descending_by_score():
    retriever = TechniqueRetriever()
    results = retriever.retrieve("network flood attack traffic discovery scan", top_k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_empty_knowledge_base_raises():
    with pytest.raises(ValueError):
        TechniqueRetriever(knowledge_base=[])


@pytest.mark.parametrize(
    "category,expected_top_technique_id",
    [
        ("DoS/DDoS", "T1498"),
        ("Brute Force", "T1110"),
        ("Web Attack", "T1190"),
        ("PortScan", "T1046"),
        ("Infiltration", "T1210"),
        ("Botnet", "T1071.001"),
    ],
)
def test_every_real_attack_category_retrieves_its_intended_technique(category, expected_top_technique_id):
    # This is the query shape reasoner.py actually uses in production --
    # the alert's exact stage2_predicted_class string, verbatim. Every
    # category this project's ATTACK_CATEGORY_MAP models must retrieve
    # its intended top-1 technique from that exact string, not a looser
    # paraphrase of it.
    retriever = TechniqueRetriever()
    results = retriever.retrieve(category, top_k=1)
    assert results[0].entry.technique_id == expected_top_technique_id


def test_custom_knowledge_base_is_used():
    custom = [
        TechniqueEntry(technique_id="T0001", name="Made Up", tactic="Test", category="Test", description="a fabricated technique for testing retrieval on a tiny custom corpus"),
    ]
    retriever = TechniqueRetriever(knowledge_base=custom)
    results = retriever.retrieve("fabricated technique", top_k=1)
    assert results[0].entry.technique_id == "T0001"
