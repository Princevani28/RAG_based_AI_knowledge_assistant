import json

from tests.conftest import GOLDEN


def test_retrieval_hit_rate_on_golden_set(assistant):
    cases = [c for c in json.loads(GOLDEN.read_text()) if c["expected_source"]]
    hits = 0
    for c in cases:
        res = assistant.store.similarity_search_with_score(c["question"], k=assistant.settings.top_k)
        hits += any(d.metadata["source"] == c["expected_source"] for d, _ in res)
    assert hits / len(cases) >= 0.85


def test_out_of_scope_query_scores_below_threshold(assistant):
    res = assistant.store.similarity_search_with_score("Who won the football world cup in 1998?", k=3)
    assert all(score < assistant.settings.min_relevance for _, score in res)
