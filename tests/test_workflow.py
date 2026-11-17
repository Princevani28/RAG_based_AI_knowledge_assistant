from langchain_core.runnables import RunnableLambda

from app.assistant import KnowledgeAssistant
from app.graph.workflow import DISCLAIMER, NO_ANSWER
from app.llm import _mock_respond
from tests.conftest import DOCS


def test_chitchat_skips_retrieval(assistant):
    r = assistant.ask("Hello!", "s1")
    assert r["route"] == "chitchat"
    assert "retrieve" not in r["steps"] and r["steps"][-1] == "finalize"


def test_knowledge_path_runs_full_rag_pipeline(assistant):
    r = assistant.ask("How many days of annual leave do employees get?", "s2")
    assert r["steps"] == ["classify", "rewrite", "retrieve", "generate", "validate", "finalize"]
    assert "24" in r["answer"] and r["sources"][0]["source"] == "hr_policy.md" and r["grounded"]


def test_out_of_scope_returns_no_answer(assistant):
    r = assistant.ask("Who won the football world cup in 1998?", "s3")
    assert r["answer"] == NO_ANSWER and "generate" not in r["steps"] and r["sources"] == []


def _flaky_validator(fail_times: int):
    calls = {"n": 0}

    def respond(pv):
        if "[TASK:VALIDATE]" in pv.to_messages()[0].content:
            calls["n"] += 1
            return "not_grounded" if calls["n"] <= fail_times else "grounded"
        return _mock_respond(pv)
    return RunnableLambda(respond)


def test_retry_loop_recovers_when_validation_fails_once(settings):
    a = KnowledgeAssistant(settings, llm=_flaky_validator(1))
    a.ingest([DOCS])
    r = a.ask("What is the minimum password length?", "s4")
    assert r["steps"].count("retrieve") == 2 and "retry" in r["steps"]
    assert r["grounded"] and DISCLAIMER not in r["answer"]


def test_retries_exhausted_adds_disclaimer(settings):
    a = KnowledgeAssistant(settings, llm=_flaky_validator(99))
    a.ingest([DOCS])
    r = a.ask("What is the minimum password length?", "s5")
    assert r["steps"].count("retry") == settings.max_retries
    assert not r["grounded"] and DISCLAIMER in r["answer"]


def test_conversation_memory_is_per_session(assistant):
    cfg = lambda sid: {"configurable": {"thread_id": sid}}
    assistant.ask("How much does the Pro plan cost per month?", "A")
    assistant.ask("What about the API rate limit?", "A")
    assistant.ask("Hi there", "B")
    assert len(assistant.graph.get_state(cfg("A")).values["messages"]) == 4
    assert len(assistant.graph.get_state(cfg("B")).values["messages"]) == 2


def test_interactions_are_logged_to_sql(assistant):
    assistant.ask("How long is customer analytics data retained?", "log1")
    h = assistant.db.history("log1")
    assert len(h) == 1 and "13 months" in h[0]["answer"]
