"""LangGraph workflow:

START -> classify -+-> chitchat ----------------------------+
                   |                                        |
                   +-> rewrite -> retrieve -+-> generate -> validate -+-> finalize -> END
                                    ^       |                         |
                                    |       +-> no_answer ------------+ (grounded / retries exhausted)
                                    +--------------- retry <----------+ (not grounded)
"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.config import Settings
from app.graph.state import GraphState
from app.prompts import (CHITCHAT_PROMPT, CLASSIFY_PROMPT, GENERATE_PROMPT, REWRITE_PROMPT,
                         VALIDATE_PROMPT)
from app.vectorstore import VectorStore

NO_ANSWER = "I couldn't find relevant information in the knowledge base to answer that question."
DISCLAIMER = "\n\n_Note: this answer could not be fully verified against the source documents._"


def format_context(docs: list[dict[str, Any]]) -> str:
    return "\n\n".join(f"[{i}] (source: {d['source']})\n{d['content']}" for i, d in enumerate(docs, 1))


class WorkflowNodes:
    def __init__(self, llm: Runnable, store: VectorStore, settings: Settings):
        self.s, self.store = settings, store
        parse = StrOutputParser()
        self.classify_chain = CLASSIFY_PROMPT | llm | parse
        self.rewrite_chain = REWRITE_PROMPT | llm | parse
        self.generate_chain = GENERATE_PROMPT | llm | parse
        self.validate_chain = VALIDATE_PROMPT | llm | parse
        self.chitchat_chain = CHITCHAT_PROMPT | llm | parse

    def _history(self, state: GraphState):
        return list(state.get("messages", []))[-self.s.max_history:]

    @staticmethod
    def _step(state: GraphState, name: str) -> list[str]:
        return list(state.get("steps", [])) + [name]

    def classify(self, state: GraphState):
        out = self.classify_chain.invoke({"question": state["question"]})
        route = "chitchat" if "chitchat" in out.lower() else "knowledge"
        return {"route": route, "steps": self._step(state, "classify")}

    def chitchat(self, state: GraphState):
        answer = self.chitchat_chain.invoke({"question": state["question"], "history": self._history(state)})
        return {"answer": answer.strip(), "documents": [], "sources": [], "grounded": True,
                "steps": self._step(state, "chitchat")}

    def rewrite(self, state: GraphState):
        history = self._history(state)
        if history:
            q = self.rewrite_chain.invoke({"question": state["question"], "history": history}).strip()
        else:
            q = state["question"]
        return {"standalone_question": q or state["question"], "steps": self._step(state, "rewrite")}

    def retrieve(self, state: GraphState):
        k = self.s.top_k + 2 * state.get("retries", 0)  # widen the net on retries
        results = self.store.similarity_search_with_score(state["standalone_question"], k=k)
        docs = [{"content": d.page_content, "source": d.metadata.get("source", "unknown"),
                 "page": d.metadata.get("page"), "chunk_index": d.metadata.get("chunk_index"),
                 "score": round(float(score), 4)}
                for d, score in results if score >= self.s.min_relevance]
        return {"documents": docs, "steps": self._step(state, "retrieve")}

    def generate(self, state: GraphState):
        docs = state["documents"]
        answer = self.generate_chain.invoke({"context": format_context(docs), "question": state["question"],
                                             "history": self._history(state)})
        sources = [{"id": i, "source": d["source"], "page": d["page"], "score": d["score"],
                    "snippet": d["content"][:200]} for i, d in enumerate(docs, 1)]
        return {"answer": answer.strip(), "sources": sources, "steps": self._step(state, "generate")}

    def validate(self, state: GraphState):
        verdict = self.validate_chain.invoke({"context": format_context(state["documents"]),
                                              "answer": state["answer"]}).strip().lower()
        grounded = "not_grounded" not in verdict and "ungrounded" not in verdict
        return {"grounded": grounded, "steps": self._step(state, "validate")}

    def retry(self, state: GraphState):
        return {"retries": state.get("retries", 0) + 1, "steps": self._step(state, "retry")}

    def no_answer(self, state: GraphState):
        return {"answer": NO_ANSWER, "sources": [], "grounded": True, "steps": self._step(state, "no_answer")}

    def finalize(self, state: GraphState):
        answer = state["answer"]
        if state.get("route") == "knowledge" and state.get("documents") and not state.get("grounded", True):
            answer += DISCLAIMER
        return {"answer": answer,
                "messages": [HumanMessage(content=state["question"]), AIMessage(content=answer)],
                "steps": self._step(state, "finalize")}

    # ---- routing ----
    def after_classify(self, state: GraphState) -> str:
        return "chitchat" if state["route"] == "chitchat" else "rewrite"

    def after_retrieve(self, state: GraphState) -> str:
        return "generate" if state["documents"] else "no_answer"

    def after_validate(self, state: GraphState) -> str:
        if state["grounded"] or state.get("retries", 0) >= self.s.max_retries:
            return "finalize"
        return "retry"


def build_graph(llm: Runnable, store: VectorStore, settings: Settings, checkpointer=None):
    n = WorkflowNodes(llm, store, settings)
    g = StateGraph(GraphState)
    for name in ["classify", "chitchat", "rewrite", "retrieve", "generate", "validate", "retry",
                 "no_answer", "finalize"]:
        g.add_node(name, getattr(n, name))
    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", n.after_classify, {"chitchat": "chitchat", "rewrite": "rewrite"})
    g.add_edge("rewrite", "retrieve")
    g.add_conditional_edges("retrieve", n.after_retrieve, {"generate": "generate", "no_answer": "no_answer"})
    g.add_edge("generate", "validate")
    g.add_conditional_edges("validate", n.after_validate, {"finalize": "finalize", "retry": "retry"})
    g.add_edge("retry", "retrieve")
    g.add_edge("chitchat", "finalize")
    g.add_edge("no_answer", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer or MemorySaver())
