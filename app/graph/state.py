from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class GraphState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]  # conversational memory (checkpointed per thread)
    question: str
    standalone_question: str
    route: str                      # knowledge | chitchat
    documents: list[dict[str, Any]]
    answer: str
    sources: list[dict[str, Any]]
    grounded: bool
    retries: int
    steps: list[str]                # execution trace, used by AI assurance tests
