"""LLM factory: ChatOpenAI for production, deterministic extractive mock for offline runs and tests."""
from __future__ import annotations

import re

from langchain_core.runnables import Runnable, RunnableLambda

from app.config import Settings

_GREETING = re.compile(r"^\s*(hi|hello|hey|thanks|thank you|good (morning|afternoon|evening))\b", re.I)


def _mock_respond(prompt_value) -> str:
    msgs = prompt_value.to_messages()
    system, human = msgs[0].content, msgs[-1].content
    if "[TASK:CLASSIFY]" in system:
        return "chitchat" if _GREETING.match(human) and len(human.split()) <= 5 else "knowledge"
    if "[TASK:REWRITE]" in system:
        return human
    if "[TASK:VALIDATE]" in system:
        return "grounded"
    if "[TASK:CHITCHAT]" in system:
        return "Hello! I'm your knowledge assistant. Ask me anything about your documents."
    if "[TASK:GENERATE]" in system:
        m = re.search(r"\[1\] \(source: [^)]*\)\n(.*?)(?:\n\n\[2\]|\n</context>)", system, re.S)
        if not m:
            return "I don't know based on the provided documents."
        return f"Based on the documents: {m.group(1).strip()[:700]} [1]"
    return "OK"


def get_llm(settings: Settings) -> Runnable:
    if settings.llm_backend == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=settings.llm_model, temperature=0, api_key=settings.openai_api_key)
    return RunnableLambda(_mock_respond)
