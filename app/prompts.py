"""Prompt templates. Each system prompt starts with a [TASK:*] tag (used by the offline mock LLM and for tracing)."""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

CLASSIFY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "[TASK:CLASSIFY]\nClassify the user message. Reply with exactly one word:\n"
     "- chitchat: greetings, thanks, small talk, questions about what you can do\n"
     "- knowledge: anything that needs information from company documents"),
    ("human", "{question}"),
])

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "[TASK:REWRITE]\nRewrite the latest user question as a standalone search query, resolving pronouns "
     "using the chat history. Return only the rewritten query."),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "[TASK:GENERATE]\nYou are an enterprise knowledge assistant. Answer ONLY from the context below.\n"
     "Rules:\n- Cite sources inline like [1], [2].\n"
     "- If the context does not contain the answer, say you do not know based on the provided documents.\n"
     "- Be concise and accurate.\n\n<context>\n{context}\n</context>"),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

VALIDATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "[TASK:VALIDATE]\nYou are a strict fact checker. Decide whether every claim in the ANSWER is supported "
     "by the CONTEXT. Reply with exactly one word: grounded or not_grounded."),
    ("human", "CONTEXT:\n{context}\n\nANSWER:\n{answer}"),
])

CHITCHAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "[TASK:CHITCHAT]\nYou are a friendly enterprise knowledge assistant. Reply briefly and invite the user "
     "to ask questions about their documents."),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])
