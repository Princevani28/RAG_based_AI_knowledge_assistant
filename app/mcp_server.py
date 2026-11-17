"""MCP server exposing the knowledge base as tools for any MCP client (Claude Desktop, IDEs, agents).
Run:  python -m app.mcp_server   (stdio transport)
"""
from mcp.server.fastmcp import FastMCP

from app.assistant import get_assistant

mcp = FastMCP("knowledge-assistant")


@mcp.tool()
def search_knowledge_base(query: str, top_k: int = 4) -> list[dict]:
    """Semantic search over enterprise documents. Returns the most relevant chunks with scores."""
    a = get_assistant()
    results = a.store.similarity_search_with_score(query, k=max(1, min(top_k, 20)))
    return [{"source": d.metadata.get("source"), "page": d.metadata.get("page"),
             "score": round(float(s), 4), "content": d.page_content} for d, s in results]


@mcp.tool()
def ask_knowledge_assistant(question: str, session_id: str = "mcp") -> dict:
    """Ask the full RAG workflow (retrieve, generate, validate) and get a cited answer."""
    r = get_assistant().ask(question, session_id)
    return {"answer": r["answer"], "sources": r["sources"], "grounded": r["grounded"]}


if __name__ == "__main__":
    mcp.run()
