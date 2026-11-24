"""Interactive terminal chat.  Usage: python scripts/chat_cli.py"""
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.assistant import get_assistant  # noqa: E402

if __name__ == "__main__":
    assistant, sid = get_assistant(), str(uuid.uuid4())[:8]
    print("Knowledge Assistant ready. Type 'exit' to quit.")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue
        r = assistant.ask(q, sid)
        print(f"\nAssistant: {r['answer']}")
        for s in r["sources"]:
            print(f"  [{s['id']}] {s['source']} (p.{s['page']}, score {s['score']})")




            
