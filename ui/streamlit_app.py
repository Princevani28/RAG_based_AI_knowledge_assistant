"""Run: streamlit run ui/streamlit_app.py"""
import sys
import uuid
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.assistant import get_assistant  # noqa: E402

st.set_page_config(page_title="AI Knowledge Assistant", page_icon="🧠")
st.title("🧠 AI-Powered Knowledge Assistant")
assistant = get_assistant()

if "sid" not in st.session_state:
    st.session_state.sid, st.session_state.chat = str(uuid.uuid4())[:8], []

with st.sidebar:
    st.header("Knowledge base")
    for d in assistant.db.list_documents():
        st.caption(f"📄 {d['source']} ({d['chunk_count']} chunks)")
    up = st.file_uploader("Add documents", type=["pdf", "txt", "md"], accept_multiple_files=True)
    if up and st.button("Ingest"):
        folder = assistant.settings.documents_dir / "uploads"
        folder.mkdir(parents=True, exist_ok=True)
        paths = []
        for f in up:
            p = folder / Path(f.name).name
            p.write_bytes(f.getvalue())
            paths.append(p)
        st.success(assistant.ingest(paths))

for role, text in st.session_state.chat:
    st.chat_message(role).write(text)

if q := st.chat_input("Ask about your documents..."):
    st.chat_message("user").write(q)
    r = assistant.ask(q, st.session_state.sid)
    with st.chat_message("assistant"):
        st.write(r["answer"])
        if r["sources"]:
            with st.expander("Sources"):
                for s in r["sources"]:
                    st.markdown(f"**[{s['id']}] {s['source']}** (score {s['score']})  \n{s['snippet']}...")
        st.caption(f"route={r['route']} · grounded={r['grounded']} · {r['latency_ms']} ms · steps={' → '.join(r['steps'])}")
    st.session_state.chat += [("user", q), ("assistant", r["answer"])]
