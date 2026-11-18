import streamlit as st
from pathlib import Path
from app.config import UPLOAD_DIR, OUTPUT_DIR
from app.pdf_utils import extract_pdf_text
from app.rag_engine import RAGEngine
from app.llm import generate_answer
from app.knowledge_graph import (
    build_conversation_graph,
    extract_conversation_facts,
    save_graph,
)

st.set_page_config(page_title="PDF Analyzer", page_icon="📄", layout="wide")
st.title("📄 PDF Analyzer")
st.caption("Chat with your PDFs using Retrieval-Augmented Generation (RAG) and explore a Knowledge Graph.")

@st.cache_resource
def get_rag_engine():
    engine = RAGEngine()
    engine.load()
    return engine

if "pages" not in st.session_state:
    st.session_state.pages = []
if "ready" not in st.session_state:
    st.session_state.ready = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "graph_ready" not in st.session_state:
    st.session_state.graph_ready = False
if "graph_image" not in st.session_state:
    st.session_state.graph_image = None
if "graph_data" not in st.session_state:
    st.session_state.graph_data = None
if "graph_entity_count" not in st.session_state:
    st.session_state.graph_entity_count = 0
if "graph_relationship_count" not in st.session_state:
    st.session_state.graph_relationship_count = 0

rag = get_rag_engine()
with st.sidebar:
    st.header("1. Upload documents")
    uploaded_files = st.file_uploader(
        "Choose PDF files", type=["pdf"], accept_multiple_files=True
    )
    if st.button("Process PDFs", type="primary", disabled=not uploaded_files):
        all_pages = []
        progress = st.progress(0)
        status = st.empty()
        for i, uploaded in enumerate(uploaded_files):
            safe_name = Path(uploaded.name).name
            destination = UPLOAD_DIR / safe_name
            destination.write_bytes(uploaded.getvalue())
            status.write(f"Reading {safe_name}...")
            all_pages.extend(extract_pdf_text(str(destination)))
            progress.progress((i + 1) / len(uploaded_files))
        if not all_pages:
            st.error("No selectable text found. Scanned PDFs need OCR before processing.")
        else:
            with st.spinner("Creating embeddings and FAISS index..."):
                count = rag.build(all_pages)
            st.session_state.pages = all_pages
            st.session_state.ready = True
            st.success(f"Processed {len(all_pages)} pages into {count} chunks.")
            st.rerun()

    st.divider()
    st.header("2. Conversation knowledge graph")

chat_tab, graph_tab, info_tab = st.tabs(
    ["💬 Chat with PDFs", "🕸️ Knowledge Graph", "ℹ️ How it works"]
)
with chat_tab:
    st.subheader("Ask questions about your documents")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    question = st.chat_input("Ask something from your uploaded PDFs...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching relevant PDF passages and asking the local model..."):
                    results = rag.search(question)
                    answer = generate_answer(question, results)
                st.markdown(answer)
                sources = "\n".join(
                    f"- **{d['source']}**, page {d['page']} (similarity {d['score']:.2f})"
                    for d in results
                )
                with st.expander("Retrieved sources"):
                    st.markdown(sources)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer + "\n\n**Retrieved sources**\n" + sources
                })
            except Exception as exc:
                st.error(f"{exc}\n\nCheck that PDFs are processed and Ollama is running.")
        try:
            with st.spinner("Extracting entities and relationships from this chat..."):
                facts = extract_conversation_facts(st.session_state.messages)
                graph = build_conversation_graph(facts)
                image_data, graph_data = save_graph(
                    graph,
                    str(OUTPUT_DIR / "knowledge_graph.graphml"),
                    str(OUTPUT_DIR / "knowledge_graph.png"),
                )
            st.session_state.graph_image = image_data
            st.session_state.graph_data = graph_data
            st.session_state.graph_entity_count = sum(
                data["kind"] == "entity"
                for _, data in graph.nodes(data=True)
            )
            st.session_state.graph_relationship_count = graph.number_of_edges()
            st.session_state.graph_ready = True
        except Exception as exc:
            st.error(f"Could not update the conversation knowledge graph: {exc}")
with graph_tab:
    st.subheader("Knowledge graph from this conversation")
    st.caption(
        "Entities are nodes; arrows are relationships explicitly stated in the chat. "
        "Only chat messages are used, not PDF contents."
    )
    if st.session_state.graph_ready and st.session_state.graph_image:
        st.caption(
            f"{st.session_state.graph_entity_count} entities · "
            f"{st.session_state.graph_relationship_count} relationships"
        )
        st.image(
            st.session_state.graph_image,
            caption="Current conversation knowledge graph",
            use_container_width=True,
        )
        st.download_button(
            "Download Knowledge Graph PNG",
            data=st.session_state.graph_image,
            file_name="conversation_knowledge_graph.png",
            mime="image/png",
        )
        if st.session_state.graph_data:
            st.download_button(
                "Download GraphML",
                data=st.session_state.graph_data,
                file_name="conversation_knowledge_graph.graphml",
            )
    else:
        st.info("Ask a question to build a knowledge graph from this conversation.")
with info_tab:
    st.markdown("""
    **RAG flow:** PDF text extraction → overlapping chunks → sentence embeddings →
    FAISS similarity search → relevant context → Ollama answer with source pages.

    **Knowledge Graph:** Ollama extracts entities and relationships explicitly
    stated in the current conversation. Entities are shown as nodes and relationships
    as labeled arrows. PDF contents and citation metadata are excluded.
    """)
