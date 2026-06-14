"""Streamlit UI for the Upwork API Support Bot.

Run from the project root:  streamlit run app.py
"""
from __future__ import annotations

import time

import streamlit as st
from groq import APIError, APIConnectionError, RateLimitError

from src import llm, rag
from src.config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL, LLM_MODEL, TOP_K

st.set_page_config(
    page_title="Upwork API Support Bot",
    page_icon="",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading vector store...")
def get_vectorstore():
    return rag.load_vectorstore()


EXAMPLE_QUESTIONS = [
    "What is the rate limit for the Upwork API?",
    "How long is an OAuth access token valid for?",
    "Can I use a Client Credentials Grant to access a user's private contract details?",
    "Who can create subscriptions on the Upwork API?",
]


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None


def render_sources(chunks) -> None:
    with st.expander(f"Sources ({len(chunks)} retrieved chunks)"):
        for i, chunk in enumerate(chunks, 1):
            page = chunk.metadata.get("page", "?")
            st.markdown(f"**Chunk {i}** · `page {page}`")
            st.code(chunk.page_content, language="text")


def answer_question(question: str, vectorstore) -> None:
    """Retrieve, stream the LLM answer, and append to chat history."""
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        # Live reasoning trace: retrieval first, then streaming generation.
        with st.status("Thinking...", expanded=True) as status:
            st.write(f"**Searching docs** for: _{question}_")
            chunks = rag.retrieve(question, vectorstore)

            st.write(f" **Retrieved top-{len(chunks)} chunks:**")
            for i, c in enumerate(chunks, 1):
                page = c.metadata.get("page", "?")
                preview = c.page_content.strip().replace("\n", " ")[:140]
                st.markdown(f"  {i}. `page {page}` — {preview}…")

            st.write(" **Generating grounded answer** (streaming)...")
            status.update(label="Reasoning over retrieved context", state="running")

        placeholder = st.empty()
        try:
            start = time.perf_counter()
            buffer = ""
            for delta in llm.stream_answer(question, chunks):
                buffer += delta
                placeholder.markdown(buffer + "▌")
            elapsed = time.perf_counter() - start
            placeholder.markdown(buffer)
        except (APIConnectionError, RateLimitError, APIError) as exc:
            placeholder.error(f"LLM API call failed: {exc}")
            return
        except RuntimeError as exc:
            placeholder.error(str(exc))
            return

        st.caption(f"{elapsed:.2f} s · top-{len(chunks)} retrieval · streamed")
        render_sources(chunks)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": buffer,
            "latency": elapsed,
            "chunks": chunks,
        }
    )


# ---------- App ----------

init_state()

with st.sidebar:
    st.header("Upwork API Bot")
    st.write(
        "A retrieval-augmented bot that answers developer questions about the "
        "Upwork API, grounded only in the official documentation."
    )

    st.subheader("Stack")
    st.markdown(
        f"- **Embeddings:** `{EMBEDDING_MODEL.split('/')[-1]}`\n"
        f"- **LLM:** `{LLM_MODEL.split('/')[-1]}`\n"
        f"- **Chunking:** {CHUNK_SIZE} chars / {CHUNK_OVERLAP} overlap\n"
        f"- **Retrieval:** top-{TOP_K}"
    )

    st.subheader("Try a question")
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        if st.button(q, key=f"ex_{i}", use_container_width=True):
            st.session_state.pending_question = q
            st.rerun()

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()

st.title("Upwork API Support Bot")
st.caption(
    "Ask a question about the Upwork API. Answers are grounded in the official "
    "documentation — if it's not in the docs, the bot will say so."
)

try:
    vectorstore = get_vectorstore()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

# Replay history.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            st.caption(
                f"{msg['latency']:.2f} s · top-{len(msg['chunks'])} retrieval"
            )
            render_sources(msg["chunks"])

# Either an example-button click or a chat-input submit.
question = st.chat_input("Ask about OAuth, endpoints, scopes, rate limits...")
if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

if question and question.strip():
    answer_question(question.strip(), vectorstore)
