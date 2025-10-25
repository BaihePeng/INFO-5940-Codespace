import streamlit as st
import os
import re
from os import environ
from openai import OpenAI
import io

# --- API Key / client setup ---
api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
if not api_key:
    st.warning("OPENAI_API_KEY or API_KEY not set — the app will not be able to call the API until you provide a key.")

client = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL") or "https://api.ai.it.cornell.edu")


# --- Optional imports for vector DB / embeddings ---
_have_chroma = True
_have_langchain = True
try:
    from langchain_core.documents import Document
    from langchain_openai import OpenAIEmbeddings
    from langchain_chroma import Chroma
except Exception:
    _have_chroma = False
    _have_langchain = False


# --- Token counting and chunking utilities ---
try:
    import tiktoken
    _has_tiktoken = True
except Exception:
    tiktoken = None
    _has_tiktoken = False


# PDF support
try:
    from pypdf import PdfReader
    _have_pypdf = True
except Exception:
    PdfReader = None
    _have_pypdf = False


def _num_tokens(text: str, model: str | None = None) -> int:
    if _has_tiktoken:
        try:
            if model:
                enc = tiktoken.encoding_for_model(model)
            else:
                enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            pass
    words = text.split()
    return max(1, int(len(words) / 1.33))


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 200, model: str | None = None) -> list:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sentences = re.split(r'(?<=[\.!?])\s+', text)
    chunks = []
    cur_sentences = []
    cur_tokens = 0

    for sent in sentences:
        sent_tokens = _num_tokens(sent, model=model)
        if sent_tokens >= chunk_size:
            # force split large sentence by words
            words = sent.split()
            part = []
            part_tokens = 0
            for w in words:
                w_tokens = _num_tokens(w, model=model)
                part.append(w)
                part_tokens += w_tokens
                if part_tokens >= chunk_size:
                    chunks.append(" ".join(part))
                    if chunk_overlap > 0:
                        keep_count = max(1, int(len(part) * (chunk_overlap / max(part_tokens, 1))))
                        part = part[-keep_count:]
                        part_tokens = _num_tokens(" ".join(part), model=model)
                    else:
                        part = []
                        part_tokens = 0
            if part:
                cur_sentences = [" ".join(part)]
                cur_tokens = _num_tokens(cur_sentences[0], model=model)
            continue

        if cur_tokens + sent_tokens <= chunk_size:
            cur_sentences.append(sent)
            cur_tokens += sent_tokens
        else:
            chunks.append(" ".join(cur_sentences))
            if chunk_overlap > 0:
                keep = []
                keep_tokens = 0
                for s in reversed(cur_sentences):
                    s_t = _num_tokens(s, model=model)
                    if keep_tokens + s_t > chunk_overlap:
                        break
                    keep.insert(0, s)
                    keep_tokens += s_t
                cur_sentences = keep[:] if keep else [sent]
                cur_tokens = sum(_num_tokens(s, model=model) for s in cur_sentences)
                if cur_sentences == [sent]:
                    cur_tokens = sent_tokens
            else:
                cur_sentences = [sent]
                cur_tokens = sent_tokens

    if cur_sentences:
        chunks.append(" ".join(cur_sentences))
    return chunks


# --- Streamlit UI and RAG pipeline ---
st.title("📝 File Q&A with RAG (upload documents)")

uploaded_files = st.file_uploader("Upload document(s)", type=("txt", "pdf"), accept_multiple_files=True)

# Indexing settings are kept in the backend (defaults here). To change them, set
# environment variables RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP, and RAG_TOP_K before
# starting the app. These settings are intentionally not shown to end users.
chunk_size = int(os.getenv("RAG_CHUNK_SIZE", "800"))
chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
top_k = int(os.getenv("RAG_TOP_K", "4"))


def index_documents(files):
    """Read and chunk uploaded files and return a list of Document objects.

    This function does not build the vectorstore; the caller will merge new
    documents with any existing index and (re)build the vectorstore as needed.
    """
    docs = []
    for f in files:
        raw = f.read()
        # detect by extension
        fname = (getattr(f, "name", "") or "").lower()
        text = ""
        if fname.endswith(".pdf"):
            if not _have_pypdf:
                st.error("pypdf is required to parse PDF files. Please install the dependency (pypdf>=4).")
                continue
            try:
                reader = PdfReader(io.BytesIO(raw))
                pages = []
                for p in reader.pages:
                    try:
                        page_text = p.extract_text() or ""
                    except Exception:
                        page_text = ""
                    pages.append(page_text)
                text = "\n\n".join(pages)
            except Exception as e:
                st.error(f"Failed to parse PDF {fname}: {e}")
                continue
        else:
            # assume text-like file
            try:
                text = raw.decode("utf-8")
            except Exception:
                # fallback: replace errors
                text = raw.decode("utf-8", errors="replace")

        chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for i, c in enumerate(chunks):
            meta = {"source": f.name, "chunk": i}
            docs.append(Document(page_content=c, metadata=meta))

    return docs


if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None
    st.session_state["docs"] = []

if uploaded_files:
    # Index newly uploaded files and merge with any previously indexed documents.
    st.info("Indexing uploaded documents — this may take a moment.")
    new_docs = index_documents(uploaded_files)

    # merge new docs by filename: replace any existing docs that have the same source
    existing_docs = st.session_state.get("docs", [])
    existing_sources = {d.metadata.get("source") for d in existing_docs}
    added = 0
    replaced = 0
    for d in new_docs:
        src = d.metadata.get("source")
        if src in existing_sources:
            # remove old docs with this source
            st.session_state["docs"] = [ed for ed in st.session_state.get("docs", []) if ed.metadata.get("source") != src]
            replaced += 1
        st.session_state.setdefault("docs", []).append(d)
        added += 1

    if not st.session_state.get("docs"):
        st.info("No documents indexed.")
    else:
        if not _have_chroma:
            st.error("Chroma/langchain libs not available in the environment. Install them or use the provided requirements.txt.")
        else:
            # Clear any previously cached vectorstore so we don't accidentally use a stale index
            st.session_state["vectorstore"] = None
            # Rebuild vectorstore from all docs (simple, reliable approach)
            embedding = OpenAIEmbeddings(model="openai.text-embedding-3-large")
            try:
                vectordb = Chroma.from_documents(documents=st.session_state["docs"], embedding=embedding)
                st.session_state["vectorstore"] = vectordb
                # keep embedding for potential temporary indexes
                st.session_state["embedding"] = embedding
                # Only notify about index update when new files are uploaded
            except Exception as e:
                st.error(f"Failed to build vectorstore: {e}")

    st.success(f"Indexed {len(st.session_state.get('docs', []))} chunks from {len({d.metadata.get('source') for d in st.session_state.get('docs', [])})} file(s) (added {added}, replaced {replaced})")

if "history" not in st.session_state:
    st.session_state["history"] = []

# Render existing chat history for the user (preserve multi-turn context)
if st.session_state["history"]:
    for msg in st.session_state["history"]:
        try:
            st.chat_message(msg["role"]).write(msg["content"])
        except Exception:
            # defensive: if role or content malformed, skip
            continue
else:
    # show a gentle assistant prompt when there's no history yet
    init_msg = "Ask something about the documents"
    st.session_state["history"].append({"role": "assistant", "content": init_msg})
    st.chat_message("assistant").write(init_msg)


def retrieve_topk(question: str, k: int = 4):
    if not st.session_state.get("vectorstore"):
        return []
    try:
        # Always use the main vectorstore which contains all uploaded files.
        return st.session_state["vectorstore"].similarity_search(question, k=k)
    except Exception:
        # Best-effort fallback
        return []


def build_prompt_from_context(docs):
    parts = []
    for d in docs:
        src = d.metadata.get("source", "(no source)")
        # Include the source filename in the context prompt. 
        parts.append(f"Source: {src}\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


query = st.chat_input("Ask something about the documents", disabled=not st.session_state.get("docs"))

if query:
    # show user message
    st.session_state["history"].append({"role": "user", "content": query})
    st.chat_message("user").write(query)

    # Retrieve
    retrieved = retrieve_topk(query, k=top_k)

    context = build_prompt_from_context(retrieved)

    system_prompt = (
        "You are an assistant that answers user questions using ONLY the provided context. "
        "If the answer is not contained in the context, say you don't know. Be concise and cite the source filenames when relevant."
    )

    if not api_key:
        st.error("API key missing — cannot call model.")
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"Context:\n{context}"},
            {"role": "user", "content": query},
        ]

        # Call the model (non-streaming)
        try:
            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL") or "openai.gpt-4o",
                messages=messages,
                temperature=0.0,
                max_tokens=512,
            )
            # best-effort extraction of assistant text
            assistant_text = None
            try:
                assistant_text = resp.choices[0].message.content
            except Exception:
                # fallback: string conversion
                assistant_text = str(resp)

            st.chat_message("assistant").write(assistant_text)
            st.session_state["history"].append({"role": "assistant", "content": assistant_text})



        except Exception as e:
            st.error(f"Model call failed: {e}")


## Debug / quick info
# show a small, non-intrusive status in the main UI (no sidebar usage)
st.caption(f"Indexed chunks: {len(st.session_state.get('docs', []))}")