import streamlit as st
import os
import shutil
import gc
import pytesseract

# 🔥 Tesseract path (IMPORTANT)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ollama import ResponseError

from get_embedding_function import get_embedding_function

# --- NEW IMPORTS ---
from utils.ocr_engine import extract_text_from_image
from utils.evaluator import calculate_metrics

# ---------------------------
# Paths & Config
# ---------------------------
CHROMA_PATH = "chroma"
UPLOAD_DIR = os.path.join("uploaded_data", "files")
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "tinyllama")
NO_ANSWER_TEXT = "I don't have enough information to answer this question."

PROMPT_TEMPLATE = """
You are a strict retrieval QA assistant.
Answer ONLY from the provided context.
If the context contains the answer, extract it clearly.
If the context does not contain the answer, respond with exactly:
"I don't have enough information to answer this question."
Keep the response concise (max 3 short sentences).

Context:
{context}

Question:
{question}

Answer:
"""


def _answer_is_grounded(answer: str, context: str) -> bool:
    answer_tokens = {t.lower() for t in answer.split() if len(t) > 3}
    context_tokens = {t.lower() for t in context.split() if len(t) > 3}
    if not answer_tokens:
        return False
    overlap = len(answer_tokens & context_tokens) / max(len(answer_tokens), 1)
    return overlap >= 0.2


# ---------------------------
# Utility: Reset Chroma safely
# ---------------------------
def reset_chroma_collection():
    if os.path.exists(CHROMA_PATH):
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=get_embedding_function()
        )
        db.delete_collection()
        del db
        gc.collect()


# ---------------------------
# Streamlit Page Setup
# ---------------------------
st.set_page_config(page_title="AI Teaching Assistant", layout="wide")
st.title("📘 AI Teaching Assistant - Enhanced")
st.caption(f"Chat model: `{CHAT_MODEL}`")


# ---------------------------
# Sidebar: Knowledge Base & OCR
# ---------------------------
st.sidebar.header("📂 Knowledge Base")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if st.sidebar.button("Build Knowledge Base"):
    if not uploaded_files:
        st.sidebar.warning("Please upload at least one PDF.")
    else:
        reset_chroma_collection()

        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        documents = []
        for file in uploaded_files:
            file_path = os.path.join(UPLOAD_DIR, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
        chunks = splitter.split_documents(documents)

        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=get_embedding_function()
        )
        db.add_documents(chunks)
        del db
        gc.collect()

        st.sidebar.success("✅ Knowledge base rebuilt!")
        st.rerun()


# ---------------------------
# OCR Section
# ---------------------------
st.sidebar.markdown("---")
st.sidebar.header("📸 Image Question (OCR)")

ocr_image = st.sidebar.file_uploader("Upload a question image", type=["png", "jpg", "jpeg"])

if ocr_image:
    st.sidebar.image(ocr_image, caption="Uploaded Image", use_container_width=True)

    if st.sidebar.button("Extract & Ask from Image"):
        with st.spinner("Reading image..."):
            ocr_result = extract_text_from_image(ocr_image)

            if ocr_result["error"]:
                st.sidebar.error(f"OCR error: {ocr_result['error']}")
            else:
                st.session_state.ocr_text = ocr_result["text"]
                st.session_state.ocr_conf = ocr_result["confidence"]
                st.session_state.ocr_method = ocr_result["method"]

                # 🔥 Auto-use OCR text as query
                st.session_state.ocr_query = ocr_result["text"]


if "ocr_text" in st.session_state:
    st.sidebar.markdown("### OCR Result")
    st.sidebar.write(
        f"Confidence: **{st.session_state.get('ocr_conf', 0):.2f}%** "
        f"({st.session_state.get('ocr_method', 'unknown')})"
    )

    edited_ocr = st.sidebar.text_area(
        "Review / edit extracted text before asking",
        value=st.session_state["ocr_text"],
        height=120,
    )

    if st.sidebar.button("Use OCR Text as Question"):
        st.session_state.ocr_query = edited_ocr


# ---------------------------
# Chat History
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# ---------------------------
# Chat Input Logic
# ---------------------------
user_input = st.chat_input("Ask a question from your material...")

query = user_input
if "ocr_query" in st.session_state:
    query = st.session_state.ocr_query
    del st.session_state.ocr_query


if query:
    st.chat_message("user").write(query)
    st.session_state.messages.append({"role": "user", "content": query})

    context = ""

    # 🔥 Case 1: No PDF but OCR exists
    if not os.path.exists(CHROMA_PATH):
        if "ocr_text" in st.session_state:
            context = st.session_state["ocr_text"]
        else:
            st.warning("Please build the knowledge base first.")
            st.stop()

    else:
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=get_embedding_function()
        )

        try:
            results_with_scores = db.similarity_search_with_relevance_scores(query, k=4)
        except ResponseError as exc:
            st.error("Embedding model missing. Run: `ollama pull nomic-embed-text`")
            st.stop()

        filtered_docs = [doc for doc, score in results_with_scores if score >= 0.25]

        if filtered_docs:
            context = "\n\n".join([doc.page_content for doc in filtered_docs])

        # 🔥 ADD OCR TEXT INTO CONTEXT
        if "ocr_text" in st.session_state:
            context = st.session_state["ocr_text"] + "\n\n" + context

        del db
        gc.collect()

    # ---------------------------
    # LLM Call
    # ---------------------------
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context,
        question=query
    )

    with st.chat_message("assistant"):
        with st.spinner("Generating answer..."):
            llm = ChatOllama(model=CHAT_MODEL, base_url="http://localhost:11434")

            try:
                response = llm.invoke(prompt).content
            except ResponseError:
                st.error("Model not available. Run: `ollama pull tinyllama`")
                st.stop()

            if response.strip().lower() != NO_ANSWER_TEXT.lower() and not _answer_is_grounded(response, context):
                response = NO_ANSWER_TEXT

            st.write(response)
            st.session_state.last_response = response

    st.session_state.messages.append({"role": "assistant", "content": response})


# ---------------------------
# Evaluation Section
# ---------------------------
if "last_response" in st.session_state:
    st.markdown("---")

    with st.expander("📊 Evaluate Last Response"):
        gold_standard = st.text_area("Paste correct answer")

        if st.button("Calculate Performance"):
            if gold_standard:
                metrics = calculate_metrics(st.session_state.last_response, gold_standard)

                c1, c2, c3 = st.columns(3)
                c1.metric("Precision", metrics["Precision"])
                c2.metric("Recall", metrics["Recall"])
                c3.metric("F1 Score", metrics["F1 Score"])
            else:
                st.warning("Please enter a reference answer.")