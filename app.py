import streamlit as st
import os
import shutil
import gc

from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from get_embedding_function import get_embedding_function

# ---------------------------
# Paths & Config
# ---------------------------
CHROMA_PATH = "chroma"
UPLOAD_DIR = "uploaded_data"

PROMPT_TEMPLATE = """
You are a teaching assistant.
Answer ONLY using the provided context.
If the answer is not present, say:
"I could not find this information in the uploaded documents."

Context:
{context}

Question:
{question}

Answer:
"""

# ---------------------------
# Utility: Reset Chroma safely (NO file deletion)
# ---------------------------
def reset_chroma_collection():
    """
    Windows-safe reset of Chroma DB.
    Deletes collection instead of SQLite files.
    """
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
st.title("📘 AI Teaching Assistant")

# ---------------------------
# Sidebar: Upload PDFs
# ---------------------------
st.sidebar.header("📂 Upload your study material")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

build_kb = st.sidebar.button("Build Knowledge Base")

# ---------------------------
# Build Knowledge Base
# ---------------------------
if build_kb:
    if not uploaded_files:
        st.sidebar.warning("Please upload at least one PDF.")
    else:
        # 1️⃣ Reset Chroma safely
        reset_chroma_collection()

        # 2️⃣ Reset upload directory
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        documents = []

        # 3️⃣ Save & load PDFs
        for file in uploaded_files:
            file_path = os.path.join(UPLOAD_DIR, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())

        # 4️⃣ Split documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80
        )
        chunks = splitter.split_documents(documents)

        # 5️⃣ Build Chroma DB
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=get_embedding_function()
        )
        db.add_documents(chunks)

        del db
        gc.collect()

        st.sidebar.success("✅ Knowledge base rebuilt successfully!")
        st.rerun()

# ---------------------------
# Chat History
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ---------------------------
# Chat Input
# ---------------------------
query = st.chat_input("Ask a question from your uploaded material...")

if query:
    st.chat_message("user").write(query)
    st.session_state.messages.append(
        {"role": "user", "content": query}
    )

    if not os.path.exists(CHROMA_PATH):
        st.warning("Please upload PDFs and build the knowledge base first.")
    else:
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=get_embedding_function()
        )

        results = db.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in results])

        prompt = ChatPromptTemplate.from_template(
            PROMPT_TEMPLATE
        ).format(
            context=context,
            question=query
        )

        with st.chat_message("assistant"):
            with st.spinner("Generating answer..."):
                llm = ChatOllama(
                    model="mistral",
                    base_url="http://localhost:11434"
                )
                response = llm.invoke(prompt).content

            st.write(response)

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )

        del db
        gc.collect()
