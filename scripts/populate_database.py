import argparse
import os
import shutil
import time
from dotenv import load_dotenv; load_dotenv()
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from backend.llm_manager import get_embedding_function


CHROMA_PATH = os.path.abspath("chroma")
# Standardize with web uploads
DATA_PATH = os.path.join("uploaded_data", "files")

def main(reset_db=False):
    import sys
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument("--reset", action="store_true", help="Reset the database.")
        args, _ = parser.parse_known_args()
        reset_db = reset_db or args.reset

    if reset_db:
        print("Clearing Database")
        clear_database()

    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


def load_documents():
    loader = PyPDFDirectoryLoader(DATA_PATH)
    return loader.load()


def split_documents(documents: list[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len,
    )
    return splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function(),
    )

    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = [
        chunk for chunk in chunks_with_ids
        if chunk.metadata["id"] not in existing_ids
    ]

    if new_chunks:
        print(f"Adding new documents: {len(new_chunks)}")
        new_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_ids)
        # ❌ db.persist() REMOVED
    else:
        print("No new documents to add")


def calculate_chunk_ids(chunks: list[Document]):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk.metadata["id"] = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()
