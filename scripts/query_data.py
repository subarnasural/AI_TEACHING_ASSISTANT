import argparse
import os
from functools import lru_cache
from dotenv import load_dotenv; load_dotenv()

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from backend.llm_manager import get_embedding_function, get_fallback_llm


CHROMA_PATH = "chroma"
DEFAULT_TOP_K = 4
MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "3200"))
MAX_CHUNK_CHARS = int(os.getenv("RAG_MAX_CHUNK_CHARS", "900"))

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def _normalize_response_text(response) -> str:
    """Convert provider-specific response payloads into plain text."""
    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
        joined = "\n".join([p for p in parts if p]).strip()
        if joined:
            return joined

    return str(content).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()

    query_rag(args.query_text)


@lru_cache(maxsize=1)
def _get_db():
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function(),
    )


def query_rag(query_text: str):
    db = _get_db()

    # Retrieve relevant chunks
    results = db.similarity_search_with_score(query_text, k=DEFAULT_TOP_K)

    # Build context with a strict budget to reduce generation latency.
    selected = []
    current_len = 0
    for doc, _score in results:
        snippet = (doc.page_content or "")[:MAX_CHUNK_CHARS].strip()
        if not snippet:
            continue

        projected = current_len + len(snippet) + 8
        if projected > MAX_CONTEXT_CHARS and selected:
            break

        selected.append(snippet)
        current_len = projected

    context_text = "\n\n---\n\n".join(selected)

    # Build prompt
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context_text,
        question=query_text,
    )

    # Gemini LLM with fallbacks
    llm = get_fallback_llm()

    # Invoke model
    response = llm.invoke(prompt)
    response_text = _normalize_response_text(response)

    # Source tracking
    sources = [doc.metadata.get("id") for doc, _score in results]

    print("\nResponse:\n")
    print(response_text)
    print("\nSources:")
    print(sources)

    return response_text


if __name__ == "__main__":
    main()
