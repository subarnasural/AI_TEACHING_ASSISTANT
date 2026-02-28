import argparse

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from get_embedding_function import get_embedding_function


CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()

    query_rag(args.query_text)


def query_rag(query_text: str):
    # Load embeddings
    embedding_function = get_embedding_function()

    # Load Chroma DB
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function,
    )

    # Retrieve relevant chunks
    results = db.similarity_search_with_score(query_text, k=5)

    # Build context
    context_text = "\n\n---\n\n".join(
        [doc.page_content for doc, _score in results]
    )

    # Build prompt
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context_text,
        question=query_text,
    )

    # Local Ollama LLM
    llm = ChatOllama(
        model="mistral",
        base_url="http://localhost:11434",
    )

    # Invoke model
    response_text = llm.invoke(prompt)

    # Source tracking
    sources = [doc.metadata.get("id") for doc, _score in results]

    print("\nResponse:\n")
    print(response_text)
    print("\nSources:")
    print(sources)

    return response_text


if __name__ == "__main__":
    main()
