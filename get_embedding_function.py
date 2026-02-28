from langchain_ollama import OllamaEmbeddings

def get_embedding_function():
    return OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )
