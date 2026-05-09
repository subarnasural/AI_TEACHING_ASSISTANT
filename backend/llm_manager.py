import os
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.embeddings import Embeddings
from google.api_core.exceptions import ResourceExhausted, InternalServerError

_AVAILABLE_MODELS_CACHE = {}
_FALLBACK_LLM_SINGLETON = None
_EMBEDDINGS_SINGLETON = None

def get_api_keys() -> List[str]:
    """Retrieve multiple API keys from .env separated by comma."""
    keys_str = os.getenv("GEMINI_API_KEYS")
    if keys_str:
        keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        if keys:
            return keys

    # Fallback to single key setup
    single_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if single_key:
        return [single_key.strip()]

    raise ValueError(
        "Missing Gemini API keys. Set GEMINI_API_KEYS (comma separated) "
        "or GEMINI_API_KEY environment variable."
    )

def get_chat_model_names() -> List[str]:
    """Resolve chat model priority from env with safe defaults."""
    configured = os.getenv("GEMINI_CHAT_MODELS", "")
    models = [m.strip() for m in configured.split(",") if m.strip()]

    preferred = os.getenv("GEMINI_CHAT_MODEL", "").strip()
    if preferred:
        models.insert(0, preferred)

    if not models:
        # Prefer latest flash variants while keeping broad compatibility.
        models = [
            "gemini-3-flash-preview",
            "gemini-flash-latest",
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-pro-latest",
        ]

    deduped = []
    seen = set()
    for m in models:
        if m.startswith("models/"):
            m = m.split("/", 1)[1]
        if m not in seen:
            deduped.append(m)
            seen.add(m)
    return deduped


def get_available_generate_models(api_key: str) -> List[str]:
    if api_key in _AVAILABLE_MODELS_CACHE:
        return _AVAILABLE_MODELS_CACHE[api_key]

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        available = []
        for model in client.models.list():
            name = getattr(model, "name", "")
            if name.startswith("models/"):
                name = name.split("/", 1)[1]

            methods = (
                getattr(model, "supported_actions", None)
                or getattr(model, "supported_generation_methods", None)
                or []
            )
            if any("generate" in str(m).lower() for m in methods):
                available.append(name)

        _AVAILABLE_MODELS_CACHE[api_key] = available
        return available
    except Exception:
        _AVAILABLE_MODELS_CACHE[api_key] = []
        return []


def _should_discover_models() -> bool:
    """Model discovery can add startup latency; keep it optional for faster first response."""
    flag = os.getenv("GEMINI_DISCOVER_MODELS", "false").strip().lower()
    return flag in {"1", "true", "yes", "on"}


class GeminiFallbackLLM:
    """Small wrapper that retries across model/key combinations."""

    def __init__(self):
        keys = get_api_keys()
        models = get_chat_model_names()
        self.clients = []

        fallback_priority = [
            "gemini-3.1-flash-lite-preview",
            "gemini-3-flash-preview",
            "gemini-flash-latest",
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-pro-latest",
        ]

        for key in keys:
            available = set(get_available_generate_models(key)) if _should_discover_models() else set()

            if available:
                selected_models = [m for m in models if m in available]
                if not selected_models:
                    selected_models = [m for m in fallback_priority if m in available]
            else:
                selected_models = models

            for model in selected_models:
                self.clients.append(
                    {
                        "model": model,
                        "client": ChatGoogleGenerativeAI(
                            model=model,
                            google_api_key=key,
                            max_retries=1,
                        ),
                    }
                )

    def invoke(self, prompt):
        last_error = None

        for entry in self.clients:
            model = entry["model"]
            client = entry["client"]
            try:
                return client.invoke(prompt)
            except Exception as e:
                last_error = e
                message = str(e).lower()

                # Retry next model/key for common API and model availability failures.
                recoverable = any(
                    token in message
                    for token in [
                        "not_found",
                        "not found",
                        "resource_exhausted",
                        "quota",
                        "rate",
                        "429",
                        "timeout",
                        "unavailable",
                        "503",
                        "internal",
                        "permission",
                        "403",
                        "401",
                    ]
                )

                if recoverable:
                    print(f"Model/key fallback triggered from {model}: {e}")
                    continue

                # Unknown error: still try remaining options.
                print(f"Model invoke error on {model}: {e}")
                continue

        if last_error:
            raise RuntimeError(f"All configured Gemini models/keys failed. Last error: {last_error}")
        raise RuntimeError("No Gemini clients were initialized.")


def get_fallback_llm():
    global _FALLBACK_LLM_SINGLETON
    if _FALLBACK_LLM_SINGLETON is None:
        _FALLBACK_LLM_SINGLETON = GeminiFallbackLLM()
    return _FALLBACK_LLM_SINGLETON

class FallbackGoogleEmbeddings(Embeddings):
    """
    A custom Embeddings wrapper that intercepts rate-limit errors and gracefully
    falls back to backup API keys.
    """
    def __init__(self, model_name: str):
        keys = get_api_keys()
        self.embedding_clients = [
            GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=key) 
            for key in keys
        ]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        last_exception = None
        for client in self.embedding_clients:
            try:
                return client.embed_documents(texts)
            except (ResourceExhausted, InternalServerError) as e:
                last_exception = e
                print(f"Embedding rate limit hit, switching to backup API key...")
                continue
        raise last_exception if last_exception else ValueError("Embeddings failed entirely.")

    def embed_query(self, text: str) -> List[float]:
        last_exception = None
        for client in self.embedding_clients:
            try:
                return client.embed_query(text)
            except (ResourceExhausted, InternalServerError) as e:
                last_exception = e
                print(f"Embedding rate limit hit, switching to backup API key...")
                continue
        raise last_exception if last_exception else ValueError("Embeddings failed entirely.")

def get_embedding_model_name() -> str:
    embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001").strip()
    if embedding_model.startswith("models/"):
        embedding_model = embedding_model.split("/", 1)[1]

    if embedding_model == "text-embedding-004":
        embedding_model = "gemini-embedding-001"

    return embedding_model

def get_embedding_function():
    global _EMBEDDINGS_SINGLETON
    if _EMBEDDINGS_SINGLETON is not None:
        return _EMBEDDINGS_SINGLETON

    model_name = get_embedding_model_name()
    _EMBEDDINGS_SINGLETON = FallbackGoogleEmbeddings(model_name)
    return _EMBEDDINGS_SINGLETON