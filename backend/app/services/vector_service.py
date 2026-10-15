"""
Vector database service: manages guideline embeddings and retrieval.
"""

import os
import logging

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings, LLMProvider

logger = logging.getLogger(__name__)
settings = get_settings()

_retriever = None


def _get_embeddings():
    """Get the embedding function based on config."""
    if settings.LLM_PROVIDER == LLMProvider.OPENAI:
        try:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(
                openai_api_key=settings.OPENAI_API_KEY,
                model="text-embedding-3-small",
            )
        except ImportError:
            logger.warning("langchain-openai not installed, falling back to Ollama embeddings")

    elif settings.LLM_PROVIDER == LLMProvider.OPENROUTER:
        try:
            from langchain_openai import OpenAIEmbeddings
            embedding_model = "openai/text-embedding-3-small"
            logger.info("Using OpenRouter embeddings: model=%s", embedding_model)
            return OpenAIEmbeddings(
                openai_api_key=settings.OPENROUTER_API_KEY,
                openai_api_base=settings.OPENROUTER_BASE_URL,
                model=embedding_model,
            )
        except Exception as e:
            logger.warning("OpenRouter embeddings failed (%s), falling back to Ollama", e)

    # Default fallback to Ollama embeddings
    logger.info("Using Ollama embeddings (%s at %s)", settings.OLLAMA_MODEL, settings.OLLAMA_HOST)
    return OllamaEmbeddings(
        base_url=settings.OLLAMA_HOST,
        model=settings.OLLAMA_MODEL,
    )



def build_vector_db(guidelines_path: str | None = None):
    """
    Build (or reload) the vector database from the guidelines file.
    Returns a retriever.
    """
    global _retriever

    path = guidelines_path or settings.effective_guidelines_path

    if not os.path.exists(path):
        raise FileNotFoundError(f"Guidelines file not found at: {path}")

    logger.info("Loading guidelines from: %s", path)
    loader = TextLoader(path)
    documents = loader.load()

    logger.info("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(documents)
    logger.info("Created %d chunks", len(chunks))

    embeddings = _get_embeddings()

    logger.info("Building vector database (persist_dir=%s)...", settings.CHROMA_PERSIST_DIR)
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=settings.CHROMA_COLLECTION_NAME,
        persist_directory=settings.CHROMA_PERSIST_DIR if settings.CHROMA_PERSIST_DIR else None,
    )

    _retriever = vector_db.as_retriever(
        search_kwargs={"k": settings.VECTOR_SEARCH_K}
    )
    logger.info("Vector database ready (k=%d)", settings.VECTOR_SEARCH_K)
    return _retriever


def get_retriever():
    """Get the current retriever instance."""
    global _retriever
    if _retriever is None:
        return build_vector_db()
    return _retriever
