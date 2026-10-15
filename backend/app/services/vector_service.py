"""
Vector database service: manages guideline embeddings and retrieval.

Guidelines come either from a plain .txt file or from a page-level .jsonl corpus
(see scripts/build_hud_corpus.py). Every chunk of the .jsonl corpus keeps its source
`page` and `section` as metadata so answers can cite the handbook.
"""

import json
import os
import logging

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings, LLMProvider

logger = logging.getLogger(__name__)
settings = get_settings()

_retriever = None


def get_embeddings():
    """Get the embedding function based on config (never a local model server)."""
    if settings.LLM_PROVIDER == LLMProvider.OPENROUTER:
        from langchain_openai import OpenAIEmbeddings
        logger.info("Using OpenRouter embeddings: model=%s", settings.EMBEDDING_MODEL)
        return OpenAIEmbeddings(
            api_key=settings.OPENROUTER_API_KEY,
            base_url=settings.OPENROUTER_BASE_URL,
            model=settings.EMBEDDING_MODEL,
            check_embedding_ctx_length=False,  # send raw strings, not tiktoken ids
        )
    if settings.LLM_PROVIDER == LLMProvider.OPENAI:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY, model="text-embedding-3-small")

    from langchain_ollama import OllamaEmbeddings
    logger.info("Using Ollama embeddings (%s at %s)", settings.OLLAMA_MODEL, settings.OLLAMA_HOST)
    return OllamaEmbeddings(base_url=settings.OLLAMA_HOST, model=settings.OLLAMA_MODEL)


def load_guideline_documents(path: str) -> list[Document]:
    """Load guideline source documents (one per page for .jsonl, one per file for .txt)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Guidelines file not found at: {path}")
    if path.endswith(".jsonl"):
        docs = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                docs.append(Document(
                    page_content=rec["text"],
                    metadata={"page": rec["page"], "section": rec.get("section", ""),
                              "source": "HUD Handbook 4000.1"},
                ))
        return docs
    return TextLoader(path).load()


def split_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Split documents into chunks; metadata (page, section) is inherited."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)


def build_vector_db(
    guidelines_path: str | None = None,
    force_rebuild: bool = False,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    k: int | None = None,
    embeddings=None,
    persist_dir: str | None = None,
    collection_name: str | None = None,
):
    """
    Build (or reload) the vector database from the guidelines file and return a retriever.

    An existing persisted collection is reused unless `force_rebuild` is set, so a restart
    does not re-embed (or duplicate) the corpus. All parameters default to settings;
    `embeddings` can be injected for tests and offline evaluation.
    """
    global _retriever

    path = guidelines_path or settings.effective_guidelines_path
    chunk_size = chunk_size or settings.CHUNK_SIZE
    chunk_overlap = settings.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
    k = k or settings.VECTOR_SEARCH_K
    persist_dir = persist_dir if persist_dir is not None else settings.CHROMA_PERSIST_DIR
    collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
    embeddings = embeddings or get_embeddings()

    store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir or None,
    )
    if force_rebuild:
        store.delete_collection()
        store = Chroma(collection_name=collection_name, embedding_function=embeddings,
                       persist_directory=persist_dir or None)

    if store._collection.count() == 0:
        logger.info("Loading guidelines from: %s", path)
        chunks = split_documents(load_guideline_documents(path), chunk_size, chunk_overlap)
        logger.info("Embedding %d chunks (size=%d, overlap=%d)", len(chunks), chunk_size, chunk_overlap)
        store.add_documents(chunks)
    else:
        logger.info("Reusing persisted collection '%s' (%d chunks)",
                    collection_name, store._collection.count())

    retriever = store.as_retriever(search_kwargs={"k": k})
    _retriever = retriever
    logger.info("Vector database ready (k=%d)", k)
    return retriever


def get_retriever():
    """Get the current retriever instance."""
    global _retriever
    if _retriever is None:
        return build_vector_db()
    return _retriever
