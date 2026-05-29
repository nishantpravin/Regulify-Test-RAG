"""
ingest.py

Handles PDF loading, text chunking, embedding (local HuggingFace model),
and persisting to ChromaDB.

Run once (or with --rebuild) to build/rebuild the local vector store:
    python ingest.py --pdf data/lintransf.pdf [--rebuild]
"""

import argparse
import logging
import os
import shutil
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------
load_dotenv()

CHROMA_DB_DIR: str = os.getenv("CHROMA_DB_DIR", "chroma_db")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def get_embeddings(model_name: str = EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    """Return a local HuggingFace embedding model (no API key needed).

    Args:
        model_name: Sentence-transformers model name.

    Returns:
        HuggingFaceEmbeddings instance.
    """
    logger.info("Loading local embedding model: %s", model_name)
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_pdf(pdf_path: str) -> List[Document]:
    """Load every page of a PDF and return a list of LangChain Documents.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        List of Document objects, one per page, with page metadata attached.

    Raises:
        FileNotFoundError: If the PDF path does not exist.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info("Loading PDF: %s", path.resolve())
    loader = PyPDFLoader(str(path))
    documents: List[Document] = loader.load()
    logger.info("Loaded %d page(s) from %s", len(documents), path.name)
    return documents


def chunk_documents(documents: List[Document], doc_id: str = "default") -> List[Document]:
    """Split documents into overlapping chunks while preserving metadata and adding doc_id.

    Args:
        documents: List of raw page Documents.
        doc_id: Unique identifier for this document.

    Returns:
        List of smaller chunk Documents, each retaining page/source metadata and doc_id.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    
    # Add doc_id to metadata for filtering
    for chunk in chunks:
        chunk.metadata["doc_id"] = doc_id
        
    logger.info(
        "Split %d page(s) into %d chunk(s) (size=%d, overlap=%d) for doc_id: %s",
        len(documents),
        len(chunks),
        CHUNK_SIZE,
        CHUNK_OVERLAP,
        doc_id
    )
    return chunks


def build_vectorstore(
    chunks: List[Document],
    persist_directory: str = CHROMA_DB_DIR,
) -> Chroma:
    """Embed chunks with a local model and persist them in ChromaDB.

    Args:
        chunks: Pre-chunked Documents to embed.
        persist_directory: Local directory for ChromaDB storage.

    Returns:
        The initialised Chroma vector store.
    """
    embeddings = get_embeddings()

    logger.info(
        "Embedding %d chunk(s) and persisting to '%s' …",
        len(chunks),
        persist_directory,
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    logger.info("Vector store saved to '%s'.", persist_directory)
    return vectorstore


def rebuild_vectorstore(pdf_path: str, persist_directory: str = CHROMA_DB_DIR, doc_id: str = "default") -> Chroma:
    """Delete and fully rebuild the ChromaDB vector store from the given PDF."""
    if Path(persist_directory).exists():
        logger.warning("Removing existing vector store at '%s' …", persist_directory)
        shutil.rmtree(persist_directory)

    documents = load_pdf(pdf_path)
    chunks = chunk_documents(documents, doc_id)
    return build_vectorstore(chunks, persist_directory)


def process_pdf(pdf_path: str, doc_id: str, persist_directory: str = CHROMA_DB_DIR) -> None:
    """Load, chunk, and ADD a PDF to the vector store without deleting the whole store."""
    documents = load_pdf(pdf_path)
    chunks = chunk_documents(documents, doc_id)
    
    embeddings = get_embeddings()
    # Loading existing or creating new
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    vectorstore.add_documents(chunks)
    logger.info("Added document %s to vector store at %s", doc_id, persist_directory)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest a PDF into the ChromaDB vector store for RAG."
    )
    parser.add_argument(
        "--pdf",
        default="data/lintransf.pdf",
        help="Path to the PDF file to ingest (default: data/lintransf.pdf).",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force a complete rebuild of the vector store even if it already exists.",
    )
    args = parser.parse_args()

    if args.rebuild or not Path(CHROMA_DB_DIR).exists():
        rebuild_vectorstore(args.pdf)
    else:
        logger.info(
            "Vector store already exists at '%s'. "
            "Use --rebuild to force re-ingestion.",
            CHROMA_DB_DIR,
        )


if __name__ == "__main__":
    main()
