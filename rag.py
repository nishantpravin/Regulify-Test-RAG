
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

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
# Config
# ---------------------------------------------------------------------------
load_dotenv()

CHROMA_DB_DIR: str = os.getenv("CHROMA_DB_DIR", "chroma_db")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
TOP_K: int = int(os.getenv("TOP_K", "5"))

# In-memory session store
_chat_histories: Dict[str, List[BaseMessage]] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_embeddings() -> HuggingFaceEmbeddings:
    """Return the local HuggingFace embedding model (no API key required)."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_vectorstore(persist_directory: str = CHROMA_DB_DIR) -> Chroma:
    """Load an existing ChromaDB vector store from disk.

    Raises:
        RuntimeError: If the vector store directory does not exist.
    """
    if not Path(persist_directory).exists():
        raise RuntimeError(
            f"Vector store not found at '{persist_directory}'. "
            "Please run: python ingest.py --pdf <path-to-pdf>"
        )
    logger.info("Loading vector store from '%s' …", persist_directory)
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=_get_embeddings(),
    )


def _format_docs(docs: List[Document]) -> str:
    """Concatenate retrieved page contents for the prompt context."""
    return "\n\n".join(doc.page_content for doc in docs)


def _format_sources(docs: List[Document]) -> List[Dict[str, Any]]:
    """Convert retrieved docs into the structured sources list."""
    seen: set = set()
    sources: List[Dict[str, Any]] = []
    for doc in docs:
        page: int = doc.metadata.get("page", 0) + 1  # 0-indexed → 1-indexed
        excerpt: str = doc.page_content[:300].replace("\n", " ").strip()
        if page not in seen:
            sources.append({"page": page, "excerpt": excerpt})
            seen.add(page)
    return sources


# ---------------------------------------------------------------------------
# RAGPipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """100% local RAG pipeline — no API keys, no costs.

    Uses:
      - HuggingFace all-MiniLM-L6-v2 for embeddings
      - Ollama qwen2.5:1.5b for answer generation

    Example::

        pipeline = RAGPipeline()
        result = pipeline.query("What is the Rank Nullity Theorem?")
        print(result["answer"])
    """

    def __init__(self, persist_directory: str = CHROMA_DB_DIR) -> None:
        self._vectorstore: Chroma = load_vectorstore(persist_directory)
        self._retriever = self._vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K},
        )
        self._llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0,
        )
        self._prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful and concise assistant answering questions based only on the "
             "provided document context.\n"
             "Answer the question directly in your own words. Do NOT simply copy/paste the raw context blocks.\n"
             "If the user's question can be answered using the provided context, answer it naturally.\n"
             "If the user's question is entirely unrelated to the provided document (e.g., 'Hello', 'What is your name?'), "
             "you must answer it conversationally, BUT you MUST begin your response exactly with the prefix 'NO_CONTEXT_USED:'.\n"
             "Do NOT append a list of references, excerpts, or sources at the end of your answer.\n\n"
             "Context:\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

    def query(self, question: str, session_id: str = "default") -> Dict[str, Any]:
        """Answer a question using fully local RAG, retaining conversation context.

        Args:
            question:   The user's question.
            session_id: Conversation session identifier.

        Returns:
            Dict with ``answer`` (str) and ``sources`` (list of dicts).
        """
        history: List[BaseMessage] = _chat_histories.setdefault(session_id, [])
        logger.info("[session=%s] Query: %s (history=%d)", session_id, question, len(history))

        # Retrieve relevant chunks
        docs: List[Document] = self._retriever.invoke(question)
        context: str = _format_docs(docs)

        # Build and invoke the chain
        chain = self._prompt | self._llm | StrOutputParser()
        answer: str = chain.invoke({
            "input": question,
            "chat_history": history,
            "context": context,
        })

        answer = answer.strip()
        final_sources = []
        if answer.startswith("NO_CONTEXT_USED:"):
            answer = answer.replace("NO_CONTEXT_USED:", "", 1).strip()
            final_sources = []
        else:
            final_sources = _format_sources(docs)

        # Update history
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=answer))

        return {"answer": answer, "sources": final_sources}

    def clear_history(self, session_id: str = "default") -> None:
        """Clear chat history for a session."""
        _chat_histories.pop(session_id, None)
        logger.info("History cleared for session '%s'.", session_id)

    @staticmethod
    def get_session_ids() -> List[str]:
        """Return all active session IDs."""
        return list(_chat_histories.keys())


# Module-level singleton
_pipeline_instance: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    """Return the shared RAGPipeline singleton (lazy-initialised)."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = RAGPipeline()
    return _pipeline_instance
