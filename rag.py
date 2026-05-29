
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


_CONVERSATIONAL_PATTERNS = {
    "hi", "hello", "hey", "howdy", "greetings",
    "how are you", "how are you doing", "how's it going",
    "good morning", "good afternoon", "good evening", "good night",
    "thanks", "thank you", "cheers", "bye", "goodbye", "see you",
    "ok", "okay", "sure", "great", "awesome", "cool", "nice",
    "what's up", "sup", "yo",
}

def _is_conversational(question: str) -> bool:
    """Return True if the question looks like a short greeting or chit-chat."""
    q = question.strip().lower().rstrip("?!.,")
    if q in _CONVERSATIONAL_PATTERNS:
        return True
    # Short sentences (≤ 4 words) are treated as conversational
    if len(q.split()) <= 4 and not any(kw in q for kw in ["what", "how", "why", "when", "where", "who", "explain", "define", "describe"]):
        return True
    return False


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
             "You are a helpful assistant for Regulify, an AI-powered regulatory document tool.\n"
             "You have two modes of answering:\n"
             "1. If the question is related to the document context below, answer it directly from the context.\n"
             "2. If the question is a general knowledge, conversational, or factual question (e.g., 'What is the capital of India?', 'Hello', greetings), answer it naturally and helpfully from your own knowledge. For these cases, begin your response with the prefix 'NO_CONTEXT_USED:'.\n"
             "3. If the question is pure gibberish, random characters, or completely incoherent, politely ask the user to rephrase and begin your response with 'NO_CONTEXT_USED:'.\n"
             "Always be polite, clear, and concise. Do NOT append source lists or excerpts at the end.\n\n"
             "Document Context:\n{context}"),
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

        # Conversational bypass: skip distance filter for greetings/chit-chat
        if _is_conversational(question):
            logger.info("[session=%s] Conversational query detected, bypassing filter.", session_id)
            chain = self._prompt | self._llm | StrOutputParser()
            answer = chain.invoke({"input": question, "chat_history": history, "context": ""}).strip()
            if answer.startswith("NO_CONTEXT_USED:"):
                answer = answer.replace("NO_CONTEXT_USED:", "", 1).strip()
            history.append(HumanMessage(content=question))
            history.append(AIMessage(content=answer))
            return {"answer": answer, "sources": []}

        # Retrieve relevant chunks with score (L2 distance)
        retrieved = self._vectorstore.similarity_search_with_score(question, k=TOP_K)
        
        # Pre-retrieval filtering: reject ONLY genuine gibberish (very high distance threshold)
        if not retrieved or retrieved[0][1] > 1.6:
            logger.info("Gibberish/no-match detected! Min distance: %f", retrieved[0][1] if retrieved else -1)
            fallback = "I didn't quite catch that! Could you rephrase? I can answer questions about the Regulify document, or general knowledge questions too."
            history.append(HumanMessage(content=question))
            history.append(AIMessage(content=fallback))
            return {"answer": fallback, "sources": []}

        docs: List[Document] = [doc for doc, _ in retrieved]
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

    from typing import Generator
    def stream_query(self, question: str, session_id: str = "default") -> Generator[Dict[str, Any], None, None]:
        """Answer a question using fully local RAG via streaming chunks."""
        history: List[BaseMessage] = _chat_histories.setdefault(session_id, [])
        logger.info("[session=%s] Stream Query: %s (history=%d)", session_id, question, len(history))

        # Conversational bypass: skip distance filter for greetings/chit-chat
        if _is_conversational(question):
            logger.info("[session=%s] Conversational stream query, bypassing filter.", session_id)
            yield {"type": "sources", "sources": []}
            chain = self._prompt | self._llm | StrOutputParser()
            full_answer = ""
            for chunk in chain.stream({"input": question, "chat_history": history, "context": ""}):
                full_answer += chunk
                # Strip prefix if model adds it
                text_to_yield = chunk
                if full_answer.startswith("NO_CONTEXT_USED:") and len(full_answer) <= len("NO_CONTEXT_USED:"):
                    text_to_yield = ""
                yield {"type": "chunk", "text": text_to_yield}
            full_answer = full_answer.replace("NO_CONTEXT_USED:", "", 1).strip()
            history.append(HumanMessage(content=question))
            history.append(AIMessage(content=full_answer))
            return

        # Retrieve relevant chunks with score
        retrieved = self._vectorstore.similarity_search_with_score(question, k=TOP_K)

        if not retrieved or retrieved[0][1] > 1.6:
            logger.info("Gibberish/no-match detected! Min distance: %f", retrieved[0][1] if retrieved else -1)
            fallback = "I didn't quite catch that! Could you rephrase? I can answer questions about the Regulify document, or general knowledge questions too."
            yield {"type": "sources", "sources": []}
            yield {"type": "chunk", "text": fallback}
            history.append(HumanMessage(content=question))
            history.append(AIMessage(content=fallback))
            return

        docs: List[Document] = [doc for doc, _ in retrieved]
        context: str = _format_docs(docs)

        # Build and stream the chain
        chain = self._prompt | self._llm | StrOutputParser()
        
        # Send sources first
        final_sources = _format_sources(docs)
        yield {"type": "sources", "sources": final_sources}

        full_answer = ""
        # stream all chunks one by one
        for chunk in chain.stream({
            "input": question,
            "chat_history": history,
            "context": context,
        }):
            full_answer += chunk
            yield {"type": "chunk", "text": chunk}

        full_answer = full_answer.strip()
        if full_answer.startswith("NO_CONTEXT_USED:"):
            # UI should hide "NO_CONTEXT_USED:" if it can, but streaming has already sent it.
            # Easiest way is for front end to strip it.
            full_answer = full_answer.replace("NO_CONTEXT_USED:", "", 1).strip()
            final_sources = []

        # Update history
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=full_answer))

    def clear_history(self, session_id: str = "default") -> None:
        """Clear chat history for a session."""
        _chat_histories.pop(session_id, None)
        logger.info("History cleared for session '%s'.", session_id)

    def get_history(self, session_id: str = "default") -> List[Dict[str, str]]:
        """Return the chat history as a list of dictionaries for the UI."""
        history = _chat_histories.get(session_id, [])
        formatted = []
        for msg in history:
            sender = "user" if isinstance(msg, HumanMessage) else "ai"
            formatted.append({"sender": sender, "text": msg.content})
        return formatted

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
