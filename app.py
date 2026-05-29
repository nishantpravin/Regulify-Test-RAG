"""
app.py

FastAPI application exposing the RAG pipeline via a REST API.

Endpoints:
    POST /query           – Ask a question; returns answer + sources.
    DELETE /history/{id}  – Clear chat history for a session.
    GET  /health          – Liveness check.

Start:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag import get_pipeline, RAGPipeline

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
# Pydantic models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """Request body for POST /query."""

    question: str = Field(
        ...,
        min_length=1,
        description="The question to ask about the document.",
        examples=["What is the Rank Nullity Theorem?"],
    )
    session_id: str = Field(
        default="default",
        description="Unique identifier for the conversation session; enables follow-up questions.",
        examples=["user123"],
    )


class SourceItem(BaseModel):
    """A single source reference returned with the answer."""

    page: int = Field(..., description="1-indexed page number from the source document.")
    excerpt: str = Field(..., description="Short text excerpt from the retrieved chunk.")


class QueryResponse(BaseModel):
    """Response body for POST /query."""

    answer: str = Field(..., description="The model's answer based on the document.")
    sources: List[SourceItem] = Field(
        default_factory=list,
        description="Document chunks used to formulate the answer.",
    )


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str
    vector_store_loaded: bool


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load the RAG pipeline once at startup."""
    logger.info("Loading RAG pipeline …")
    try:
        get_pipeline()  # warm-up; raises RuntimeError if not ingested yet
        logger.info("RAG pipeline ready.")
    except RuntimeError as exc:
        logger.error("Failed to load vector store: %s", exc)
        logger.error(
            "Run:  python ingest.py --pdf data/lintransf.pdf  then restart."
        )
    yield
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Document RAG API",
    description=(
        "Ask questions about the ingested PDF document. "
        "Answers include source page references and support multi-turn conversation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post(
    "/query",
    response_model=QueryResponse,
    summary="Query the document",
    status_code=status.HTTP_200_OK,
)
async def query_document(request: QueryRequest) -> QueryResponse:
    """Accept a natural-language question and return an answer with sources.

    The answer is derived exclusively from the ingested PDF document.
    Conversational context is preserved within a ``session_id``.
    """
    try:
        pipeline: RAGPipeline = get_pipeline()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                str(exc)
                + " – run `python ingest.py --pdf <path>` then restart the server."
            ),
        )

    try:
        result: Dict[str, Any] = pipeline.query(
            question=request.question,
            session_id=request.session_id,
        )
    except Exception as exc:
        logger.exception("Query failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {exc}",
        )

    return QueryResponse(
        answer=result["answer"],
        sources=[SourceItem(**s) for s in result["sources"]],
    )

from fastapi.responses import StreamingResponse
import json

@app.post(
    "/query_stream",
    summary="Query the document with streaming chunks",
    status_code=status.HTTP_200_OK,
)
async def query_document_stream(request: QueryRequest) -> StreamingResponse:
    try:
        pipeline: RAGPipeline = get_pipeline()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(str(exc) + " – restart the server.")
        )

    def event_stream():
        try:
            for event in pipeline.stream_query(question=request.question, session_id=request.session_id):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:
            logger.exception("Stream failed: %s", exc)
            yield f"data: {json.dumps({'type': 'error', 'content': str(exc)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.delete(
    "/history/{session_id}",
    summary="Clear conversation history",
    status_code=status.HTTP_200_OK,
)
async def clear_history(session_id: str) -> JSONResponse:
    try:
        pipeline: RAGPipeline = get_pipeline()
        pipeline.clear_history(session_id)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )
    return JSONResponse(
        content={"message": f"History cleared for session '{session_id}'."}
    )

@app.get(
    "/sessions",
    summary="List all active session IDs",
    status_code=status.HTTP_200_OK,
)
async def list_sessions() -> JSONResponse:
    try:
        pipeline: RAGPipeline = get_pipeline()
        return JSONResponse(content={"sessions": pipeline.get_session_ids()})
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

@app.get(
    "/history/{session_id}",
    summary="Get chat history for a specific session",
    status_code=status.HTTP_200_OK,
)
async def get_history(session_id: str) -> JSONResponse:
    try:
        pipeline: RAGPipeline = get_pipeline()
        history = pipeline.get_history(session_id)
        return JSONResponse(content={"history": history})
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    status_code=status.HTTP_200_OK,
)
async def health_check() -> HealthResponse:
    """Return the liveness status and whether the vector store is loaded."""
    loaded = True
    try:
        get_pipeline()
    except RuntimeError:
        loaded = False


# ---------------------------------------------------------------------------
# Static frontend routes
# ---------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=FileResponse, summary="Serve Web UI", status_code=status.HTTP_200_OK)
async def serve_index():
    """Serve the frontend chat interface."""
    return "static/index.html"
