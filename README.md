# Document RAG Application

A production-ready Retrieval-Augmented Generation (RAG) application using FastAPI, LangChain, ChromaDB, HuggingFace embeddings, and Ollama. **This implementation runs 100% locally and free, with zero API costs.**

## Architecture
- **Embeddings**: `all-MiniLM-L6-v2` (via HuggingFace sentence-transformers)
- **Vector Database**: ChromaDB (persisted locally)
- **LLM**: `qwen2.5:1.5b` (via Ollama)
- **API**: FastAPI

---

## Setup Instructions

1. Ensure you have **Python 3.9+** installed and setup a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
4. **Install and run Ollama** (required for the LLM component):
   - Make sure Ollama is installed.
   - Run the server: `ollama serve`
   - Pull the model: `ollama pull qwen2.5:1.5b`

5. Place your target PDF inside the `data` folder (e.g., `data/lintransf.pdf`).

---

## Data Ingestion

Run the ingestion script to parse the PDF, chunk it, and store embeddings locally using ChromaDB:
```bash
python ingest.py --pdf data/lintransf.pdf
```
*Note: You only need to run this once per document or when you want to rebuild the vector database.*

---

## Running the API

Start the FastAPI application:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## Example Requests

Send a query using your preferred HTTP client or via the Interactive Docs at `http://localhost:8000/docs`:

**Request**:
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{
           "question": "What is the Rank Nullity Theorem?",
           "session_id": "user123"
         }'
```

**Response**:
```json
{
  "answer": "The Rank-Nullity Theorem states that...",
  "sources": [
    {
      "page": 44,
      "excerpt": "Theorem 3.4 (Rank-Nullity Theorem)..."
    }
  ]
}
```

*Conversational context is maintained based on `session_id`. Subsequent requests with the same session id will remember prior interactions.*
