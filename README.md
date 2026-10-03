# DocMind — document question answering with RAG

DocMind is a Python/FastAPI portfolio application for uploading documents and asking questions with source citations. The implementation lives in [`docmind/`](docmind/).

It combines sentence-transformer embeddings and persistent Chroma retrieval with BM25 keyword search, reciprocal rank fusion, and cosine-similarity reranking. An Ollama-compatible chat endpoint generates structured answers. Citations are checked against the retrieved document/chunk and source excerpt; invalid citations cause a refusal. This verifies citation provenance, not whether every answer statement follows logically from its source.

## Run locally

Python 3.12 and Ollama are recommended. From the repository root:

```bash
cd docmind
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Set your own API_KEY in .env.
ollama pull qwen2.5:7b
PYTHONPATH=src uvicorn docmind.main:app --host 127.0.0.1 --port 8000
```

Start Ollama separately (`ollama serve`) if it is not already running. The sentence-transformer model downloads on first use and requires internet access or an existing local model cache. API docs: `http://127.0.0.1:8000/docs`.

```bash
curl -H "X-API-Key: $API_KEY" -F 'file=@sop.txt' http://127.0.0.1:8000/api/v1/documents/upload
curl -H "X-API-Key: $API_KEY" -H 'Content-Type: application/json' \
  -d '{"query":"What does the SOP require?","top_k":8}' http://127.0.0.1:8000/api/v1/query
```

The shell variable `API_KEY` must match the value in `.env`; copying the file alone does not export shell variables.

## Implementation

- Upload and parse PDF, TXT, Markdown, DOCX, and PPTX; validate extensions, selected file signatures, and upload size.
- Create bounded text chunks and store metadata in SQLite; use explicit application embeddings in persistent Chroma.
- Rebuild BM25 from indexed database chunks at startup, update after ingestion, and remove entries when documents are deleted.
- Apply metadata filters to both dense and keyword candidates before reranking.
- Return structured answers, source citations, query audit records, token usage when provided, and measured processing latency. Monetary cost is unset because provider pricing is not configured.
- Protect document/query endpoints with API-key authentication and rate limits. Production configuration rejects the development default key.
- `/health` is a liveness check. It does not certify database, vector-store, or model availability.

## Tests

```bash
cd docmind
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-core.txt pytest pytest-asyncio
pytest -q
```

Tests exercise real FastAPI routes, SQLite, and persistent Chroma with deterministic embedding and LLM test doubles. They cover document lifecycle, filtered retrieval, citation rejection, failure status, index rebuild, authentication, chunk bounds, and the chat endpoint/schema envelope. They do **not** measure answer accuracy or real-model latency. GitHub Actions runs this suite.

## Scope and remaining validation

This is a local, single-process portfolio demo, not a verified production deployment. Run one Uvicorn worker because BM25 is an in-memory index. The shared API key grants access to the entire knowledge base; tenant isolation and per-user permissions are not implemented. Model downloads, real Ollama generation, retrieval-quality benchmarks, concurrency/load tests, deployment monitoring, and backups need separate validation. Prompt-injection pattern logging is a heuristic, not a security guarantee.

Use a fresh demo data directory after schema changes. Existing SQLite files need a migration before reuse; do not delete valuable data to apply an update. Do not commit document contents, `.env`, API keys, or the `data/` directory.
