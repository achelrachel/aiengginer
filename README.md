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

## Cloud LLM providers (OpenAI / xAI Grok)

Local Ollama remains the default. To use a hosted provider, set these variables in the server's `.env` or secret store:

| Provider | Configuration | API base |
| --- | --- | --- |
| OpenAI | `LLM_PROVIDER=openai`, `LLM_MODEL=<available-model-id>`, `OPENAI_API_KEY=<server-secret>` | `https://api.openai.com/v1` |
| xAI/Grok | `LLM_PROVIDER=xai`, `LLM_MODEL=<available-model-id>`, `XAI_API_KEY=<server-secret>` | `https://api.x.ai/v1` |
| Compatible gateway | `LLM_PROVIDER=compatible`, `LLM_MODEL=<model-id>`, `LLM_BASE_URL=<https-api-base>`, `LLM_API_KEY=<server-secret>` | Explicit full API base |

Model IDs are intentionally not guessed or automatically upgraded. Choose a text model that supports Chat Completions and your selected structured-output mode. `LLM_API_KEY` can explicitly override the chosen provider's key. Official providers reject custom origins; select `compatible` for gateways. Remote endpoints require HTTPS. Unversioned root URLs receive `/v1`; already versioned/custom API paths are preserved.

OpenAI uses `max_completion_tokens`; other providers default to `max_tokens`. Override with `LLM_TOKEN_PARAMETER` when your model requires it. Temperature is omitted unless `LLM_TEMPERATURE` is set, avoiding unsupported sampling parameters on reasoning models. `LLM_OUTPUT_MODE=json_schema` is the default; use `json_object` only if the chosen model does not support JSON Schema. Pydantic and citation validation still run in either mode. Empty, refused, and truncated outputs fail closed. Requests have a timeout and token budget, with no automatic paid-provider retries or fallback.

Hosted generation does not replace document embeddings: the sentence-transformer remains local and needs its own dependencies/model cache. Changing embedding models requires reindexing documents.

From `docmind/`, run a single live API probe after configuring credentials:

```bash
PYTHONPATH=src python scripts/check_provider.py
```

This sends one small request that can incur API usage charges. It reports status, model, latency, and token counts, without printing API keys or provider error bodies. A successful probe proves API connectivity and structured output, not document-answer quality. Provider contract tests use HTTP test doubles; no live OpenAI or xAI success is implied by CI.

References: [OpenAI structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs), [xAI structured outputs](https://docs.x.ai/developers/model-capabilities/text/structured-outputs).
