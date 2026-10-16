# DocMind

> **Ask questions across documents and receive answers with checkable source citations.**

DocMind is a Python/FastAPI document-question-answering service. Users upload supported files, create a searchable knowledge base, and ask natural-language questions. The answer pipeline combines semantic and keyword retrieval with structured LLM output and citation validation. The implementation is in [docmind/](docmind/).

## What users can do

| Function | Input | Output |
| --- | --- | --- |
| Document ingestion | PDF, TXT, Markdown, DOCX, or PPTX file | Parsed, chunked, indexed content with source metadata |
| Hybrid search | User question and optional metadata filters | Relevant excerpts ranked using embeddings, BM25, reciprocal rank fusion, and cosine reranking |
| Document Q&A | Question and retrieved passages | Structured response containing citations to retrieved document chunks |
| Source verification | Generated answer and cited excerpts | Rejection of citations that do not match retrieved source identity or excerpts |
| Document management | Indexed document records | A maintained search index that updates after ingestion or deletion |
| Query audit | Query execution and status | Duration, status, and available token-usage metadata without storing new raw question/answer content in audit rows |

## Example workflow

1. Upload a supported document through the authenticated API.
2. DocMind parses the text, creates bounded chunks, and indexes them in SQLite and persistent Chroma.
3. Ask a question and optionally restrict retrieval with metadata filters.
4. BM25 and vector search find candidates; fusion and reranking select evidence.
5. The configured model produces a structured answer. Citations must reference retrieved chunks and matching excerpts or the response is rejected.

**Expected result:** an answer tied to inspectable document evidence, rather than an unsupported free-form response. Citation validity proves source linkage, not the factual correctness of every statement.

## Technology

- **API:** FastAPI with API-key authentication, rate limiting, and an interactive OpenAPI endpoint.
- **Retrieval:** sentence-transformer embeddings, persistent Chroma, SQLite metadata, BM25, reciprocal rank fusion, and reranking.
- **Generation:** Ollama-compatible local chat by default; optional OpenAI, xAI/Grok, or compatible HTTPS chat-completions providers.
- **Safety controls:** document format/size validation, bounded chunks, structured output validation, citation rejection, and an explicit CORS allowlist.
- **Audit:** keyed question digests and redacted answer markers for newly created audit rows, with processing metrics when available.

## Run locally

Use Python 3.12 and a local Ollama installation.

```bash
cd docmind
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Set a unique API_KEY in .env
ollama pull qwen2.5:7b
PYTHONPATH=src uvicorn docmind.main:app --host 127.0.0.1 --port 8000
```

Start `ollama serve` separately when needed. The embedding model downloads on its first use unless already cached. API docs: `http://127.0.0.1:8000/docs`.

```bash
curl -H "X-API-Key: $API_KEY" -F 'file=@sop.txt' http://127.0.0.1:8000/api/v1/documents/upload
curl -H "X-API-Key: $API_KEY" -H 'Content-Type: application/json' \
  -d '{"query":"What does the SOP require?","top_k":8}' http://127.0.0.1:8000/api/v1/query
```

Export `API_KEY` in the terminal before the curl commands. It must match the server's configured key.

## Choose a generation provider

Ollama remains the default. For a cloud LLM, configure server-side secrets:

| Provider | Required settings |
| --- | --- |
| OpenAI | `LLM_PROVIDER=openai`, `LLM_MODEL`, `OPENAI_API_KEY` |
| xAI/Grok | `LLM_PROVIDER=xai`, `LLM_MODEL`, `XAI_API_KEY` |
| Compatible gateway | `LLM_PROVIDER=compatible`, `LLM_MODEL`, `LLM_BASE_URL`, `LLM_API_KEY` |

Select an available chat-completions model and an appropriate structured-output mode. Remote providers require HTTPS. Generation does not replace local embeddings; a changed embedding model requires document reindexing.

To probe a configured provider from `docmind/`:

```bash
PYTHONPATH=src python scripts/check_provider.py
```

The probe makes an actual API request and may incur usage costs. It tests connectivity and structured responses, not answer quality.

## Tests

```bash
cd docmind
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-core.txt pytest pytest-asyncio
pytest -q
```

The test suite exercises routes, SQLite, persistent Chroma, citation handling, filters, authentication, and index lifecycle using deterministic model doubles.

## Scope and security

DocMind uses one shared API key for one knowledge base. It is not a multi-tenant document permission system; operate one Uvicorn worker because BM25 is kept in memory. Do not commit files, data directories, credentials, or sensitive content. Production deployment, real-model answer quality, concurrency, backups, and access boundaries require independent validation.
