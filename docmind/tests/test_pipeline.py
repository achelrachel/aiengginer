import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from docmind.main import app
from docmind.db.models import Document, Chunk, get_session
from docmind.embedding import model
from docmind.llm.client import LLMResponse
from docmind.services import query as query_module
from docmind.retrieval.bm25 import get_bm25_storage, BM25Index
from docmind.retrieval.hybrid import reciprocal_rank_fusion
from docmind.security.validation import is_safe_path, secure_upload_path
from docmind.indexing.chunker import chunk_text
from docmind.config import Settings, settings


class DeterministicEmbedder:
    """Test double only; no claim of real model quality."""
    def embed_text(self, text):
        return [1.0, 0.5, 0.25]
    def embed_batch(self, texts):
        return [self.embed_text(t) for t in texts]


class GroundedLLM:
    model = "test-double"
    async def chat_completion(self, messages, **kwargs):
        async with get_session() as session:
            doc = (await session.execute(select(Document).where(Document.status == 'indexed'))).scalars().first()
        return LLMResponse(json.dumps({
            'answer': 'Laporkan insiden kepada tim keamanan.', 'answer_status': 'answered',
            'citations': [{'doc_id': doc.id, 'filename': doc.original_filename,
                           'chunk_index': 0, 'excerpt': 'Laporkan insiden kepada tim keamanan.',
                           'similarity_score': 1.0}],
        }), input_tokens=10, output_tokens=20, latency_ms=2.5)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(model, '_default_embedder', DeterministicEmbedder())
    monkeypatch.setattr(query_module, 'get_llm_client', lambda: GroundedLLM())
    with TestClient(app) as c:
        yield c


HEADERS = {'X-API-Key': 'test-only-key'}


def test_authentication_and_liveness(client):
    assert client.get('/health').json()['check'] == 'liveness'
    assert client.get('/api/v1/documents').status_code == 401
    assert client.get('/api/v1/documents', headers={'X-API-Key': 'wrong'}).status_code == 403
    assert client.get('/api/v1/documents', headers=HEADERS).status_code == 200


def test_upload_query_filter_and_delete(client):
    body = 'Laporkan insiden kepada tim keamanan.'
    response = client.post('/api/v1/documents/upload', headers=HEADERS,
                           files={'file': ('sop.txt', body, 'text/plain')})
    assert response.status_code == 200, response.text
    doc_id = response.json()['id']
    assert response.json()['status'] == 'indexed'
    assert client.get(f'/api/v1/documents/{doc_id}', headers=HEADERS).status_code == 200
    assert get_bm25_storage().get_index().search('insiden')
    answer = client.post('/api/v1/query', headers=HEADERS, json={'query': 'Bagaimana lapor insiden?'})
    assert answer.status_code == 200, answer.text
    assert answer.json()['answer_status'] == 'answered'
    assert answer.json()['citations'][0]['doc_id'] == doc_id
    assert answer.json()['retrieval_latency_ms'] > 0
    assert answer.json()['estimated_cost_usd'] is None
    filtered = client.post('/api/v1/query', headers=HEADERS,
                           json={'query': 'insiden', 'filters': {'filename': 'other.txt'}})
    assert filtered.json()['answer_status'] == 'refused'
    assert filtered.json()['citations'] == []
    assert client.delete(f'/api/v1/documents/{doc_id}', headers=HEADERS).status_code == 200
    assert not get_bm25_storage().get_index().search('insiden')
    assert client.get(f'/api/v1/documents/{doc_id}', headers=HEADERS).status_code == 404
    assert client.post('/api/v1/query', headers=HEADERS, json={'query': 'insiden'}).json()['answer_status'] == 'refused'


def test_metadata_survives_fusion():
    result = reciprocal_rank_fusion([('chunk1', 1)], [('chunk1', .1, {'filename': 'sop.txt'})])
    assert result[0]['metadata']['filename'] == 'sop.txt'


def test_bm25_upsert_does_not_duplicate():
    index = BM25Index()
    index.add_document('c', 'old')
    index.add_document('c', 'new')
    assert len(index) == 1
    assert not index.search('old')
    assert len(index.search('new')) == 1


def test_paths_and_long_paragraph():
    assert secure_upload_path('../../sop.txt').suffix == '.txt'
    assert not is_safe_path(Path('/tmp/data-other/a'), Path('/tmp/data'))
    chunks = chunk_text('x' * 10000, 'doc1')
    assert len(chunks) > 1
    assert all(len(c.text) <= settings.chunk_size_tokens * 4 for c in chunks)


def test_production_key_and_invalid_overlap():
    with pytest.raises(ValueError):
        Settings(APP_ENV='production', API_KEY='dev-key-change-me', _env_file=None)
    with pytest.raises(ValueError):
        Settings(CHUNK_SIZE_TOKENS=4, CHUNK_OVERLAP_TOKENS=4, _env_file=None)


def test_reject_invalid_file(client):
    r = client.post('/api/v1/documents/upload', headers=HEADERS,
                    files={'file': ('fake.pdf', b'not a pdf', 'application/pdf')})
    assert r.status_code == 400


def test_unverifiable_citation_is_refused(client, monkeypatch):
    class InvalidLLM:
        model = 'test-double'
        async def chat_completion(self, **kwargs):
            return LLMResponse(json.dumps({
                'answer': 'Made up answer', 'answer_status': 'answered',
                'citations': [{'doc_id': 'made-up', 'filename': 'fake.txt', 'chunk_index': 0,
                               'excerpt': 'invented quote', 'similarity_score': 1}],
            }))
    monkeypatch.setattr(query_module, 'get_llm_client', lambda: InvalidLLM())
    r = client.post('/api/v1/documents/upload', headers=HEADERS,
                    files={'file': ('source.txt', 'Verified source text.', 'text/plain')})
    doc_id = r.json()['id']
    answer = client.post('/api/v1/query', headers=HEADERS, json={'query': 'source text'})
    assert answer.status_code == 200, answer.text
    assert answer.json()['answer_status'] == 'refused'
    assert not answer.json()['citations']
    client.delete(f'/api/v1/documents/{doc_id}', headers=HEADERS)


@pytest.mark.asyncio
async def test_llm_endpoint_and_schema():
    import httpx
    from docmind.llm.client import LLMClient
    from docmind.llm.schema import QUERY_ANSWER_SCHEMA
    def handle(request):
        assert str(request.url) == 'http://localhost:11434/v1/chat/completions'
        payload = json.loads(request.content)
        assert payload['response_format']['type'] == 'json_schema'
        assert payload['response_format']['json_schema']['name'] == 'query_answer'
        return httpx.Response(200, json={'choices': [{'message': {'content': '{}'}}]})
    llm = LLMClient()
    llm._client = httpx.AsyncClient(base_url='http://localhost:11434/v1/', transport=httpx.MockTransport(handle))
    assert (await llm.chat_completion([], response_format=QUERY_ANSWER_SCHEMA)).content == '{}'
    await llm.close()


def test_index_rebuild_on_restart(monkeypatch):
    from docmind.retrieval.bm25 import reset_bm25
    from docmind.vectorstore.client import reset_vector_store
    monkeypatch.setattr(model, '_default_embedder', DeterministicEmbedder())
    with TestClient(app) as client:
        r = client.post('/api/v1/documents/upload', headers=HEADERS,
                        files={'file': ('persist.txt', 'Persistent unique keyword.', 'text/plain')})
        assert r.status_code == 200, r.text
        doc_id = r.json()['id']
    reset_bm25()
    reset_vector_store()
    with TestClient(app) as client:
        assert get_bm25_storage().get_index().search('unique')
        assert client.delete(f'/api/v1/documents/{doc_id}', headers=HEADERS).status_code == 200


def test_vector_failure_marks_document_failed(client, monkeypatch):
    from docmind.vectorstore.client import get_vector_store
    async def fail(*args, **kwargs):
        raise RuntimeError('test-only indexing failure')
    monkeypatch.setattr(get_vector_store(), 'upsert_chunks', fail)
    r = client.post('/api/v1/documents/upload', headers=HEADERS,
                    files={'file': ('failure.txt', 'Failed source.', 'text/plain')})
    assert r.status_code == 500
    assert 'test-only' not in r.text
    docs = client.get('/api/v1/documents', headers=HEADERS).json()['documents']
    failed = [d for d in docs if d['filename'] == 'failure.txt']
    assert len(failed) == 1 and failed[0]['status'] == 'failed'
    assert not get_bm25_storage().get_index().search('Failed')


@pytest.mark.asyncio
async def test_query_audit_never_persists_raw_question_or_answer():
    from docmind.db.models import QueryAudit, init_db
    from docmind.services.query import QueryService

    await init_db()
    private_question = "Confidential incident number 12345 belongs to this document"
    private_answer = "Private incident details must not leak through an audit trail"
    await QueryService()._audit_log(
        query=private_question,
        response_data={"answer": private_answer, "answer_status": "answered",
                       "total_latency_ms": 10.0, "input_tokens": 2, "output_tokens": 4},
        retrieval_count=1, retrieval_latency_ms=4.0, llm_latency_ms=6.0,
    )
    async with get_session() as session:
        audits = (await session.execute(select(QueryAudit))).scalars().all()
    matching = [row for row in audits
                if row.query_text.startswith("hmac-sha256:")
                and row.answer_status == "answered"]
    assert matching
    stored = matching[-1]
    assert len(stored.query_text) == len("hmac-sha256:") + 64
    assert stored.query_text != private_question
    assert private_question not in stored.query_text
    assert private_answer not in stored.response_text
    assert stored.response_text == "[redacted]"
    assert stored.input_tokens == 2
    assert stored.output_tokens == 4


@pytest.mark.asyncio
async def test_partial_chroma_write_is_compensated_when_ingest_fails(client, monkeypatch):
    from docmind.vectorstore.client import get_vector_store
    store = get_vector_store()
    original_upsert = store.upsert_chunks

    async def partially_commit_then_fail(chunks, embeddings):
        await original_upsert(chunks, embeddings)
        raise RuntimeError('synthetic post-write failure with private input')

    monkeypatch.setattr(store, 'upsert_chunks', partially_commit_then_fail)
    result = client.post('/api/v1/documents/upload', headers=HEADERS,
                         files={'file': ('partial.txt', 'Private uniquerollback phrase.', 'text/plain')})
    assert result.status_code == 500
    async with get_session() as session:
        row = (await session.execute(select(Document).where(
            Document.original_filename == 'partial.txt'))).scalars().first()
    assert row is not None
    assert row.status == 'failed'
    assert row.error_message == 'INGESTION_FAILED_RuntimeError'
    assert 'private input' not in row.error_message
    remaining = await store.get_chunks(where={'doc_id': row.id})
    assert remaining['ids'] == []
    assert not get_bm25_storage().get_index().search('uniquerollback')


def test_new_lifespan_discards_stale_keyword_index_entries(monkeypatch):
    from docmind.retrieval.bm25 import get_bm25_storage
    from docmind.embedding import model
    index = get_bm25_storage().get_index()
    index.add_document('orphan-not-in-db', 'orphankeywordnameddoc')
    assert index.search('orphankeywordnameddoc')
    monkeypatch.setattr(model, '_default_embedder', DeterministicEmbedder())
    with TestClient(app):
        assert not get_bm25_storage().get_index().search('orphankeywordnameddoc')


def test_bm25_batch_matches_individual_updates_and_rebuilds_once(monkeypatch):
    corpus = [(f"chunk-{i}", f"common token{i} a repeated term") for i in range(150)]
    expected = BM25Index()
    for chunk_id, text in corpus:
        expected.add_document(chunk_id, text)

    batched = BM25Index()
    original_recompute = batched._recompute_idf
    called = []

    def observed_recompute():
        called.append(True)
        original_recompute()

    monkeypatch.setattr(batched, '_recompute_idf', observed_recompute)
    batched.add_documents(corpus)
    assert len(called) == 1
    assert batched.document_count() == expected.document_count()
    assert batched.search('common repeated', top_k=10) == expected.search(
        'common repeated', top_k=10,
    )

    batched.add_documents([('chunk-4', 'replaced distinctterm')])
    assert batched.document_count() == len(corpus)
    assert batched.search('distinctterm', top_k=10)[0][0] == 'chunk-4'
    assert len(called) == 2


@pytest.mark.parametrize("invalid_filters", [
    {}, {"filename": "sop.txt", "doc_id": "other"},
    {"$or": [{"filename": "sop.txt"}]},
    {"filename": {"$ne": "sop.txt"}},
    {"filename": ""}, {"filename": " "}, {"doc_id": 7},
    {"filename": "x" * 256}, {"unsupported": "value"},
])
def test_query_rejects_unsupported_metadata_filters(client, invalid_filters):
    result = client.post("/api/v1/query", headers=HEADERS,
                         json={"query": "insiden", "filters": invalid_filters})
    assert result.status_code == 422


def test_query_accepts_document_id_metadata_filter():
    from docmind.models.query import QueryRequest
    parsed = QueryRequest.model_validate({
        "query": "  bagaimana lapor insiden?  ",
        "filters": {"doc_id": "  doc-42  "},
    })
    assert parsed.query == "bagaimana lapor insiden?"
    assert parsed.filters == {"doc_id": "doc-42"}
