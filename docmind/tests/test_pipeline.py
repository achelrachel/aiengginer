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
    llm._client = httpx.AsyncClient(base_url='http://localhost:11434/', transport=httpx.MockTransport(handle))
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
