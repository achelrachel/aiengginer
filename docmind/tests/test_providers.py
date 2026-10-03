import json

import httpx
import pytest

from docmind.config import Settings
from docmind.llm.provider import normalize_base_url, resolve_provider
from docmind.llm.schema import QUERY_ANSWER_SCHEMA
from docmind.llm import client as module


def config(**kwargs):
    return Settings(_env_file=None, **kwargs)


@pytest.mark.parametrize('provider,url,key_field', [
    ('openai', 'https://api.openai.com/v1/', 'OPENAI_API_KEY'),
    ('xai', 'https://api.x.ai/v1/', 'XAI_API_KEY'),
])
def test_official_provider_defaults(provider, url, key_field):
    settings = config(LLM_PROVIDER=provider, LLM_MODEL='test-model', **{key_field: 'test-key'})
    resolved = resolve_provider(settings)
    assert resolved.base_url == url
    assert resolved.model == 'test-model'
    assert resolved.api_key == 'test-key'
    assert 'test-key' not in repr(resolved)
    assert 'test-key' not in repr(settings)


def test_provider_keys_are_not_cross_used():
    s = config(LLM_PROVIDER='xai', LLM_MODEL='model', OPENAI_API_KEY='wrong', XAI_API_KEY='right')
    assert resolve_provider(s).api_key == 'right'


@pytest.mark.parametrize('kwargs', [
    {'LLM_PROVIDER': 'openai'},
    {'LLM_PROVIDER': 'xai', 'LLM_MODEL': 'model'},
    {'LLM_PROVIDER': 'compatible', 'LLM_MODEL': 'model', 'LLM_API_KEY': 'key'},
    {'LLM_PROVIDER': 'openai', 'LLM_MODEL': 'model', 'OPENAI_API_KEY': 'key', 'LLM_BASE_URL': 'https://other.example/v1'},
])
def test_missing_or_misrouted_credentials_fail(kwargs):
    with pytest.raises(ValueError):
        resolve_provider(config(**kwargs))


@pytest.mark.parametrize('value', [
    'http://remote.example/v1', 'https://user:key@example.com/v1',
    'https://example.com/v1?key=value', 'https://example.com/v1#token',
])
def test_reject_unsafe_base_urls(value):
    with pytest.raises(ValueError):
        normalize_base_url(value)


@pytest.mark.parametrize('url,expected', [
    ('https://api.x.ai', 'https://api.x.ai/v1/'),
    ('https://api.x.ai/v1/', 'https://api.x.ai/v1/'),
    ('https://gateway.example/api/v1', 'https://gateway.example/api/v1/'),
])
def test_api_path_is_not_duplicated(url, expected):
    assert normalize_base_url(url) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize('provider,token_parameter', [('openai', 'max_completion_tokens'), ('xai', 'max_tokens')])
async def test_request_payload_and_token_usage(monkeypatch, provider, token_parameter):
    monkeypatch.setattr(module, 'settings', config(LLM_PROVIDER=provider, LLM_MODEL='test-model', LLM_API_KEY='secret-test-only'))
    def handle(request):
        assert request.url.path == '/v1/chat/completions'
        assert request.headers['Authorization'] == 'Bearer secret-test-only'
        payload = json.loads(request.content)
        assert payload[token_parameter] == 1024
        assert 'temperature' not in payload
        assert payload['response_format']['json_schema'] == QUERY_ANSWER_SCHEMA
        return httpx.Response(200, json={
            'choices': [{'message': {'content': '{"answer":"test"}'}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 12, 'completion_tokens': 6},
        })
    real_client = httpx.AsyncClient
    monkeypatch.setattr(module.httpx, "AsyncClient", lambda **kwargs: real_client(**kwargs, transport=httpx.MockTransport(handle)))
    llm = module.LLMClient()
    result = await llm.chat_completion([], response_format=QUERY_ANSWER_SCHEMA)
    assert (result.input_tokens, result.output_tokens) == (12, 6)
    await llm.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('choice', [
    {'message': {'content': '{}'}, 'finish_reason': 'length'},
    {'message': {'content': None, 'refusal': 'blocked'}, 'finish_reason': 'stop'},
    {'message': {'content': ''}, 'finish_reason': 'stop'},
])
async def test_bad_provider_answers_fail_closed(choice):
    llm = module.LLMClient()
    llm._client = httpx.AsyncClient(base_url=llm.base_url, transport=httpx.MockTransport(lambda req: httpx.Response(200, json={'choices': [choice]})))
    with pytest.raises(ValueError):
        await llm.chat_completion([])
    await llm.close()


def test_strict_schema_objects_are_closed_and_required():
    def walk(schema):
        if schema.get('type') == 'object':
            assert schema.get('additionalProperties') is False
            assert set(schema['properties']) == set(schema['required'])
            for value in schema['properties'].values():
                walk(value)
        if schema.get('type') == 'array':
            walk(schema['items'])
    walk(QUERY_ANSWER_SCHEMA['schema'])


@pytest.mark.asyncio
async def test_compatible_json_mode_and_sampling_override(monkeypatch):
    monkeypatch.setattr(module, 'settings', config(
        LLM_PROVIDER='compatible', LLM_BASE_URL='https://gateway.example/api/v1',
        LLM_MODEL='model', LLM_API_KEY='key', LLM_OUTPUT_MODE='json_object',
        LLM_TOKEN_PARAMETER='max_completion_tokens', LLM_TEMPERATURE=.2,
    ))
    def handle(request):
        assert request.url.path == '/api/v1/chat/completions'
        body = json.loads(request.content)
        assert body['response_format'] == {'type': 'json_object'}
        assert body['temperature'] == .2
        assert body['max_completion_tokens'] == 1024
        return httpx.Response(200, json={'choices': [{'message': {'content': '{}'}}]})
    real_client = httpx.AsyncClient
    monkeypatch.setattr(module.httpx, 'AsyncClient', lambda **kwargs: real_client(**kwargs, transport=httpx.MockTransport(handle)))
    llm = module.LLMClient()
    assert (await llm.chat_completion([], response_format=QUERY_ANSWER_SCHEMA)).content == '{}'
    await llm.close()
