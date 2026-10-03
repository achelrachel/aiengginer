"""Run one live structured-output probe with keys supplied through .env/env."""
import asyncio
import json
import sys
import httpx
from docmind.llm.client import LLMClient

SCHEMA = {
    'name': 'readiness_probe', 'strict': True,
    'schema': {'type': 'object', 'properties': {'ok': {'type': 'boolean'}},
               'required': ['ok'], 'additionalProperties': False},
}

async def main():
    client = None
    try:
        client = LLMClient()
        result = await client.chat_completion(
            [{'role': 'user', 'content': 'Return JSON with the field ok set to true.'}],
            response_format=SCHEMA,
        )
        assert json.loads(result.content) == {'ok': True}
        print(json.dumps({'status': 'passed', 'model': client.model, 'latency_ms': result.latency_ms,
                          'input_tokens': result.input_tokens, 'output_tokens': result.output_tokens}))
        return 0
    except Exception as exc:
        report = {'status': 'failed', 'error_type': type(exc).__name__}
        if isinstance(exc, httpx.HTTPStatusError):
            report['http_status'] = exc.response.status_code
        print(json.dumps(report))
        return 1
    finally:
        if client:
            await client.close()

if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
