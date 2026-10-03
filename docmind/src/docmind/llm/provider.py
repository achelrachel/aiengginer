"""Explicit provider configuration. Keys stay in environment variables."""
from dataclasses import dataclass, field
from urllib.parse import urlsplit, urlunsplit

from docmind.config import Settings


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    model: str
    api_key: str = field(repr=False)
    token_parameter: str = "max_tokens"
    output_mode: str = "json_schema"


def normalize_base_url(value: str, *, allow_local_http: bool = False) -> str:
    url = urlsplit(value.strip())
    if not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError("LLM_BASE_URL must be an API base URL without credentials, query, or fragment")
    local_http = allow_local_http and url.scheme == "http" and url.hostname in {"localhost", "127.0.0.1", "::1", "ollama"}
    if url.scheme != "https" and not local_http:
        raise ValueError("Use HTTPS for remote LLM providers")
    path = url.path.rstrip("/")
    if not path:
        path = "/v1"
    return urlunsplit((url.scheme, url.netloc, path + "/", "", ""))


def resolve_provider(config: Settings) -> ProviderConfig:
    provider = config.llm_provider
    defaults = {
        "ollama": config.ollama_base_url,
        "openai": "https://api.openai.com/v1",
        "xai": "https://api.x.ai/v1",
    }
    base = config.llm_base_url or defaults.get(provider)
    if not base:
        raise ValueError("LLM_BASE_URL is required for the compatible provider")
    # Keep official API keys on their own origin. Custom gateways use compatible.
    parsed = urlsplit(base)
    official_host = {"openai": "api.openai.com", "xai": "api.x.ai"}.get(provider)
    if official_host and (parsed.hostname != official_host or parsed.port not in {None, 443}):
        raise ValueError("Use the compatible provider for custom API gateways")
    model = config.llm_model or (config.ollama_model if provider == "ollama" else None)
    if not model or not model.strip():
        raise ValueError("LLM_MODEL is required for cloud providers; use a model available to your account")
    key = config.llm_api_key.get_secret_value()
    if not key:
        key = {"openai": config.openai_api_key, "xai": config.xai_api_key}.get(provider, config.llm_api_key).get_secret_value()
    if provider != "ollama" and not key.strip():
        raise ValueError("Configure LLM_API_KEY or the selected provider's API key")
    return ProviderConfig(
        name=provider,
        base_url=normalize_base_url(base, allow_local_http=provider in {"ollama", "compatible"}),
        model=model.strip(), api_key=key.strip() or "ollama",
        token_parameter=config.llm_token_parameter or ("max_completion_tokens" if provider == "openai" else "max_tokens"),
        output_mode=config.llm_output_mode,
    )
