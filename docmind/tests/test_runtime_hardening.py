import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from docmind.config import Settings
from docmind.main import app


def configured(**values):
    return Settings(_env_file=None, **values)


def test_cors_is_closed_by_default():
    assert configured().cors_origins() == []


def test_explicit_https_browser_origins_are_accepted():
    settings = configured(APP_ENV="production", API_KEY="production-test-key",
                          CORS_ALLOWED_ORIGINS="https://docs.example, https://app.example:8443")
    assert settings.cors_origins() == ["https://docs.example", "https://app.example:8443"]


@pytest.mark.parametrize("origins", [
    "*",
    "http://docs.example",
    "https://user:secret@docs.example",
    "https://docs.example/path",
    "https://docs.example?token=secret",
    "https://docs.example#fragment",
    "https://docs.example, https://docs.example",
])
def test_production_rejects_unsafe_cors_origins(origins):
    with pytest.raises(ValidationError):
        configured(APP_ENV="production", API_KEY="production-test-key",
                   CORS_ALLOWED_ORIGINS=origins)


def test_local_development_may_opt_into_localhost():
    assert configured(CORS_ALLOWED_ORIGINS="http://localhost:3000").cors_origins() == [
        "http://localhost:3000"]


@pytest.mark.parametrize("key,value", [
    ("RATE_LIMIT_REQUESTS", 0),
    ("RATE_LIMIT_REQUESTS", 5),
    ("RATE_LIMIT_WINDOW_SECONDS", 0),
    ("MAX_UPLOAD_SIZE_MB", 0),
])
def test_invalid_limits_are_rejected_during_startup(key, value):
    with pytest.raises(ValidationError):
        configured(**{key: value})


def test_untrusted_origin_gets_no_cors_grant():
    with TestClient(app) as client:
        response = client.get("/health", headers={"Origin": "https://untrusted.example"})
        assert response.status_code == 200
        assert "access-control-allow-origin" not in response.headers


@pytest.mark.parametrize("auth_header, expected", [
    ("Bearer test-only-key", 200),
    ("bearer test-only-key", 200),
    ("Basic test-only-key", 401),
    ("test-only-key", 401),
    ("ApiKey test-only-key", 401),
    ("Bearer", 401),
])
def test_authorization_requires_explicit_bearer_scheme(auth_header, expected):
    with TestClient(app) as client:
        response = client.get("/api/v1/documents", headers={"Authorization": auth_header})
        assert response.status_code == expected
