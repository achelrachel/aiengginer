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


def test_rate_limited_http_response_uses_numeric_retry_after():
    from types import SimpleNamespace
    from docmind.rate_limit import rate_limit_exceeded_handler
    from docmind.config import settings

    response = rate_limit_exceeded_handler(None, SimpleNamespace(detail="10 per 1 minute"))
    assert response.status_code == 429
    assert response.headers["retry-after"].isdigit()
    assert int(response.headers["retry-after"]) >= max(60, settings.rate_limit_window_seconds)
    assert b'"retry_after_seconds"' in response.body
    assert b'"retry_after":"10 per 1 minute"' in response.body


def test_ingestion_logs_do_not_interpolate_user_document_names_or_temporary_paths():
    """Prevent accidental sensitive filename/path leakage into production logs."""
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "docmind"
    for path in (root / "api" / "routes.py", root / "services" / "ingestion.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if not isinstance(node.func.value, ast.Name) or node.func.value.id != "logger":
                continue
            assert all(k.arg not in {"filename", "path"} for k in node.keywords), (path, node.lineno)


def test_office_upload_rejects_non_zip_even_with_matching_magic(tmp_path):
    from docmind.security.validation import validate_file_type
    fake = tmp_path / "fake.docx"
    fake.write_bytes(bytes((80, 75, 3, 4)) + b"not-a-central-directory")
    is_valid, _ = validate_file_type(fake)
    assert is_valid is False


def test_office_upload_rejects_path_traversal_in_archive(tmp_path):
    from zipfile import ZipFile
    from docmind.security.validation import validate_file_type
    malicious = tmp_path / "traversal.docx"
    with ZipFile(malicious, "w") as archive:
        archive.writestr("../payload.xml", "<p/>")
    assert validate_file_type(malicious)[0] is False


def test_office_upload_rejects_decompression_bomb_before_parsing(tmp_path, monkeypatch):
    from docmind.security import validation
    archive_path = tmp_path / "large.docx"
    archive_path.write_bytes(bytes((80, 75, 3, 4)))
    class Entry:
        filename = "word/document.xml"
        file_size = validation.MAX_ARCHIVE_EXPANDED_BYTES + 1
        flag_bits = 0
    class FakeZip:
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            return False
        def infolist(self):
            return [Entry()]
    monkeypatch.setattr(validation, "ZipFile", lambda _path: FakeZip())
    valid, reason = validation.validate_file_type(archive_path)
    assert not valid
    assert "size limit" in reason


def test_parser_rejection_excludes_user_paths_from_errors_and_logs(tmp_path, monkeypatch):
    from docmind.indexing import parser
    path = tmp_path / "sensitive-client-name.txt"
    path.write_text(" ")
    logs = []
    monkeypatch.setattr(parser.logger, "warning", lambda *args, **kwargs: logs.append((args, kwargs)))
    with pytest.raises(ValueError) as err:
        parser.parse_file(path)
    assert path.name not in str(err.value)
    assert all("path" not in kwargs for _args, kwargs in logs)
    assert all(path.name not in str(item) for item in logs)


def test_unhandled_errors_do_not_log_exception_text_or_traceback():
    import ast
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / 'src' / 'docmind' / 'main.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    candidates = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == 'error' and n.args
        and isinstance(n.args[0], ast.Constant)
        and n.args[0].value == 'unhandled_exception'
    ]
    assert len(candidates) == 1
    assert {kw.arg for kw in candidates[0].keywords} == {'error_type'}


def test_partial_upload_failures_do_not_log_original_exception_text():
    import inspect
    from docmind.services.ingestion import IngestionService
    source = inspect.getsource(IngestionService.ingest_file)
    assert 'safe_path.unlink(missing_ok=True)' in source
    assert 'logger.error("file_save_failed", error_type=type(exc).__name__)' in source


def test_parser_exception_is_redacted_instead_of_masked_by_logging(tmp_path, monkeypatch):
    """Regression: stdlib Logger.error cannot accept structured keyword fields."""
    from docmind.indexing import parser

    path = tmp_path / "private-client.txt"
    path.write_text("customer-only content", encoding="utf-8")

    def failing_parser(_path):
        raise RuntimeError("private upstream parser error")

    monkeypatch.setitem(parser.PARSERS, ".txt", failing_parser)
    with pytest.raises(RuntimeError, match="^Document parsing failed$") as error:
        parser.parse_file(path)
    assert "private upstream" not in str(error.value)


def test_parser_empty_document_reports_validation_error(tmp_path):
    """The warning logger must also support structured fields."""
    from docmind.indexing import parser

    path = tmp_path / "empty.txt"
    path.write_text("   ", encoding="utf-8")
    with pytest.raises(ValueError, match="Document contains no extractable text"):
        parser.parse_file(path)


@pytest.mark.parametrize("exception_type", [ValueError, RuntimeError])
def test_query_api_never_exposes_or_logs_private_provider_errors(monkeypatch, exception_type):
    from docmind.api import routes

    secret = "customer-private-query-provider-payload"
    observed = []

    async def unavailable(**_kwargs):
        raise exception_type(secret)

    monkeypatch.setattr(routes.query_service, "query", unavailable)
    monkeypatch.setattr(routes.logger, "warning", lambda *a, **k: observed.append((a, k)))
    monkeypatch.setattr(routes.logger, "error", lambda *a, **k: observed.append((a, k)))
    with TestClient(app) as client:
        result = client.post("/api/v1/query",
                             headers={"Authorization": "Bearer test-only-key"},
                             json={"query": "Periksa laporan insiden"})
    assert result.status_code == (400 if exception_type is ValueError else 500)
    assert secret not in result.text
    assert secret not in repr(observed)
    assert observed
    assert all("exc_info" not in kwargs for _, kwargs in observed)


@pytest.mark.parametrize("exception_type", [ValueError, RuntimeError])
def test_upload_api_never_exposes_or_logs_private_parser_errors(monkeypatch, exception_type):
    from docmind.api import routes

    secret = "customer-private-upload-path-or-document-content"
    observed = []

    async def broken(_file, _filename):
        raise exception_type(secret)

    monkeypatch.setattr(routes.ingestion_service, "ingest_file", broken)
    monkeypatch.setattr(routes.logger, "warning", lambda *a, **k: observed.append((a, k)))
    monkeypatch.setattr(routes.logger, "error", lambda *a, **k: observed.append((a, k)))
    with TestClient(app) as client:
        result = client.post("/api/v1/documents/upload",
                             headers={"Authorization": "Bearer test-only-key"},
                             files={"file": ("customer.txt", b"private-content", "text/plain")})
    assert result.status_code == (400 if exception_type is ValueError else 500)
    assert secret not in result.text
    assert secret not in repr(observed)
    assert observed
    assert all("exc_info" not in kwargs for _, kwargs in observed)


def test_unreadable_upload_magic_does_not_expose_storage_path(tmp_path, monkeypatch):
    from docmind.security import validation

    path = tmp_path / "private-client-name.pdf"
    path.write_bytes(b"%PDF" + b"placeholder")

    def denied(*_args, **_kwargs):
        raise OSError("private-client-name.pdf is unreadable")

    monkeypatch.setattr("builtins.open", denied)
    valid, reason = validation.validate_file_type(path)
    assert not valid
    assert "private-client-name" not in reason
    assert "could not be read" in reason


def test_query_provider_failure_is_not_a_successful_refusal(monkeypatch):
    """A failed provider must result in HTTP 503, not a fake 200 refusal."""
    from docmind.api import routes
    from docmind.services.query import LLMServiceUnavailable

    async def broken_provider(**_kwargs):
        raise LLMServiceUnavailable("secret upstream provider detail")

    monkeypatch.setattr(routes.query_service, "query", broken_provider)
    with TestClient(app) as client:
        result = client.post(
            "/api/v1/query",
            headers={"Authorization": "Bearer test-only-key"},
            json={"query": "Apa isi dokumen tersebut?"},
        )
    assert result.status_code == 503
    assert result.json() == {"detail": "Inference provider unavailable"}
    assert "secret upstream provider detail" not in result.text
