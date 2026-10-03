import os
import tempfile
from pathlib import Path

_test_dir = Path(tempfile.mkdtemp(prefix="docmind-tests-"))
os.environ.update({
    "DATA_DIR": str(_test_dir), "UPLOAD_DIR": str(_test_dir / "uploads"),
    "CHROMADB_DIR": str(_test_dir / "chroma"),
    "DATABASE_URL": f"sqlite+aiosqlite:///{_test_dir / 'test.db'}",
    "API_KEY": "test-only-key", "RATE_LIMIT_REQUESTS": "6000",
})
