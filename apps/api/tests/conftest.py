import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


TEST_DB = Path(__file__).parent / "test_touchline.db"
os.environ["TOUCHLINE_DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB.as_posix()}"

from app.main import app  # noqa: E402


@pytest.fixture
def client():
    if TEST_DB.exists():
        TEST_DB.unlink()
    with TestClient(app) as test_client:
        yield test_client
    if TEST_DB.exists():
        TEST_DB.unlink()

