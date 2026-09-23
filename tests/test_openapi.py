import json
import os
from pathlib import Path

from vadsa.config import Settings
from vadsa.main import create_app

COMMITTED = Path(__file__).parent.parent / "openapi.json"


def test_committed_openapi_matches_the_app() -> None:
    """After an intended API change: UPDATE_OPENAPI=1 uv run pytest tests/test_openapi.py"""
    schema = create_app(Settings(environment="development", engine="fake")).openapi()
    if os.environ.get("UPDATE_OPENAPI"):
        COMMITTED.write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n")
    assert json.loads(COMMITTED.read_text()) == schema, "openapi.json is out of date"
