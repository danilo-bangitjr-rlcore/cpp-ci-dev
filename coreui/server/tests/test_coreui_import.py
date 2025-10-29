import sys
from pathlib import Path

from fastapi import FastAPI

# Add the server directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from server.core_ui import create_app


def test_create_app_import_and_call():
    """Test that create_app can be imported and called, returning a FastAPI app."""
    app = create_app()
    assert isinstance(app, FastAPI)
