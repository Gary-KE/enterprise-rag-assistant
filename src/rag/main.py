"""Production entry point: `uvicorn rag.main:app`. All settings come from environment variables."""
import logging

from rag.api import create_app

logging.basicConfig(level=logging.INFO, format="%(message)s")
app = create_app()
