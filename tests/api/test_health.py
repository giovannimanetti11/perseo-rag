from fastapi.testclient import TestClient

from perseo_rag.api import create_app
from perseo_rag.config import Settings


def test_health_check() -> None:
    app = create_app(Settings(_env_file=None))
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_api_docs_are_disabled_by_default() -> None:
    app = create_app(Settings(_env_file=None))
    client = TestClient(app)

    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404
