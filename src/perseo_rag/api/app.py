from fastapi import FastAPI

from perseo_rag import __version__
from perseo_rag.config import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings or get_settings()
    docs_url = "/docs" if config.expose_api_docs else None
    openapi_url = "/openapi.json" if config.expose_api_docs else None

    app = FastAPI(
        title="Perseo RAG",
        version=__version__,
        docs_url=docs_url,
        redoc_url=None,
        openapi_url=openapi_url,
    )

    @app.get("/health", include_in_schema=False)
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
