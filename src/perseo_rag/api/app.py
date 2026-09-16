from fastapi import FastAPI

from perseo_rag import __version__
from perseo_rag.api.query import build_query_router
from perseo_rag.api.scope import ScopeResolver
from perseo_rag.application import QueryHandler
from perseo_rag.config import Settings, get_settings


def create_app(
    settings: Settings | None = None,
    *,
    query_handler: QueryHandler | None = None,
    scope_resolver: ScopeResolver | None = None,
) -> FastAPI:
    config = settings or get_settings()
    docs_url = "/docs" if config.expose_api_docs else None
    openapi_url = "/openapi.json" if config.expose_api_docs else None

    if (query_handler is None) != (scope_resolver is None):
        raise ValueError("query_handler and scope_resolver must be configured together")

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

    if query_handler is not None and scope_resolver is not None:
        app.include_router(build_query_router(query_handler, scope_resolver))

    return app


app = create_app()
